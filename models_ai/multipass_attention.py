import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkCondenser(nn.Module):
    """Pass 1: Sliding window that condenses chunks into summary vectors.

    Divides the sequence into non-overlapping chunks of size C.
    Each chunk is compressed to a single summary vector via learned pooling.
    """

    def __init__(self, dim, chunk_size=64):
        super().__init__()
        self.chunk_size = chunk_size
        # Learned query that pools each chunk into a summary
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pool_kv = nn.Linear(dim, dim * 2)
        self.scale = (dim) ** -0.5

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, dim]

        Returns:
            summaries: [batch, n_chunks, dim]
            chunk_boundaries: list of (start, end) tuples
        """
        B, N, D = x.shape
        C = self.chunk_size

        # Pad sequence to be divisible by chunk_size
        pad = (C - N % C) % C
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        n_chunks = x.shape[1] // C

        # Reshape into chunks: [B, n_chunks, C, D]
        chunks = x.view(B, n_chunks, C, D)

        # Pool each chunk: query attends to chunk tokens
        # query: [1, 1, D] -> [B, n_chunks, 1, D]
        q = self.pool_query.expand(B, n_chunks, -1, -1)

        # kv from chunk tokens: [B, n_chunks, C, 2*D]
        kv = self.pool_kv(chunks)
        k, v = kv.chunk(2, dim=-1)  # each [B, n_chunks, C, D]

        # Attention: [B, n_chunks, 1, C]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: [B, n_chunks, 1, D] -> [B, n_chunks, D]
        summaries = (attn @ v).squeeze(-2)

        boundaries = [(i * C, min(i * C + C, N)) for i in range(n_chunks)]

        return summaries, boundaries


class MultiPassAttention(nn.Module):
    """Two-pass attention: condense then focus.

    Pass 1 (Condense): Sliding window compresses sequence into chunk summaries.
    Pass 2 (Select): Each query scores against summaries, picks top-k chunks.
    Pass 3 (Focus): Full attention only within the selected chunks.

    Cost: O(N*C) + O(N*N/C) + O(N*K*C) instead of O(N²)
    For N=128K, C=256, K=8: ~50x cheaper than full attention.
    """

    def __init__(self, dim, n_heads=4, chunk_size=64, top_k_chunks=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        self.scale = self.head_dim ** -0.5

        # Pass 1: chunk condenser
        self.condenser = ChunkCondenser(dim, chunk_size)

        # Pass 2: summary scoring (lightweight)
        self.summary_query = nn.Linear(dim, dim)
        self.summary_key = nn.Linear(dim, dim)

        # Pass 3: focused attention (standard QKV on selected chunks)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Diagnostics
        self._last_selected_chunks = None
        self._last_chunk_scores = None

    def forward(self, x, energy_k=None):
        """
        Args:
            x: [batch, seq_len, dim]
            energy_k: int or None — override top-k chunks

        Returns:
            out: [batch, seq_len, dim]
            attn_cost: int — effective attention operations
        """
        B, N, D = x.shape

        # === PASS 1: Condense ===
        summaries, boundaries = self.condenser(x)  # [B, n_chunks, D]
        n_chunks = summaries.shape[1]

        # === PASS 2: Score & Select ===
        # Each token scores against chunk summaries to find relevant chunks
        # Use a lightweight projection for scoring
        sq = self.summary_query(x)          # [B, N, D]
        sk = self.summary_key(summaries)    # [B, n_chunks, D]

        # Chunk relevance scores: [B, N, n_chunks]
        chunk_scores = torch.bmm(sq, sk.transpose(1, 2)) * (D ** -0.5)

        # Select top-k chunks per token
        k = min(self.top_k_chunks, n_chunks)
        if energy_k is not None:
            k = min(k, max(1, energy_k))

        # For efficiency: select top-k chunks globally (same for all tokens in batch)
        # Average across tokens to get per-chunk importance
        chunk_importance = chunk_scores.mean(dim=1)  # [B, n_chunks]
        _, selected_idx = torch.topk(chunk_importance, k, dim=1)  # [B, k]

        self._last_selected_chunks = selected_idx.detach()
        self._last_chunk_scores = chunk_importance.detach()

        # === PASS 3: Focused Attention ===
        # Gather tokens from selected chunks, pad to uniform length
        C = self.chunk_size
        L = k * C  # max possible length

        gathered_tokens = []
        attn_pad_masks = []  # 1 = real token, 0 = padding

        for b in range(B):
            tokens_for_b = []
            mask_for_b = []
            for ci in selected_idx[b]:
                start, end = boundaries[ci.item()]
                chunk_tokens = x[b, start:end]  # [chunk_len, D]
                chunk_len = chunk_tokens.shape[0]
                # Pad short chunks
                if chunk_len < C:
                    pad = torch.zeros(C - chunk_len, D, device=x.device)
                    chunk_tokens = torch.cat([chunk_tokens, pad], dim=0)
                    mask_for_b.extend([1.0] * chunk_len + [0.0] * (C - chunk_len))
                else:
                    mask_for_b.extend([1.0] * C)
                tokens_for_b.append(chunk_tokens)
            gathered_tokens.append(torch.cat(tokens_for_b, dim=0))
            attn_pad_masks.append(torch.tensor(mask_for_b, device=x.device))

        kv_tokens = torch.stack(gathered_tokens, dim=0)  # [B, L, D]
        pad_mask = torch.stack(attn_pad_masks, dim=0)    # [B, L]

        # Full QKV attention: queries = all tokens, keys/values = selected chunks
        q = self.qkv(x)[:, :, :D]  # just Q projection for queries
        q = q.reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        kv = self.qkv(kv_tokens)[:, :, D:]  # K and V projections
        kv = kv.reshape(B, L, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k_focused, v_focused = kv.unbind(0)  # [B, heads, L, head_dim]

        # Attention: [B, heads, N, L]
        attn = (q @ k_focused.transpose(-2, -1)) * self.scale
        # Mask out padded positions
        pad_attn_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        attn = attn.masked_fill(pad_attn_mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)  # guard against all-masked rows
        attn = self.dropout(attn)

        out = (attn @ v_focused).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)

        # Attention cost: pass1 (N*C) + pass2 (N*n_chunks) + pass3 (N*k*C)
        attn_cost = N * self.chunk_size + N * n_chunks + N * k * self.chunk_size

        return out, attn_cost


class MultiPassTransformerBlock(nn.Module):
    """Transformer block using multi-pass attention with SSAN dynamics."""

    def __init__(self, dim, n_heads=4, chunk_size=64, top_k_chunks=4,
                 mlp_ratio=2.0, lambda_decay=0.9, dropout=0.0):
        super().__init__()
        self.lambda_decay = lambda_decay

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiPassAttention(dim, n_heads, chunk_size, top_k_chunks, dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, V_prev=None, energy_k=None):
        attn_out, attn_cost = self.attn(self.norm1(x), energy_k=energy_k)

        if V_prev is not None:
            V = self.lambda_decay * V_prev + attn_out
        else:
            V = attn_out

        x = x + V
        x = x + self.mlp(self.norm2(x))
        return x, V, attn_cost


class MultiPassTransformerSSAN(nn.Module):
    """Sparse Spiking Transformer with Multi-Pass Attention.

    Two-pass attention mechanism:
      Pass 1: Cheap sliding window condenses sequence into chunk summaries
      Pass 2: Each token scores summaries, selects top-k relevant chunks
      Pass 3: Full attention only on selected chunks

    Cost: O(N×C + N×N/C + N×K×C) instead of O(N²)

    Combined with SSAN dynamics:
      - Leaky membrane potential across layers
      - Spike strength per token (reinforcement-learned importance)
      - Energy budget controlling how many chunks are attended to
    """

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        tcfg = config.transformer
        ssan_cfg = config.ssan

        self.patch_size = tcfg.patch_size
        self.dim = tcfg.dim
        self.n_layers = tcfg.n_layers

        chunk_size = getattr(tcfg, "chunk_size", 4)
        top_k_chunks = getattr(tcfg, "top_k_chunks", 4)

        # Reinforcement
        self.alpha = ssan_cfg.alpha
        self.beta = ssan_cfg.beta
        self.s_min = ssan_cfg.s_min
        self.s_max = ssan_cfg.s_max
        self.enable_reinforcement = ssan_cfg.enable_reinforcement

        # Energy
        self.energy_budget = getattr(ssan_cfg, "energy_budget", None)
        self.energy_per_layer_max = getattr(ssan_cfg, "energy_per_layer_max", None)

        # Patch embedding
        image_size = 28
        n_patches_side = image_size // self.patch_size
        self.n_patches = n_patches_side * n_patches_side
        self.n_tokens = self.n_patches + 1
        patch_dim = self.patch_size * self.patch_size

        self.patch_embed = nn.Linear(patch_dim, self.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

        # Multi-pass transformer blocks
        self.blocks = nn.ModuleList([
            MultiPassTransformerBlock(
                dim=self.dim,
                n_heads=tcfg.n_heads,
                chunk_size=chunk_size,
                top_k_chunks=top_k_chunks,
                mlp_ratio=getattr(tcfg, "mlp_ratio", 2.0),
                lambda_decay=ssan_cfg.lambda_decay,
                dropout=getattr(tcfg, "dropout", 0.0),
            )
            for _ in range(self.n_layers)
        ])

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, output_dim)

        # Spike strength per token
        self.register_buffer("S", torch.ones(self.n_tokens))
        self._pending_update = False
        self._last_attn_costs = []

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 1, 28, 28)

        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, self.n_patches, -1)

        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        # Spike strength scaling
        S_snap = self.S.detach().unsqueeze(0).unsqueeze(-1)
        tokens = tokens * S_snap

        energy = self.energy_budget
        self._last_attn_costs = []

        V = None
        for block in self.blocks:
            energy_k = None
            if energy is not None:
                if self.energy_per_layer_max is not None:
                    layer_budget = min(energy, self.energy_per_layer_max)
                else:
                    layer_budget = energy
                energy_k = max(1, int(layer_budget / max(self.n_tokens, 1)))

            tokens, V, attn_cost = block(tokens, V_prev=V, energy_k=energy_k)

            if energy is not None:
                energy = max(0.0, energy - attn_cost)

            self._last_attn_costs.append(attn_cost)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if self.training and self.enable_reinforcement:
            self._pending_update = True

        return logits

    def apply_reinforcement(self):
        if not self._pending_update:
            return
        with torch.no_grad():
            new_S = self.S + self.alpha * 0.5
            self.S.copy_(new_S.clamp(self.s_min, self.s_max))
        self._pending_update = False

    def get_diagnostics(self):
        full_attn_cost = self.n_tokens * self.n_tokens * self.n_layers
        actual_cost = sum(self._last_attn_costs) if self._last_attn_costs else 0

        diag = {
            "n_tokens": self.n_tokens,
            "n_layers": self.n_layers,
            "attn_costs_per_layer": self._last_attn_costs,
            "total_attn_cost": actual_cost,
            "full_attn_cost": full_attn_cost,
            "attn_savings": 1.0 - actual_cost / max(full_attn_cost, 1),
            "S_mean": self.S.mean().item(),
        }
        return diag
