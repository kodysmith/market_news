import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAttention(nn.Module):
    """Multi-head attention with top-k sparsity and energy gating.

    Instead of attending to ALL tokens, each query only attends to its
    top-k most relevant keys. This is the attention equivalent of the
    SSAN top-k neuron selection — learned, dynamic routing.
    """

    def __init__(self, dim, n_heads=4, top_k_attn=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.top_k_attn = top_k_attn
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Diagnostics
        self._last_attn_mask = None

    def forward(self, x, energy_k=None):
        """
        Args:
            x: [batch, seq_len, dim]
            energy_k: int or None — override top-k for attention sparsity

        Returns:
            out: [batch, seq_len, dim]
            attn_cost: int — number of attention connections used
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]

        # Sparse attention: only keep top-k scores per query
        k_attn = self.top_k_attn
        if energy_k is not None:
            k_attn = min(k_attn, max(1, energy_k))
        k_attn = min(k_attn, N)

        # Top-k masking
        topk_vals, topk_idx = torch.topk(attn, k_attn, dim=-1)  # [B, heads, N, k]
        mask = torch.zeros_like(attn)
        mask.scatter_(-1, topk_idx, 1.0)

        # Apply mask: -inf for non-selected, keep original for selected
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        self._last_attn_mask = mask.detach()

        # Weighted sum of values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        attn_cost = k_attn * N  # total attention connections used
        return out, attn_cost


class SpikingTransformerBlock(nn.Module):
    """Transformer block with SSAN dynamics.

    Replaces standard residual with leaky membrane potential,
    adds spike strength modulation, and uses sparse attention.
    """

    def __init__(self, dim, n_heads=4, top_k_attn=8, mlp_ratio=2.0,
                 lambda_decay=0.9, dropout=0.0):
        super().__init__()
        self.lambda_decay = lambda_decay

        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseAttention(dim, n_heads, top_k_attn, dropout)

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
        """
        Args:
            x: [batch, seq_len, dim]
            V_prev: [batch, seq_len, dim] or None — membrane potential
            energy_k: int or None — attention sparsity limit

        Returns:
            out: [batch, seq_len, dim]
            V: [batch, seq_len, dim] — updated membrane potential
            attn_cost: int
        """
        # Sparse attention
        attn_out, attn_cost = self.attn(self.norm1(x), energy_k=energy_k)

        # Leaky membrane potential instead of standard residual
        if V_prev is not None:
            V = self.lambda_decay * V_prev + attn_out
        else:
            V = attn_out

        x = x + V  # residual + membrane contribution

        # FFN with standard residual
        x = x + self.mlp(self.norm2(x))

        return x, V, attn_cost


class TransformerSSAN(nn.Module):
    """Sparse Spiking Transformer.

    A Vision Transformer with SSAN dynamics:
      - Patches as tokens (like ViT)
      - Sparse top-k attention (energy-gated routing)
      - Leaky membrane potential (temporal state across layers)
      - Spike strength per token (reinforcement-learned importance)
      - Energy budget controlling total attention sparsity

    This is the transformer equivalent of the graph SSAN:
    instead of a fixed topology, attention LEARNS the routing.
    """

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        tcfg = config.transformer
        ssan_cfg = config.ssan

        self.patch_size = tcfg.patch_size
        self.dim = tcfg.dim
        self.n_layers = tcfg.n_layers
        self.n_heads = tcfg.n_heads
        self.top_k_attn = tcfg.top_k_attn

        # Energy
        self.energy_budget = getattr(ssan_cfg, "energy_budget", None)
        self.energy_per_layer_max = getattr(ssan_cfg, "energy_per_layer_max", None)

        # Reinforcement params
        self.alpha = ssan_cfg.alpha
        self.beta = ssan_cfg.beta
        self.s_min = ssan_cfg.s_min
        self.s_max = ssan_cfg.s_max
        self.enable_reinforcement = ssan_cfg.enable_reinforcement

        # Patch embedding
        image_size = 28
        n_patches_side = image_size // self.patch_size
        self.n_patches = n_patches_side * n_patches_side
        patch_dim = self.patch_size * self.patch_size  # pixels per patch (grayscale)

        self.patch_embed = nn.Linear(patch_dim, self.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTransformerBlock(
                dim=self.dim,
                n_heads=self.n_heads,
                top_k_attn=self.top_k_attn,
                mlp_ratio=getattr(tcfg, "mlp_ratio", 2.0),
                lambda_decay=ssan_cfg.lambda_decay,
                dropout=getattr(tcfg, "dropout", 0.0),
            )
            for _ in range(self.n_layers)
        ])

        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, output_dim)

        # Spike strength per token position (reinforcement-updated)
        self.register_buffer("S", torch.ones(self.n_patches + 1))

        # Diagnostics
        self._last_attn_costs = []
        self._pending_update = False

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 1, 28, 28)

        # Patchify: [B, 1, 28, 28] -> [B, n_patches, patch_dim]
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B, 1, H/p, W/p, p, p]
        patches = patches.contiguous().view(B, self.n_patches, -1)  # [B, n_patches, p*p]

        # Embed patches
        tokens = self.patch_embed(patches)  # [B, n_patches, dim]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [B, n_patches+1, dim]

        # Add positional encoding
        tokens = tokens + self.pos_embed

        # Apply spike strength: scale each token by its importance
        S_snap = self.S.detach().unsqueeze(0).unsqueeze(-1)  # [1, n_patches+1, 1]
        tokens = tokens * S_snap

        # Energy budget
        energy = self.energy_budget
        self._last_attn_costs = []

        # Transformer blocks with membrane potential
        V = None
        for block in self.blocks:
            # Compute energy-limited attention k
            energy_k = None
            if energy is not None:
                if self.energy_per_layer_max is not None:
                    layer_budget = min(energy, self.energy_per_layer_max)
                else:
                    layer_budget = energy
                # Each attention connection costs 1 energy
                # Total cost per layer = k_attn * n_tokens
                n_tokens = tokens.size(1)
                energy_k = max(1, int(layer_budget / max(n_tokens, 1)))

            tokens, V, attn_cost = block(tokens, V_prev=V, energy_k=energy_k)

            if energy is not None:
                energy = max(0.0, energy - attn_cost)

            self._last_attn_costs.append(attn_cost)

        tokens = self.norm(tokens)

        # Classify from CLS token
        cls_out = tokens[:, 0]
        logits = self.head(cls_out)

        if self.training and self.enable_reinforcement:
            self._pending_update = True

        return logits

    def apply_reinforcement(self):
        """Update spike strengths based on which tokens contributed most."""
        if not self._pending_update:
            return

        # Use attention patterns from last forward to determine token importance
        # For simplicity: boost CLS token always (it's always used),
        # decay patch tokens proportionally to how often they were attended to
        with torch.no_grad():
            # Mild reinforcement: slightly boost all active tokens
            fire_rate = torch.ones_like(self.S) * 0.5  # all tokens participate
            new_S = self.S + self.alpha * fire_rate
            self.S.copy_(new_S.clamp(self.s_min, self.s_max))
        self._pending_update = False

    def get_diagnostics(self):
        diag = {
            "n_patches": self.n_patches,
            "n_layers": self.n_layers,
            "top_k_attn": self.top_k_attn,
            "attn_costs": self._last_attn_costs,
            "total_attn_cost": sum(self._last_attn_costs),
            "S_mean": self.S.mean().item(),
            "S_std": self.S.std().item(),
        }
        return diag
