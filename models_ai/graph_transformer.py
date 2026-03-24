import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import build_adjacency


class GraphAttention(nn.Module):
    """Multi-head attention constrained by graph topology.

    Tokens can ONLY attend to their graph neighbors. The graph defines
    which connections exist; attention learns the weights. This combines
    structural sparsity (graph) with learned routing (attention).

    Compared to full attention (N^2), this computes only E attention
    scores where E = number of edges. For a sparse graph, E << N^2.
    """

    def __init__(self, dim, n_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self._last_attn = None

    def forward(self, x, graph_mask):
        """
        Args:
            x: [batch, n_tokens, dim]
            graph_mask: [n_tokens, n_tokens] binary — 1 where attention is allowed

        Returns:
            out: [batch, n_tokens, dim]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Attention scores: [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Graph mask: block attention where no edge exists
        # graph_mask is [N, N], expand to [1, 1, N, N] for broadcasting
        attn_mask = graph_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # NaN guard: if a token has NO neighbors, softmax produces NaN
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        self._last_attn = attn.detach()

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class GraphTransformerBlock(nn.Module):
    """Transformer block with graph-constrained attention and SSAN dynamics."""

    def __init__(self, dim, n_heads=4, mlp_ratio=2.0, lambda_decay=0.9, dropout=0.0):
        super().__init__()
        self.lambda_decay = lambda_decay

        self.norm1 = nn.LayerNorm(dim)
        self.attn = GraphAttention(dim, n_heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, graph_mask, V_prev=None):
        """
        Returns:
            out: [batch, n_tokens, dim]
            V: updated membrane potential
        """
        attn_out = self.attn(self.norm1(x), graph_mask)

        # Leaky membrane potential
        if V_prev is not None:
            V = self.lambda_decay * V_prev + attn_out
        else:
            V = attn_out

        x = x + V
        x = x + self.mlp(self.norm2(x))
        return x, V


class GraphTransformerSSAN(nn.Module):
    """Graph Transformer with SSAN dynamics.

    Combines:
      - Graph topology: defines WHICH tokens can communicate
      - Attention: learns HOW MUCH they communicate
      - Membrane potential: temporal state across layers
      - Spike strength: reinforcement-learned token importance

    The graph constrains the attention pattern. A token at position i
    can only attend to position j if there's an edge (i,j) in the graph.
    This means:
      - Nearby patches communicate (local edges)
      - Skip connections allow long-range communication (skip edges)
      - The CLS token connects to all patches (hub node)
      - The total attention cost is proportional to edges, not N^2
    """

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        tcfg = config.transformer
        gcfg = config.graph
        ssan_cfg = config.ssan

        self.patch_size = tcfg.patch_size
        self.dim = tcfg.dim
        self.n_layers = tcfg.n_layers
        self.n_heads = tcfg.n_heads

        # Reinforcement
        self.alpha = ssan_cfg.alpha
        self.beta = ssan_cfg.beta
        self.s_min = ssan_cfg.s_min
        self.s_max = ssan_cfg.s_max
        self.enable_reinforcement = ssan_cfg.enable_reinforcement

        # Patch layout
        image_size = 28
        n_patches_side = image_size // self.patch_size
        self.n_patches = n_patches_side * n_patches_side
        self.n_tokens = self.n_patches + 1  # +1 for CLS
        patch_dim = self.patch_size * self.patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, self.dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim) * 0.02)

        # Build graph over tokens
        # Patches are arranged in a grid. Build adjacency based on spatial proximity
        # + CLS token connects to everything.
        graph_mask = self._build_token_graph(
            n_patches_side, self.n_tokens,
            graph_type=getattr(gcfg, "graph_type", "spatial"),
            edge_density=gcfg.edge_density,
            seed=config.seed,
        )
        self.register_buffer("graph_mask", graph_mask)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                dim=self.dim,
                n_heads=self.n_heads,
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

    def _build_token_graph(self, grid_size, n_tokens, graph_type="spatial",
                           edge_density=0.3, seed=42):
        """Build adjacency over token positions.

        Token 0 = CLS (connects to all patches)
        Tokens 1..n_patches = patches in row-major grid order
        """
        n = n_tokens
        mask = torch.zeros(n, n)

        # CLS token (index 0) connects to ALL patches and itself
        mask[0, :] = 1.0
        mask[:, 0] = 1.0

        if graph_type == "spatial":
            # Patches connect to spatial neighbors (4-connected grid + diagonals)
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = 1 + i * grid_size + j  # +1 for CLS offset
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                                nidx = 1 + ni * grid_size + nj
                                mask[idx, nidx] = 1.0

            # Add sparse long-range connections
            gen = torch.Generator().manual_seed(seed)
            for i in range(1, n):
                for j in range(i + 1, n):
                    if mask[i, j] == 0 and torch.rand(1, generator=gen).item() < edge_density * 0.3:
                        mask[i, j] = 1.0
                        mask[j, i] = 1.0

        elif graph_type == "full":
            mask = torch.ones(n, n)

        else:
            # Use the standard graph builder for non-spatial types
            adj = build_adjacency(n, graph_type=graph_type, edge_density=edge_density, seed=seed)
            mask[1:, 1:] = adj[:n-1, :n-1]
            mask[0, :] = 1.0
            mask[:, 0] = 1.0

        return mask

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 1, 28, 28)

        # Patchify
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, self.n_patches, -1)

        # Embed
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        # Spike strength scaling
        S_snap = self.S.detach().unsqueeze(0).unsqueeze(-1)
        tokens = tokens * S_snap

        # Transformer blocks with graph-constrained attention
        V = None
        for block in self.blocks:
            tokens, V = block(tokens, self.graph_mask, V_prev=V)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])  # CLS token

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
        n_edges = int(self.graph_mask.sum().item())
        n_possible = self.n_tokens * self.n_tokens
        diag = {
            "n_tokens": self.n_tokens,
            "n_edges": n_edges,
            "n_possible_edges": n_possible,
            "edge_density": n_edges / n_possible,
            "attn_savings": 1.0 - n_edges / n_possible,
            "S_mean": self.S.mean().item(),
            "S_std": self.S.std().item(),
        }
        return diag
