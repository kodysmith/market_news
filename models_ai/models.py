import torch
import torch.nn as nn

from .layers import SSANLayer


class BaselineMLP(nn.Module):
    """Standard MLP with ReLU activations."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, C, H, W] or [batch, input_dim]
        x = x.view(x.size(0), -1)
        return self.net(x)


class SparseMLP(nn.Module):
    """MLP with top-k sparsity and soft relaxation, but no temporal memory
    or reinforcement. Isolates the sparsity contribution."""

    def __init__(self, input_dim, hidden_dims, output_dim, top_k=128,
                 sharpness=10.0, theta=0.0):
        super().__init__()
        self.top_k = top_k
        self.sharpness = sharpness
        self.theta = theta

        self.layers = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.classifier = nn.Linear(prev, output_dim)

        # For diagnostics
        self._last_masks = []

    def forward(self, x):
        x = x.view(x.size(0), -1)
        self._last_masks = []

        for linear in self.layers:
            h = linear(x)
            soft = torch.sigmoid((h - self.theta) * self.sharpness)

            k = min(self.top_k, h.shape[1])
            _, topk_idx = torch.topk(h, k, dim=1)
            mask = torch.zeros_like(h)
            mask.scatter_(1, topk_idx, 1.0)

            x = soft * mask
            self._last_masks.append(mask.detach())

        return self.classifier(x)


class SSANModel(nn.Module):
    """Full Sparse Spiking Adaptive Network with temporal processing."""

    def __init__(self, input_dim, output_dim, config):
        super().__init__()
        self.temporal_mode = config.data.temporal_mode
        self.num_patches = getattr(config.data, "num_patches", 4)
        self.image_h = 28
        self.image_w = 28

        # Energy system
        self.energy_budget = getattr(config.ssan, "energy_budget", None)

        # Determine step input dim based on temporal mode
        if self.temporal_mode == "single":
            step_dim = input_dim  # 784
        elif self.temporal_mode == "rows":
            step_dim = self.image_w  # 28
        elif self.temporal_mode == "patches":
            step_dim = input_dim // self.num_patches
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

        self.step_dim = step_dim
        hidden_dims = config.model.hidden_dims

        # Build SSAN layers
        self.ssan_layers = nn.ModuleList()
        prev = step_dim
        for h in hidden_dims:
            self.ssan_layers.append(SSANLayer(prev, h, config.ssan))
            prev = h

        self.classifier = nn.Linear(prev, output_dim)

    def _prepare_temporal(self, x):
        """Convert input images into a list of timestep tensors.

        Args:
            x: [batch, C, H, W]

        Returns:
            List of [batch, step_dim] tensors, one per timestep.
        """
        batch = x.size(0)

        if self.temporal_mode == "single":
            return [x.view(batch, -1)]
        elif self.temporal_mode == "rows":
            # [batch, 1, 28, 28] -> 28 steps of [batch, 28]
            x = x.view(batch, self.image_h, self.image_w)
            return [x[:, t, :] for t in range(self.image_h)]
        elif self.temporal_mode == "patches":
            flat = x.view(batch, -1)  # [batch, 784]
            chunks = flat.chunk(self.num_patches, dim=1)
            return list(chunks)
        else:
            raise ValueError(f"Unknown temporal_mode: {self.temporal_mode}")

    def forward(self, x):
        timesteps = self._prepare_temporal(x)

        # Initialize membrane potentials
        V_states = [None] * len(self.ssan_layers)

        for x_t in timesteps:
            h = x_t
            new_V = []
            # Reset energy budget each timestep
            energy = self.energy_budget
            for i, layer in enumerate(self.ssan_layers):
                h, v, mask, energy = layer(h, V_states[i], energy=energy)
                new_V.append(v)
            V_states = new_V

        # Classify from final hidden state at last timestep
        logits = self.classifier(h)
        return logits


def build_model(config, input_dim, output_dim):
    """Factory function to build the configured model."""
    model_type = config.model.type
    hidden_dims = config.model.hidden_dims
    dropout = getattr(config.model, "dropout", 0.0)

    if model_type == "mlp":
        return BaselineMLP(input_dim, hidden_dims, output_dim, dropout=dropout)
    elif model_type == "sparse_mlp":
        top_k = config.ssan.top_k
        if config.ssan.top_k_ratio is not None:
            top_k = max(1, int(config.ssan.top_k_ratio * hidden_dims[0]))
        return SparseMLP(
            input_dim, hidden_dims, output_dim,
            top_k=top_k,
            sharpness=config.ssan.sharpness,
            theta=config.ssan.theta,
        )
    elif model_type == "ssan":
        return SSANModel(input_dim, output_dim, config)
    elif model_type == "graph_ssan":
        from .graph import GraphSSAN
        return GraphSSAN(input_dim, output_dim, config)
    elif model_type == "transformer_ssan":
        from .transformer import TransformerSSAN
        return TransformerSSAN(input_dim, output_dim, config)
    elif model_type == "graph_transformer_ssan":
        from .graph_transformer import GraphTransformerSSAN
        return GraphTransformerSSAN(input_dim, output_dim, config)
    elif model_type == "multipass_ssan":
        from .multipass_attention import MultiPassTransformerSSAN
        return MultiPassTransformerSSAN(input_dim, output_dim, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
