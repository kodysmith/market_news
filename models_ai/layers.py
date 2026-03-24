import torch
import torch.nn as nn


class SSANLayer(nn.Module):
    """Sparse Spiking Adaptive Network layer.

    Replaces Linear + Activation with:
      1. Leaky integration of membrane potential
      2. Soft sigmoid relaxation (differentiable spike proxy)
      3. Top-k sparse activation mask
      4. Spike-strength modulated output with energy-ranked scaling
      5. Reinforcement-based spike strength updates
    """

    def __init__(self, input_dim, output_dim, ssan_cfg):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

        # SSAN parameters
        self.lambda_decay = ssan_cfg.lambda_decay
        self.theta = ssan_cfg.theta
        self.sharpness = ssan_cfg.sharpness
        self.alpha = ssan_cfg.alpha
        self.beta = ssan_cfg.beta
        self.s_min = ssan_cfg.s_min
        self.s_max = ssan_cfg.s_max

        # Resolve top_k
        if ssan_cfg.top_k_ratio is not None:
            self.top_k = max(1, int(ssan_cfg.top_k_ratio * output_dim))
        else:
            self.top_k = min(ssan_cfg.top_k, output_dim)

        # Ablation flags
        self.enable_sparsity = ssan_cfg.enable_sparsity
        self.enable_memory = ssan_cfg.enable_memory
        self.enable_reinforcement = ssan_cfg.enable_reinforcement

        # Energy system config
        self.energy_per_layer_max = getattr(ssan_cfg, "energy_per_layer_max", None)
        self.energy_decay_per_neuron = getattr(ssan_cfg, "energy_decay_per_neuron", 0.0)

        # Sparsity fixes
        # Leak: non-active neurons contribute a tiny signal (keeps gradients alive)
        self.sparse_leak = getattr(ssan_cfg, "sparse_leak", 0.0)
        # Straight-through: during backward, gradients flow through ALL neurons
        self.straight_through = getattr(ssan_cfg, "straight_through", False)
        # Sparsity warmup: start training with more neurons active, anneal down
        self.sparsity_warmup_epochs = getattr(ssan_cfg, "sparsity_warmup_epochs", 0)
        self._current_epoch = 0

        # Spike strength buffer (shared across batch, not a learned parameter)
        self.register_buffer("S", torch.full((output_dim,), ssan_cfg.s_init))

        # Diagnostics storage (updated each forward pass)
        self._last_mask = None
        self._last_V = None
        self._last_grad_norm = None
        self._last_energy_remaining = None
        self._last_neurons_fired = 0
        self._last_energy_cost = 0.0

    def forward(self, x, V_prev=None, energy=None):
        """
        Args:
            x: [batch, input_dim]
            V_prev: [batch, output_dim] or None
            energy: float or None — remaining energy budget.

        Returns:
            y: [batch, output_dim] sparse output
            V: [batch, output_dim] updated membrane potential
            mask: [batch, output_dim] binary activation mask
            energy_remaining: float or None
        """
        # Leaky integration
        drive = self.linear(x)

        if self.enable_memory and V_prev is not None:
            V = self.lambda_decay * V_prev + drive
        else:
            V = drive

        # Soft relaxation (differentiable spike proxy)
        soft_spike = torch.sigmoid((V - self.theta) * self.sharpness)

        # --- Determine how many neurons to fire ---
        if self.enable_sparsity:
            k = min(self.top_k, V.shape[1])

            # Sparsity warmup: start with more neurons, anneal to target top_k
            if self.sparsity_warmup_epochs > 0 and self._current_epoch < self.sparsity_warmup_epochs:
                progress = self._current_epoch / self.sparsity_warmup_epochs
                # Start at 100% of neurons, anneal to top_k
                warmup_k = int(self.output_dim - progress * (self.output_dim - self.top_k))
                k = min(warmup_k, V.shape[1])

            if energy is not None:
                # Per-layer cap: don't let this layer eat more than its share
                if self.energy_per_layer_max is not None:
                    layer_budget = min(energy, self.energy_per_layer_max)
                else:
                    layer_budget = energy

                # Diminishing energy: neuron i costs 1 + i * decay_rate
                if self.energy_decay_per_neuron > 0:
                    k_energy = _max_neurons_for_budget(
                        layer_budget, self.energy_decay_per_neuron
                    )
                else:
                    k_energy = max(1, int(layer_budget))
                k = min(k, k_energy)
                k = max(1, k)

            _, topk_idx = torch.topk(V, k, dim=1)
            mask = torch.zeros_like(V)
            mask.scatter_(1, topk_idx, 1.0)
        else:
            k = self.output_dim
            topk_idx = None
            mask = torch.ones_like(V)

        # --- Compute output ---
        S_snap = self.S.detach().unsqueeze(0)

        # Energy-ranked scaling (only if decay is enabled)
        if self.enable_sparsity and self.energy_decay_per_neuron > 0 and topk_idx is not None:
            rank_scales = torch.linspace(1.0, 1.0 / k, k, device=V.device)
            energy_scale = torch.zeros_like(V)
            rank_scales_expanded = rank_scales.unsqueeze(0).expand(V.shape[0], -1)
            energy_scale.scatter_(1, topk_idx, rank_scales_expanded)
            y = soft_spike * S_snap * mask * energy_scale
        elif self.straight_through and self.training and self.enable_sparsity:
            # Straight-through estimator: forward uses hard mask,
            # backward pretends mask wasn't there (gradients flow to all neurons)
            # y = soft_spike * S * mask  in forward
            # dy/d(soft_spike) = S * 1   in backward (mask removed from gradient)
            y_full = soft_spike * S_snap
            y_masked = y_full * mask
            # Trick: y = y_masked in forward, but gradient of y_full in backward
            y = y_full + (y_masked - y_full).detach()
        else:
            y = soft_spike * S_snap * mask

        # Sparse leak: non-active neurons contribute a tiny signal
        if self.sparse_leak > 0 and self.enable_sparsity:
            leak = soft_spike * S_snap * (1.0 - mask) * self.sparse_leak
            y = y + leak

        # --- Compute energy cost ---
        energy_remaining = None
        energy_cost = 0.0
        if energy is not None:
            if self.energy_decay_per_neuron > 0:
                # Sum of costs: 1 + (1+d) + (1+2d) + ... for k neurons
                d = self.energy_decay_per_neuron
                energy_cost = k + d * k * (k - 1) / 2.0
            else:
                energy_cost = float(k)

            # Cap by per-layer max
            if self.energy_per_layer_max is not None:
                energy_cost = min(energy_cost, self.energy_per_layer_max)

            energy_remaining = max(0.0, energy - energy_cost)

        # Store for diagnostics
        self._last_mask = mask.detach()
        self._last_V = V.detach()
        self._last_energy_remaining = energy_remaining
        self._last_neurons_fired = k
        self._last_energy_cost = energy_cost

        # Queue reinforcement update
        if self.training and self.enable_reinforcement:
            self._pending_mask = mask.detach()
        else:
            self._pending_mask = None

        return y, V, mask, energy_remaining

    def set_epoch(self, epoch):
        """Set current epoch for sparsity warmup scheduling."""
        self._current_epoch = epoch

    def apply_reinforcement(self):
        """Apply deferred spike strength update. Call after optimizer.step()."""
        if self._pending_mask is not None:
            self._update_spike_strength(self._pending_mask)
            self._pending_mask = None

    @torch.no_grad()
    def _update_spike_strength(self, mask):
        """Update S based on firing activity. Proportional boost."""
        fire_rate = mask.mean(dim=0)
        fired = fire_rate > 0

        new_S = torch.where(
            fired,
            self.S + self.alpha * fire_rate,
            self.S * self.beta,
        )
        self.S.copy_(new_S.clamp(self.s_min, self.s_max))

    def get_diagnostics(self):
        """Return diagnostic info from the last forward pass."""
        diag = {}
        if self._last_mask is not None:
            mask = self._last_mask
            diag["active_ratio"] = mask.mean().item()
            diag["active_per_sample"] = mask.sum(dim=1).mean().item()

            fire_freq = mask.mean(dim=0)
            diag["neuron_fire_freq_mean"] = fire_freq.mean().item()
            diag["neuron_fire_freq_std"] = fire_freq.std().item()
            diag["neurons_never_fired"] = (fire_freq == 0).sum().item()
            diag["neurons_always_fired"] = (fire_freq == 1).sum().item()

        if self._last_energy_remaining is not None:
            diag["energy_remaining"] = self._last_energy_remaining
            diag["energy_cost"] = self._last_energy_cost
            diag["neurons_fired"] = self._last_neurons_fired

        diag["S_mean"] = self.S.mean().item()
        diag["S_std"] = self.S.std().item()
        diag["S_min"] = self.S.min().item()
        diag["S_max"] = self.S.max().item()
        diag["S_at_floor"] = (self.S <= self.s_min * 1.01).sum().item()
        diag["S_at_ceiling"] = (self.S >= self.s_max * 0.99).sum().item()

        if self._last_V is not None:
            V = self._last_V
            diag["V_mean"] = V.mean().item()
            diag["V_std"] = V.std().item()
            diag["V_min"] = V.min().item()
            diag["V_max"] = V.max().item()

        return diag


def _max_neurons_for_budget(budget, decay_rate):
    """How many neurons can fire given a budget where neuron i costs 1 + i*decay_rate?

    Total cost for k neurons = k + decay_rate * k*(k-1)/2
    Solve: k + d*k*(k-1)/2 <= budget
    """
    if decay_rate <= 0:
        return max(1, int(budget))

    # Quadratic: d/2 * k^2 + (1 - d/2) * k - budget <= 0
    a = decay_rate / 2.0
    b = 1.0 - decay_rate / 2.0
    c = -budget
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return 1
    k = (-b + discriminant ** 0.5) / (2 * a)
    return max(1, int(k))
