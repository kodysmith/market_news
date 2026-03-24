import os
import time
import json
from collections import deque

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .layers import SSANLayer


class TrainingDiagnostics:
    """Monitors training health: gradients, loss stalls, neuron activity,
    spike strength dynamics, and membrane potential."""

    def __init__(self, model, config, output_dir="results/diagnostics"):
        self.model = model
        self.config = config
        self.diag_cfg = config.diagnostics
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.enabled = self.diag_cfg.enabled
        self.log_every = self.diag_cfg.log_every_n_batches
        self.stall_window = self.diag_cfg.stall_window
        self.stall_epsilon = self.diag_cfg.stall_epsilon
        self.grad_vanish_thresh = self.diag_cfg.grad_vanish_threshold
        self.grad_explode_thresh = self.diag_cfg.grad_explode_threshold
        self.dead_patience = self.diag_cfg.dead_neuron_patience

        # History buffers
        self.loss_history = deque(maxlen=self.stall_window * 2)
        self.grad_norms = {}  # layer_name -> deque of norms
        self.epoch_diagnostics = []  # per-epoch summaries
        self.batch_diagnostics = []  # sampled per-batch snapshots
        self.warnings = []

        # Neuron tracking
        self._batches_since_fired = {}  # layer_name -> tensor of batch counts
        self._step = 0
        self._epoch = 0

        # Initialize per-layer tracking
        self._ssan_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, SSANLayer):
                self._ssan_layers[name] = module
                self.grad_norms[name] = deque(maxlen=500)
                self._batches_since_fired[name] = torch.zeros(module.output_dim)

    def on_batch_end(self, loss, epoch, batch_idx):
        """Called after each training batch."""
        if not self.enabled:
            return

        self._step += 1
        self._epoch = epoch
        self.loss_history.append(loss)

        # Collect gradient norms
        self._collect_grad_norms()

        # Update neuron firing tracking
        self._update_neuron_tracking()

        # Periodic detailed logging
        if self._step % self.log_every == 0:
            snapshot = self._build_snapshot(loss, epoch, batch_idx)
            self.batch_diagnostics.append(snapshot)

            # Check for problems
            self._check_gradient_health(snapshot)
            self._check_loss_stall(snapshot)
            self._check_neuron_health(snapshot)
            self._check_spike_strength(snapshot)

        return self.get_active_warnings()

    def on_epoch_end(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Called after each epoch. Produces epoch-level summary."""
        if not self.enabled:
            return

        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "timestamp": time.time(),
            "layers": {},
        }

        for name, layer in self._ssan_layers.items():
            summary["layers"][name] = layer.get_diagnostics()

            # Add gradient norm stats
            norms = list(self.grad_norms.get(name, []))
            if norms:
                summary["layers"][name]["grad_norm_mean"] = float(np.mean(norms))
                summary["layers"][name]["grad_norm_std"] = float(np.std(norms))
                summary["layers"][name]["grad_norm_min"] = float(np.min(norms))
                summary["layers"][name]["grad_norm_max"] = float(np.max(norms))

        summary["active_warnings"] = [w for w in self.warnings if w.get("active", True)]
        self.epoch_diagnostics.append(summary)

        return summary

    def _collect_grad_norms(self):
        """Collect gradient L2 norms for each SSAN layer's weights."""
        for name, layer in self._ssan_layers.items():
            w = layer.linear.weight
            if w.grad is not None:
                norm = w.grad.data.norm(2).item()
                self.grad_norms[name].append(norm)

    def _update_neuron_tracking(self):
        """Track how long each neuron has gone without firing."""
        for name, layer in self._ssan_layers.items():
            if layer._last_mask is not None:
                # Any neuron that fired in this batch resets its counter
                fired_any = layer._last_mask.any(dim=0).cpu()
                counter = self._batches_since_fired[name]
                counter[fired_any] = 0
                counter[~fired_any] += 1
                self._batches_since_fired[name] = counter

    def _build_snapshot(self, loss, epoch, batch_idx):
        """Build a detailed diagnostic snapshot."""
        snap = {
            "step": self._step,
            "epoch": epoch,
            "batch": batch_idx,
            "loss": loss,
            "timestamp": time.time(),
            "layers": {},
        }

        for name, layer in self._ssan_layers.items():
            layer_diag = layer.get_diagnostics()

            # Add gradient info
            norms = list(self.grad_norms.get(name, []))
            if norms:
                recent = norms[-min(10, len(norms)):]
                layer_diag["grad_norm_recent_mean"] = float(np.mean(recent))

            # Add neuron liveness
            counter = self._batches_since_fired.get(name)
            if counter is not None:
                layer_diag["dead_neurons"] = int((counter >= self.dead_patience).sum().item())
                layer_diag["max_batches_without_fire"] = int(counter.max().item())

            snap["layers"][name] = layer_diag

        return snap

    def _check_gradient_health(self, snapshot):
        """Detect vanishing or exploding gradients."""
        for name in self._ssan_layers:
            norms = list(self.grad_norms.get(name, []))
            if len(norms) < 5:
                continue

            recent = norms[-5:]
            mean_norm = np.mean(recent)

            if mean_norm < self.grad_vanish_thresh:
                self._add_warning(
                    "vanishing_gradient",
                    f"Layer {name}: grad norm {mean_norm:.2e} < {self.grad_vanish_thresh:.2e} "
                    f"for 5 consecutive batches. Sigmoid may be saturating — "
                    f"try reducing sharpness or checking V magnitude.",
                    snapshot,
                )
            elif mean_norm > self.grad_explode_thresh:
                self._add_warning(
                    "exploding_gradient",
                    f"Layer {name}: grad norm {mean_norm:.2e} > {self.grad_explode_thresh}. "
                    f"Consider gradient clipping or reducing learning rate.",
                    snapshot,
                )

        # Check grad flow ratio between first and last SSAN layer
        layer_names = list(self._ssan_layers.keys())
        if len(layer_names) >= 2:
            first_norms = list(self.grad_norms.get(layer_names[0], []))
            last_norms = list(self.grad_norms.get(layer_names[-1], []))
            if first_norms and last_norms:
                ratio = np.mean(last_norms[-5:]) / max(np.mean(first_norms[-5:]), 1e-12)
                if ratio > 100:
                    self._add_warning(
                        "gradient_bottleneck",
                        f"Grad norm ratio (last/first layer) = {ratio:.1f}x. "
                        f"Early layers are getting much weaker gradients.",
                        snapshot,
                    )

    def _check_loss_stall(self, snapshot):
        """Detect loss plateaus, oscillation, or divergence."""
        if len(self.loss_history) < self.stall_window:
            return

        recent = list(self.loss_history)[-self.stall_window:]
        arr = np.array(recent)

        # Divergence: NaN or Inf
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            self._add_warning(
                "divergence",
                "Loss is NaN or Inf! Training has diverged.",
                snapshot,
                severity="critical",
            )
            return

        # Divergence: monotonic increase
        if len(arr) >= 10:
            tail = arr[-10:]
            if all(tail[i] <= tail[i + 1] for i in range(len(tail) - 1)):
                self._add_warning(
                    "loss_increasing",
                    f"Loss has increased for 10 consecutive batches "
                    f"({tail[0]:.4f} -> {tail[-1]:.4f}). "
                    f"Reduce learning rate or check for instability.",
                    snapshot,
                )

        # Plateau
        variance = np.var(arr)
        improvement = arr[0] - arr[-1]
        if variance < self.stall_epsilon and abs(improvement) < self.stall_epsilon:
            self._add_warning(
                "loss_plateau",
                f"Loss plateaued for {self.stall_window} batches "
                f"(var={variance:.6f}, improvement={improvement:.6f}). "
                f"Try increasing learning rate or adjusting sparsity (top_k).",
                snapshot,
            )

        # Oscillation: high variance but no trend
        if variance > 0.1 and abs(improvement) < self.stall_epsilon * 10:
            self._add_warning(
                "loss_oscillation",
                f"Loss oscillating (var={variance:.4f}) without convergence. "
                f"Learning rate may be too high, or sparsity mask is unstable.",
                snapshot,
            )

    def _check_neuron_health(self, snapshot):
        """Detect dead or dominant neurons."""
        for name, layer in self._ssan_layers.items():
            layer_snap = snapshot["layers"].get(name, {})

            dead = layer_snap.get("dead_neurons", 0)
            if dead > layer.output_dim * 0.25:
                self._add_warning(
                    "dead_neurons",
                    f"Layer {name}: {dead}/{layer.output_dim} neurons dead "
                    f"(haven't fired in {self.dead_patience} batches). "
                    f"Increase top_k, reduce beta, or lower sharpness.",
                    snapshot,
                )

            always = layer_snap.get("neurons_always_fired", 0)
            if always > layer.output_dim * 0.1:
                self._add_warning(
                    "dominant_neurons",
                    f"Layer {name}: {always}/{layer.output_dim} neurons fire in every sample. "
                    f"Network may be over-relying on few neurons. "
                    f"Reduce alpha or increase top_k.",
                    snapshot,
                )

    def _check_spike_strength(self, snapshot):
        """Detect spike strength saturation."""
        for name, layer in self._ssan_layers.items():
            layer_snap = snapshot["layers"].get(name, {})

            at_floor = layer_snap.get("S_at_floor", 0)
            at_ceil = layer_snap.get("S_at_ceiling", 0)
            total = layer.output_dim

            if at_floor > total * 0.5:
                self._add_warning(
                    "S_collapsed",
                    f"Layer {name}: {at_floor}/{total} neurons at S_min. "
                    f"Spike strength is collapsing — increase beta or reduce alpha.",
                    snapshot,
                )

            if at_ceil > total * 0.5:
                self._add_warning(
                    "S_saturated",
                    f"Layer {name}: {at_ceil}/{total} neurons at S_max. "
                    f"Spike strength saturating — reduce alpha or increase S_max.",
                    snapshot,
                )

    def _add_warning(self, warning_type, message, snapshot, severity="warning"):
        """Record a warning, avoiding duplicates within a short window."""
        # Deduplicate: don't repeat the same warning type within 100 steps
        for w in self.warnings:
            if w["type"] == warning_type and w.get("active") and \
               self._step - w["step"] < 100:
                return

        warning = {
            "type": warning_type,
            "message": message,
            "severity": severity,
            "step": self._step,
            "epoch": self._epoch,
            "active": True,
        }
        self.warnings.append(warning)
        print(f"  [{severity.upper()}] {message}")

        # Save snapshot on critical warnings
        if severity == "critical" or self.diag_cfg.snapshot_on_stall:
            self._save_snapshot(snapshot, warning_type)

    def get_active_warnings(self):
        """Return list of currently active warnings."""
        return [w for w in self.warnings if w.get("active", True)]

    def _save_snapshot(self, snapshot, label):
        """Save diagnostic snapshot to disk."""
        path = os.path.join(self.output_dir, f"snapshot_{label}_step{self._step}.json")
        # Convert to serializable format
        serializable = json.loads(json.dumps(snapshot, default=str))
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)

    def save_history(self, path=None):
        """Save full diagnostic history."""
        if path is None:
            path = os.path.join(self.output_dir, "diagnostics_history.json")
        data = {
            "epoch_diagnostics": self.epoch_diagnostics,
            "warnings": self.warnings,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def plot_summary(self, save_path=None):
        """Generate multi-panel diagnostic summary plot."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "diagnostics_summary.png")

        if not self.epoch_diagnostics:
            return

        epochs = [d["epoch"] for d in self.epoch_diagnostics]
        train_losses = [d["train_loss"] for d in self.epoch_diagnostics]
        val_losses = [d["val_loss"] for d in self.epoch_diagnostics]
        train_accs = [d["train_acc"] for d in self.epoch_diagnostics]
        val_accs = [d["val_acc"] for d in self.epoch_diagnostics]

        layer_names = list(self._ssan_layers.keys())
        n_layers = len(layer_names)

        fig, axes = plt.subplots(2 + n_layers, 2, figsize=(14, 4 * (2 + n_layers)))
        if axes.ndim == 1:
            axes = axes.reshape(-1, 2)

        # Row 0: Loss and Accuracy
        axes[0, 0].plot(epochs, train_losses, "b-", label="Train")
        axes[0, 0].plot(epochs, val_losses, "r-", label="Val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, train_accs, "b-", label="Train")
        axes[0, 1].plot(epochs, val_accs, "r-", label="Val")
        axes[0, 1].set_title("Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Row 1: Gradient norms over training (from batch diagnostics)
        for name in layer_names:
            norms = list(self.grad_norms.get(name, []))
            if norms:
                axes[1, 0].plot(norms, label=name, alpha=0.7)
        axes[1, 0].set_title("Gradient Norms (per batch)")
        axes[1, 0].set_xlabel("Batch")
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend(fontsize=7)
        axes[1, 0].grid(True, alpha=0.3)

        # Warnings timeline
        if self.warnings:
            w_steps = [w["step"] for w in self.warnings]
            w_types = [w["type"] for w in self.warnings]
            colors = {"vanishing_gradient": "blue", "exploding_gradient": "red",
                      "loss_plateau": "orange", "loss_oscillation": "purple",
                      "dead_neurons": "gray", "dominant_neurons": "green",
                      "S_collapsed": "cyan", "S_saturated": "magenta",
                      "divergence": "black", "loss_increasing": "brown",
                      "gradient_bottleneck": "pink"}
            for step, wtype in zip(w_steps, w_types):
                axes[1, 1].axvline(step, color=colors.get(wtype, "gray"),
                                   alpha=0.7, linewidth=2)
            # Legend
            seen = set()
            for wtype in w_types:
                if wtype not in seen:
                    axes[1, 1].plot([], [], color=colors.get(wtype, "gray"),
                                   label=wtype, linewidth=2)
                    seen.add(wtype)
            axes[1, 1].legend(fontsize=6)
        axes[1, 1].set_title("Warnings Timeline")
        axes[1, 1].set_xlabel("Step")

        # Per-layer rows: S distribution + active ratio
        for li, name in enumerate(layer_names):
            row = 2 + li
            layer_epochs = []
            s_means = []
            s_stds = []
            active_ratios = []

            for d in self.epoch_diagnostics:
                if name in d["layers"]:
                    layer_epochs.append(d["epoch"])
                    ld = d["layers"][name]
                    s_means.append(ld.get("S_mean", 0))
                    s_stds.append(ld.get("S_std", 0))
                    active_ratios.append(ld.get("active_ratio", 0))

            if layer_epochs:
                s_means = np.array(s_means)
                s_stds = np.array(s_stds)
                axes[row, 0].plot(layer_epochs, s_means, "b-")
                axes[row, 0].fill_between(layer_epochs, s_means - s_stds,
                                          s_means + s_stds, alpha=0.2)
                axes[row, 0].set_title(f"{name} — Spike Strength (S)")
                axes[row, 0].set_xlabel("Epoch")
                axes[row, 0].grid(True, alpha=0.3)

                axes[row, 1].plot(layer_epochs, active_ratios, "g-")
                axes[row, 1].set_title(f"{name} — Active Neuron Ratio")
                axes[row, 1].set_xlabel("Epoch")
                axes[row, 1].set_ylim(0, 1)
                axes[row, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Diagnostic summary saved to {save_path}")

    def format_batch_status(self, loss, acc, epoch, batch_idx):
        """One-line console summary for periodic logging."""
        parts = [f"E{epoch} B{batch_idx} | loss={loss:.4f} acc={acc:.4f}"]

        for name, layer in self._ssan_layers.items():
            short_name = name.split(".")[-1]
            diag = layer.get_diagnostics()
            parts.append(
                f"{short_name}[act={diag.get('active_ratio', 0):.2f} "
                f"S={diag.get('S_mean', 0):.2f}±{diag.get('S_std', 0):.2f}]"
            )

        # Gradient norms
        grad_parts = []
        for name in self._ssan_layers:
            norms = list(self.grad_norms.get(name, []))
            if norms:
                grad_parts.append(f"{norms[-1]:.2e}")
        if grad_parts:
            parts.append(f"grads=[{', '.join(grad_parts)}]")

        active_warns = [w["type"] for w in self.warnings
                        if w.get("active") and self._step - w["step"] < 100]
        if active_warns:
            parts.append(f"WARNS: {', '.join(set(active_warns))}")

        return " | ".join(parts)
