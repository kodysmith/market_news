import time
import json
import os

import torch
import numpy as np

from .layers import SSANLayer


class MetricsTracker:
    """Collects per-batch and per-epoch training metrics."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.epoch_losses = []
        self.epoch_correct = 0
        self.epoch_total = 0
        self.epoch_start_time = None
        self.history = []

        # Find SSAN layers for activity tracking
        self._ssan_layers = {}
        for name, module in model.named_modules():
            if isinstance(module, SSANLayer):
                self._ssan_layers[name] = module

    def start_epoch(self):
        self.epoch_losses = []
        self.epoch_correct = 0
        self.epoch_total = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()

    def update(self, loss, output, target):
        """Called after each batch."""
        self.epoch_losses.append(loss.item())
        pred = output.argmax(dim=1)
        self.epoch_correct += (pred == target).sum().item()
        self.epoch_total += target.size(0)

    def end_epoch(self, epoch, phase="train"):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.epoch_start_time

        avg_loss = float(np.mean(self.epoch_losses)) if self.epoch_losses else 0
        accuracy = self.epoch_correct / max(self.epoch_total, 1)

        # Collect SSAN-specific metrics
        layer_metrics = {}
        for name, layer in self._ssan_layers.items():
            layer_metrics[name] = layer.get_diagnostics()

        record = {
            "epoch": epoch,
            "phase": phase,
            "loss": avg_loss,
            "accuracy": accuracy,
            "wall_time_s": elapsed,
            "samples": self.epoch_total,
            "layer_metrics": layer_metrics,
        }
        self.history.append(record)
        return record

    def compute_flops(self, input_shape):
        """Estimate FLOPs for one forward pass.

        Returns:
            dict with 'dense_flops' (what GPU actually computes) and
            'effective_flops' (what a sparse runtime would compute).
        """
        dense_flops = 0
        effective_flops = 0

        # Build ordered list of (Linear, parent_ssan_layer_or_None)
        ssan_layer_linears = set()
        for layer in self._ssan_layers.values():
            ssan_layer_linears.add(id(layer.linear))

        # Map each SSAN layer to its top_k for computing effective input to next layer
        prev_effective_width = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                in_f = module.in_features
                out_f = module.out_features
                dense = 2 * in_f * out_f
                dense_flops += dense

                # Effective: use reduced input width if previous layer was sparse
                eff_in = prev_effective_width if prev_effective_width else in_f
                effective_flops += 2 * eff_in * out_f

                # If this linear belongs to an SSAN layer, output is sparse
                if id(module) in ssan_layer_linears:
                    for layer in self._ssan_layers.values():
                        if id(layer.linear) == id(module) and layer.enable_sparsity:
                            prev_effective_width = layer.top_k
                            break
                    else:
                        prev_effective_width = None
                else:
                    prev_effective_width = None

        return {
            "dense_flops": dense_flops,
            "effective_flops": effective_flops,
            "efficiency_ratio": effective_flops / max(dense_flops, 1),
        }

    def save(self, path):
        """Save metrics history to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
