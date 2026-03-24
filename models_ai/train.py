import os
import time

import torch
import torch.nn as nn

from .models import build_model
from .layers import SSANLayer
from .data import get_dataloaders
from .metrics import MetricsTracker
from .diagnostics import TrainingDiagnostics
from .utils import set_seed, get_device, ensure_dirs


def _apply_reinforcement(model):
    """Apply deferred spike strength reinforcement for all SSAN layers and graph/transformer models."""
    from .graph import GraphSSAN
    from .transformer import TransformerSSAN
    from .graph_transformer import GraphTransformerSSAN
    from .multipass_attention import MultiPassTransformerSSAN
    if isinstance(model, (GraphSSAN, TransformerSSAN, GraphTransformerSSAN,
                          MultiPassTransformerSSAN)):
        model.apply_reinforcement()
    else:
        for module in model.modules():
            if isinstance(module, SSANLayer):
                module.apply_reinforcement()


def train_one_epoch(model, loader, optimizer, criterion, device,
                    metrics, diagnostics, epoch, config):
    model.train()
    metrics.start_epoch()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Apply deferred reinforcement after gradient update
        _apply_reinforcement(model)

        metrics.update(loss, output, target)

        # Diagnostics
        if diagnostics is not None:
            acc = (output.argmax(1) == target).float().mean().item()
            warns = diagnostics.on_batch_end(loss.item(), epoch, batch_idx)

            # Periodic status line
            if (batch_idx + 1) % config.training.log_interval == 0:
                status = diagnostics.format_batch_status(
                    loss.item(), acc, epoch, batch_idx
                )
                print(f"  {status}")

            # Halt on critical divergence
            if warns:
                for w in warns:
                    if w.get("severity") == "critical":
                        print(f"CRITICAL: {w['message']} — halting training.")
                        return metrics.end_epoch(epoch, "train"), True

    return metrics.end_epoch(epoch, "train"), False


def evaluate(model, loader, criterion, device, metrics, epoch, phase="val"):
    model.eval()
    metrics.start_epoch()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            metrics.update(loss, output, target)

    return metrics.end_epoch(epoch, phase)


def run_training(config, verbose=True):
    """Full training run. Returns metrics history."""
    set_seed(config.seed)
    ensure_dirs()
    device = get_device(config.device)

    if verbose:
        print(f"Device: {device}")
        print(f"Model: {config.model.type}")
        if config.model.type == "ssan":
            print(f"Temporal mode: {config.data.temporal_mode}")
            print(f"Sparsity: {config.ssan.enable_sparsity}, "
                  f"Memory: {config.ssan.enable_memory}, "
                  f"Reinforcement: {config.ssan.enable_reinforcement}")

    # Data
    train_loader, val_loader, test_loader, input_dim, output_dim = get_dataloaders(config)

    # Model
    model = build_model(config, input_dim, output_dim).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"Parameters: {param_count:,}")

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Scheduler
    scheduler = None
    if config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )

    # Metrics and diagnostics
    train_metrics = MetricsTracker(model, device)
    val_metrics = MetricsTracker(model, device)

    diagnostics = None
    if config.diagnostics.enabled:
        exp_name = getattr(config.experiment, "name", "default")
        diag_dir = os.path.join("results", "diagnostics", exp_name)
        diagnostics = TrainingDiagnostics(model, config, output_dir=diag_dir)

    # FLOP estimate
    flops = train_metrics.compute_flops((1, 1, 28, 28))
    if verbose:
        print(f"FLOPs (dense): {flops['dense_flops']:,}")
        print(f"FLOPs (effective): {flops['effective_flops']:,}")
        print(f"Efficiency ratio: {flops['efficiency_ratio']:.3f}")
        print("-" * 60)

    best_val_acc = 0
    halted = False

    for epoch in range(config.training.epochs):
        # Update epoch for sparsity warmup scheduling
        for module in model.modules():
            if isinstance(module, SSANLayer):
                module.set_epoch(epoch)

        if verbose:
            print(f"\nEpoch {epoch + 1}/{config.training.epochs}")

        # Train
        train_record, halted = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            train_metrics, diagnostics, epoch, config,
        )
        if halted:
            break

        # Validate
        val_record = evaluate(
            model, val_loader, criterion, device, val_metrics, epoch, "val"
        )

        if verbose:
            print(f"  Train — loss: {train_record['loss']:.4f}, "
                  f"acc: {train_record['accuracy']:.4f}, "
                  f"time: {train_record['wall_time_s']:.1f}s")
            print(f"  Val   — loss: {val_record['loss']:.4f}, "
                  f"acc: {val_record['accuracy']:.4f}")

        # Diagnostics epoch summary
        if diagnostics is not None:
            diagnostics.on_epoch_end(
                epoch, train_record["loss"], val_record["loss"],
                train_record["accuracy"], val_record["accuracy"],
            )

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Checkpoint best model
        if val_record["accuracy"] > best_val_acc:
            best_val_acc = val_record["accuracy"]
            if config.experiment.save_checkpoints:
                exp_name = getattr(config.experiment, "name", "default")
                ckpt_path = os.path.join(
                    "results", "checkpoints", f"{exp_name}_best.pt"
                )
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)

    # Test evaluation
    test_metrics = MetricsTracker(model, device)
    test_record = evaluate(
        model, test_loader, criterion, device, test_metrics, 0, "test"
    )
    if verbose:
        print(f"\nTest — loss: {test_record['loss']:.4f}, "
              f"acc: {test_record['accuracy']:.4f}")
        print(f"Best val acc: {best_val_acc:.4f}")

    # Save results
    if config.experiment.save_logs:
        exp_name = getattr(config.experiment, "name", "default")
        log_path = os.path.join("results", "logs", f"{exp_name}_metrics.json")
        train_metrics.save(log_path)

    if diagnostics is not None:
        diagnostics.save_history()
        diagnostics.plot_summary()

    return {
        "train_history": train_metrics.history,
        "val_history": val_metrics.history,
        "test": test_record,
        "best_val_acc": best_val_acc,
        "flops": flops,
        "param_count": param_count,
        "halted": halted,
    }
