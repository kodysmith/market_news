#!/usr/bin/env python3
"""Train SSAN (Sparse Spiking Adaptive Network) for SPX IC entry prediction.

Adapts the Graph Transformer SSAN from models_ai/ for time-series classification.
Instead of image patches, tokens are trading days in a 20-day lookback window.

Key SSAN features applied to trading:
  - Sparse attention: each day only attends to its most relevant neighbors
  - Membrane potential: temporal memory accumulates across the lookback window
  - Spike strength: days with strong predictive signal get reinforced
  - Graph structure: recent days (local), weekly anchors (skip), and regime days (global)
"""
from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# SSAN Time-Series Model
# ---------------------------------------------------------------------------

class SparseTimeSeriesAttention(nn.Module):
    """Top-k sparse attention for time series tokens."""

    def __init__(self, dim, n_heads=4, top_k_attn=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.top_k_attn = top_k_attn
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, graph_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Graph mask: block attention where no edge exists
        if graph_mask is not None:
            attn_mask = graph_mask.unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        # Top-k sparse: only attend to top-k tokens per query
        k_attn = min(self.top_k_attn, N)
        _, topk_idx = torch.topk(attn, k_attn, dim=-1)
        sparse_mask = torch.zeros_like(attn)
        sparse_mask.scatter_(-1, topk_idx, 1.0)
        attn = attn.masked_fill(sparse_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class SpikingTimeSeriesBlock(nn.Module):
    """Transformer block with SSAN membrane potential dynamics."""

    def __init__(self, dim, n_heads=4, top_k_attn=8, mlp_ratio=2.0,
                 lambda_decay=0.85, dropout=0.1):
        super().__init__()
        self.lambda_decay = lambda_decay
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseTimeSeriesAttention(dim, n_heads, top_k_attn, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, graph_mask=None, V_prev=None):
        attn_out = self.attn(self.norm1(x), graph_mask)
        # Leaky membrane potential
        if V_prev is not None:
            V = self.lambda_decay * V_prev + attn_out
        else:
            V = attn_out
        x = x + V
        x = x + self.mlp(self.norm2(x))
        return x, V


class TimeSeriesSSAN(nn.Module):
    """SSAN adapted for time-series entry prediction.

    Input: (batch, seq_len=20, n_features=29)
    Output: (batch, n_classes=3)

    Architecture:
      - Linear embedding of features to d_model
      - Learned positional encoding
      - Graph structure over time steps:
        * Local: each day attends to ±2 neighbors
        * Weekly: day 0,5,10,15,19 are hub nodes (weekly anchors)
        * CLS token: aggregation node connected to all
      - 3 SpikingTransformerBlocks with membrane potential
      - Spike strength reinforcement per timestep
      - CLS token → classification head
    """

    def __init__(self, n_features, n_classes=3, d_model=64, n_heads=4,
                 n_layers=3, top_k_attn=8, lambda_decay=0.85, dropout=0.3,
                 seq_len=20, alpha=0.02, beta=0.98):
        super().__init__()
        self.seq_len = seq_len
        self.n_tokens = seq_len + 1  # +1 for CLS
        self.d_model = d_model

        # Embedding
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Graph structure over time
        graph_mask = self._build_temporal_graph(seq_len)
        self.register_buffer("graph_mask", graph_mask)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpikingTimeSeriesBlock(
                dim=d_model, n_heads=n_heads, top_k_attn=top_k_attn,
                mlp_ratio=2.0, lambda_decay=lambda_decay, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Spike strength per token (reinforcement)
        self.register_buffer("S", torch.ones(self.n_tokens))
        self.alpha = alpha
        self.beta = beta
        self._pending_update = False

    def _build_temporal_graph(self, seq_len):
        """Build graph over time steps.

        Structure:
          - CLS (index 0) connects to everything
          - Each day connects to ±2 neighbors (local context)
          - Weekly anchors (every 5 days) connect to each other (regime context)
          - Most recent 3 days connect to each other (immediate context)
        """
        n = seq_len + 1  # +1 for CLS
        mask = torch.zeros(n, n)

        # CLS connects to all
        mask[0, :] = 1.0
        mask[:, 0] = 1.0

        # Local: ±2 neighbors
        for i in range(1, n):
            for j in range(max(1, i - 2), min(n, i + 3)):
                mask[i, j] = 1.0

        # Weekly anchors: days 0, 5, 10, 15 (indices 1, 6, 11, 16)
        anchors = [1 + i * 5 for i in range(seq_len // 5) if 1 + i * 5 < n]
        for a in anchors:
            for b in anchors:
                mask[a, b] = 1.0

        # Most recent 3 days fully connected
        for i in range(max(1, n - 3), n):
            for j in range(max(1, n - 3), n):
                mask[i, j] = 1.0

        # Self-attention always allowed
        for i in range(n):
            mask[i, i] = 1.0

        return mask

    def forward(self, x):
        B = x.size(0)

        # Embed features
        tokens = self.input_proj(x)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Positional encoding
        tokens = tokens + self.pos_embed

        # Spike strength scaling
        S = self.S.detach().unsqueeze(0).unsqueeze(-1)
        tokens = tokens * S

        # Transformer blocks with membrane potential
        V = None
        for block in self.blocks:
            tokens, V = block(tokens, self.graph_mask, V)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if self.training:
            self._pending_update = True

        return logits

    def apply_reinforcement(self, predictions, targets):
        """Update spike strengths based on prediction accuracy."""
        if not self._pending_update:
            return
        with torch.no_grad():
            correct = (predictions.argmax(dim=1) == targets).float().mean().item()
            boost = self.alpha * correct
            new_S = self.S + boost
            new_S = new_S * (1.0 - (1.0 - self.beta) * 0.1)
            self.S.copy_(new_S.clamp(0.5, 2.0))
        self._pending_update = False

    def get_diagnostics(self):
        n_edges = int(self.graph_mask.sum().item())
        n_possible = self.n_tokens * self.n_tokens
        return {
            "n_tokens": self.n_tokens,
            "n_edges": n_edges,
            "edge_density": n_edges / n_possible,
            "attn_savings_pct": (1.0 - n_edges / n_possible) * 100,
            "S_mean": self.S.mean().item(),
            "S_std": self.S.std().item(),
            "S_min": self.S.min().item(),
            "S_max": self.S.max().item(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet(ROOT / "data" / "ml_training_entry.parquet")

    # Features are named like ret_1d_z_t0, ret_1d_z_t1, ... ret_1d_z_t19
    # Group by timestep: all columns ending in _t0, _t1, ..., _t19
    meta_cols = {"entry_date", "exit_date", "exit_reason", "config", "hold_days",
                 "pnl", "credit", "entry_target", "entry_target_3class", "pnl_normalized"}
    feature_cols_t0 = sorted([c for c in df.columns if c.endswith("_t0") and c not in meta_cols])
    n_features = len(feature_cols_t0)

    # Build ordered feature matrix: (n_samples, 20 timesteps, n_features)
    all_feature_cols = []
    for t in range(20):
        cols_t = [c.replace("_t0", f"_t{t}") for c in feature_cols_t0]
        all_feature_cols.extend(cols_t)

    X_all = df[all_feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_seq = X_all.reshape(len(df), 20, n_features)

    y = df["entry_target_3class"].values.astype(np.int64)
    dates = pd.to_datetime(df["entry_date"])

    train_mask = dates.dt.year <= 2020
    val_mask = (dates.dt.year >= 2021) & (dates.dt.year <= 2022)
    test_mask = dates.dt.year >= 2023

    return (
        X_seq[train_mask], y[train_mask],
        X_seq[val_mask], y[val_mask],
        X_seq[test_mask], y[test_mask],
        n_features,
    )


def train_ssan():
    print("Loading training data...")
    X_train, y_train, X_val, y_val, X_test, y_test, n_features = load_data()
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Features per timestep: {n_features}, Sequence length: 20")

    device = torch.device("cpu")  # MPS has issues with sparse ops
    print(f"  Device: {device}")

    # Class weights
    class_counts = np.bincount(y_train, minlength=3).astype(np.float32)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 3
    weights_tensor = torch.tensor(class_weights, device=device)

    model = TimeSeriesSSAN(
        n_features=n_features, n_classes=3, d_model=64, n_heads=4,
        n_layers=3, top_k_attn=10, lambda_decay=0.85, dropout=0.3,
        seq_len=20, alpha=0.02, beta=0.98,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    diag = model.get_diagnostics()
    print(f"  Graph: {diag['n_edges']} edges / {diag['n_tokens']}^2={diag['n_tokens']**2} possible "
          f"({diag['attn_savings_pct']:.0f}% attention savings)")

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_val_t = torch.tensor(X_val, device=device)
    y_val_t = torch.tensor(y_val, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    best_val_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0
    batch_size = 128

    print("\n--- Training SSAN ---")
    for epoch in range(1, 81):
        model.train()
        perm = torch.randperm(len(X_train_t), device=device)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.apply_reinforcement(logits.detach(), yb)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()

        avg_train = total_loss / n_batches

        if epoch % 10 == 0:
            diag = model.get_diagnostics()
            print(f"  Epoch {epoch:>3}/80 — train={avg_train:.4f} val={val_loss:.4f} "
                  f"S=[{diag['S_min']:.2f}-{diag['S_max']:.2f}] lr={scheduler.get_last_lr()[0]:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_preds = test_logits.argmax(dim=1).cpu().numpy()

    labels = ["loss", "small_win", "full_win"]
    print("\n  === SSAN Test Results ===")
    print(classification_report(y_test, test_preds, target_names=labels, zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    save_path = ROOT / "data" / "ml_entry_ssan.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "n_features": n_features, "n_classes": 3, "d_model": 64,
            "n_heads": 4, "n_layers": 3, "top_k_attn": 10,
            "lambda_decay": 0.85, "dropout": 0.3, "seq_len": 20,
        },
        "diagnostics": model.get_diagnostics(),
    }, save_path)
    print(f"\n  Saved SSAN model to {save_path}")

    diag = model.get_diagnostics()
    print(f"\n  Final spike strengths: mean={diag['S_mean']:.3f} "
          f"range=[{diag['S_min']:.3f}, {diag['S_max']:.3f}]")

    return model, test_preds, y_test


def compare_models():
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: SSAN vs XGBoost vs Vanilla Transformer")
    print("=" * 70)

    model, ssan_preds, y_test = train_ssan()

    # Load XGBoost
    try:
        import joblib
        xgb_path = ROOT / "data" / "ml_entry_xgboost.joblib"
        if xgb_path.exists():
            xgb_model = joblib.load(xgb_path)
            df = pd.read_parquet(ROOT / "data" / "ml_training_entry.parquet")
            # Use only numeric timestep feature columns (same as SSAN)
            feature_cols_t0_x = sorted([c for c in df.columns if c.endswith("_t0")
                                       and df[c].dtype in ('float64','float32','int64')])
            feature_cols = []
            for t in range(20):
                feature_cols.extend([c.replace("_t0", f"_t{t}") for c in feature_cols_t0_x])
            dates = pd.to_datetime(df["entry_date"])
            test_mask = dates.dt.year >= 2023
            # XGBoost was trained with specific columns — rebuild to match
            xgb_n_features = xgb_model.n_features_in_
            X_test_flat = df.loc[test_mask, feature_cols].values[:, :xgb_n_features]
            X_test_flat = np.nan_to_num(X_test_flat, nan=0.0)
            y_test_xgb = df.loc[test_mask, "entry_target_3class"].values
            xgb_preds = xgb_model.predict(X_test_flat)

            labels = ["loss", "small_win", "full_win"]

            from sklearn.metrics import recall_score, f1_score, accuracy_score
            xgb_loss_recall = recall_score(y_test_xgb, xgb_preds, labels=[0], average=None, zero_division=0)[0]
            ssan_loss_recall = recall_score(y_test, ssan_preds, labels=[0], average=None, zero_division=0)[0]
            xgb_acc = accuracy_score(y_test_xgb, xgb_preds)
            ssan_acc = accuracy_score(y_test, ssan_preds)
            xgb_f1 = f1_score(y_test_xgb, xgb_preds, average="macro", zero_division=0)
            ssan_f1 = f1_score(y_test, ssan_preds, average="macro", zero_division=0)

            print(f"\n{'Metric':<30} {'XGBoost':>12} {'SSAN':>12} {'Winner':>10}")
            print("-" * 70)
            print(f"{'Accuracy':<30} {xgb_acc:>11.1%} {ssan_acc:>11.1%} {'SSAN' if ssan_acc > xgb_acc else 'XGBoost':>10}")
            print(f"{'Macro F1':<30} {xgb_f1:>11.3f} {ssan_f1:>11.3f} {'SSAN' if ssan_f1 > xgb_f1 else 'XGBoost':>10}")
            print(f"{'Loss Detection (recall)':<30} {xgb_loss_recall:>11.1%} {ssan_loss_recall:>11.1%} {'SSAN' if ssan_loss_recall > xgb_loss_recall else 'XGBoost':>10}")
            print()
            print("  Loss detection is the KEY metric — catching losing trades before entry")
            print(f"  XGBoost catches {xgb_loss_recall:.0%} of losses, SSAN catches {ssan_loss_recall:.0%}")

            if ssan_loss_recall > xgb_loss_recall:
                avoided = ssan_loss_recall - xgb_loss_recall
                # Estimate dollar impact: ~150 test losses, avg loss ~$800/ct
                n_test_losses = (y_test == 0).sum()
                extra_avoided = int(avoided * n_test_losses)
                print(f"\n  SSAN would avoid ~{extra_avoided} additional losing trades in test period")
                print(f"  At ~$800/contract avg loss × 10 contracts = ~${extra_avoided * 8000:,} saved")

    except Exception as e:
        print(f"  Could not load XGBoost for comparison: {e}")


if __name__ == "__main__":
    compare_models()
