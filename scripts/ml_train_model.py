#!/usr/bin/env python3
"""Train transformer + XGBoost models for SPX IC entry prediction."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOOKBACK = 20
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
DIM_FF = 128
DROPOUT = 0.3
NUM_CLASSES = 3
BATCH_SIZE = 128
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 0.01


class ICDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, seq_len * n_features) -> reshape to (N, seq_len, n_features)
        self.n_features = X.shape[1] // LOOKBACK
        self.X = torch.tensor(
            X.reshape(-1, LOOKBACK, self.n_features), dtype=torch.float32
        )
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2 + d_model % 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ICTransformer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.embed = nn.Linear(n_features, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL, max_len=LOOKBACK + 10)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=DROPOUT, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        # Pool: use last timestep
        x = x[:, -1, :]
        return self.head(x)


def prepare_data(df: pd.DataFrame):
    """Extract feature matrix X and target y from training dataframe."""
    meta_cols = [
        "strategy", "entry_date", "exit_date", "exit_reason",
        "pnl", "hold_days", "entry_target", "entry_target_3class", "pnl_normalized",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df["entry_target_3class"].values.astype(np.int64)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, feature_cols


def walk_forward_split(df: pd.DataFrame):
    """Split by entry_date: train 2015-2020, val 2021-2022, test 2023-2025."""
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    train = df[df["entry_date"] < "2021-01-01"]
    val = df[(df["entry_date"] >= "2021-01-01") & (df["entry_date"] < "2023-01-01")]
    test = df[df["entry_date"] >= "2023-01-01"]
    return train, val, test


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse frequency class weights."""
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_transformer(train_df, val_df, test_df):
    """Train transformer model with walk-forward validation."""
    X_train, y_train, feat_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    n_features = X_train.shape[1] // LOOKBACK
    print(f"  Features per timestep: {n_features}, Sequence length: {LOOKBACK}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = ICDataset(X_train, y_train)
    val_ds = ICDataset(X_val, y_val)
    test_ds = ICDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ICTransformer(n_features).to(device)
    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        scheduler.step()
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                loss = criterion(model(xb), yb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} — train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    print("\n  === Transformer Test Results ===")
    print(classification_report(all_true, all_preds, target_names=["loss", "small_win", "full_win"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(all_true, all_preds))

    # Save model
    model_path = ROOT / "data" / "ml_entry_model.pt"
    torch.save({
        "model_state": best_state,
        "n_features": n_features,
        "feature_cols": feat_cols,
    }, model_path)
    print(f"\n  Saved transformer model to {model_path}")

    return model, n_features


def train_xgboost_baseline(train_df, val_df, test_df):
    """Train XGBoost baseline for comparison."""
    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not installed, skipping baseline.")
        return

    X_train, y_train, feat_cols = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)
    X_test, y_test, _ = prepare_data(test_df)

    # Compute sample weights for class imbalance
    counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(float)
    counts = np.maximum(counts, 1.0)
    weight_map = {i: len(y_train) / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)}
    sample_weights = np.array([weight_map[yi] for yi in y_train])

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=NUM_CLASSES,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    preds = model.predict(X_test)
    print("\n  === XGBoost Baseline Test Results ===")
    print(classification_report(y_test, preds, target_names=["loss", "small_win", "full_win"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Feature importance (top 20 by gain)
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[-20:][::-1]
    print("\n  Top 20 features (XGBoost):")
    for i in top_idx:
        # Map flat index to (feature_name, timestep)
        feat_idx = i % (len(feat_cols) // LOOKBACK) if len(feat_cols) > LOOKBACK else i
        print(f"    {feat_cols[i] if i < len(feat_cols) else f'feat_{i}'}: {imp[i]:.4f}")

    import joblib
    xgb_path = ROOT / "data" / "ml_entry_xgboost.joblib"
    joblib.dump(model, xgb_path)
    print(f"\n  Saved XGBoost model to {xgb_path}")


def main():
    entry_path = ROOT / "data" / "ml_training_entry.parquet"
    if not entry_path.exists():
        print("Training data not found. Run ml_generate_training_data.py first.")
        return

    print("Loading training data...")
    df = pd.read_parquet(entry_path)
    print(f"  {len(df)} trades loaded")

    train_df, val_df, test_df = walk_forward_split(df)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("\n--- Training Transformer ---")
    train_transformer(train_df, val_df, test_df)

    print("\n--- Training XGBoost Baseline ---")
    train_xgboost_baseline(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
