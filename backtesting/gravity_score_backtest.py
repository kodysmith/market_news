"""
Gravity Score Backtest - Pin vs Volatile Day Prediction

Tests whether a single "gravity score" (derived from put wall, call wall, max pain, spot)
predicts end-of-day behavior: high score → pinning (small range), low score → volatile (large range).

Design doc: docs/TRADING_DASHBOARD_UX_DESIGN.md

Usage:
  # With CSV (columns: date, symbol, spot_open, put_wall, call_wall, max_pain, high, low, close)
  python gravity_score_backtest.py --csv data/gravity_daily.csv

  # With synthetic data (demo)
  python gravity_score_backtest.py --synthetic --days 252
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

# Optional: numpy/pandas for richer stats
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def gravity_score_midpoint(
    spot: float,
    put_wall: float,
    call_wall: float,
    *,
    wall_range_floor: float = 1.0,
) -> float:
    """
    Score 0-100: how close spot is to the midpoint of put/call walls.
    Higher = more "gravitating" toward center.
    """
    if put_wall is None or call_wall is None or put_wall >= call_wall:
        return 50.0  # neutral
    center = (put_wall + call_wall) / 2.0
    wall_range = max(call_wall - put_wall, wall_range_floor)
    distance = abs(spot - center)
    # 0 distance -> 100; distance >= range -> 0
    score = max(0.0, 100.0 - (distance / wall_range) * 100.0)
    return round(score, 2)


def gravity_score_max_pain(
    spot: float,
    max_pain: float,
    put_wall: float,
    call_wall: float,
    *,
    wall_range_floor: float = 1.0,
) -> float:
    """
    Score 0-100: how close spot is to max pain, normalized by put-call range.
    """
    if max_pain is None:
        return 50.0
    wall_range = 1.0
    if put_wall is not None and call_wall is not None and call_wall > put_wall:
        wall_range = max(call_wall - put_wall, wall_range_floor)
    distance = abs(spot - max_pain)
    score = max(0.0, 100.0 - (distance / wall_range) * 100.0)
    return round(score, 2)


def gravity_score_hybrid(
    spot: float,
    put_wall: float,
    call_wall: float,
    max_pain: Optional[float] = None,
    *,
    weight_midpoint: float = 0.5,
    weight_max_pain: float = 0.5,
    wall_range_floor: float = 1.0,
) -> float:
    """
    Combine midpoint and max-pain scores. If max_pain is None, uses only midpoint.
    """
    s_mid = gravity_score_midpoint(spot, put_wall, call_wall, wall_range_floor=wall_range_floor)
    if max_pain is None:
        return s_mid
    s_mp = gravity_score_max_pain(spot, max_pain, put_wall, call_wall, wall_range_floor=wall_range_floor)
    total_w = weight_midpoint + weight_max_pain
    score = (weight_midpoint * s_mid + weight_max_pain * s_mp) / total_w
    return round(score, 2)


def realized_range_pts(high: float, low: float) -> float:
    """Daily range in points."""
    return high - low if (high is not None and low is not None) else 0.0


def load_csv(path: Path) -> List[dict]:
    """Load daily data from CSV. Expected columns: date, symbol, spot_open, put_wall, call_wall, max_pain, high, low, close."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "date": row.get("date", ""),
                    "symbol": row.get("symbol", ""),
                    "spot_open": float(row["spot_open"]) if row.get("spot_open") else None,
                    "put_wall": float(row["put_wall"]) if row.get("put_wall") else None,
                    "call_wall": float(row["call_wall"]) if row.get("call_wall") else None,
                    "max_pain": float(row["max_pain"]) if row.get("max_pain") else None,
                    "high": float(row["high"]) if row.get("high") else None,
                    "low": float(row["low"]) if row.get("low") else None,
                    "close": float(row["close"]) if row.get("close") else None,
                })
            except (ValueError, KeyError):
                continue
    return rows


def generate_synthetic_data(days: int = 252) -> List[dict]:
    """Generate synthetic daily data for demo (no real predictive power)."""
    rows = []
    # Simple random walk for spot; walls and max_pain around spot
    spot = 600.0
    for i in range(days):
        put_wall = spot - 5 - (i % 3) * 2
        call_wall = spot + 5 + (i % 4) * 2
        max_pain = spot + (i % 5 - 2) * 1.5
        # Random move
        move = (hash(str(i)) % 100) / 100.0 - 0.5
        spot = spot + move * 3
        high = spot + abs(hash(str(i + 1)) % 100) / 100.0 * 2
        low = spot - abs(hash(str(i + 2)) % 100) / 100.0 * 2
        if high < low:
            high, low = low, high
        close = low + (hash(str(i + 3)) % 100) / 100.0 * (high - low)
        rows.append({
            "date": f"2024-{(i % 252) // 21 + 1:02d}-{(i % 21) + 1:02d}",
            "symbol": "SPY",
            "spot_open": spot - move * 3,
            "put_wall": put_wall,
            "call_wall": call_wall,
            "max_pain": max_pain,
            "high": high,
            "low": low,
            "close": close,
        })
    return rows


def run_backtest(
    rows: List[dict],
    score_fn: str = "hybrid",
    num_buckets: int = 4,
) -> Tuple[List[dict], List[dict]]:
    """
    Compute gravity score and realized range per row; stratify by score buckets.
    Returns (per_day_results, bucket_summary).
    """
    results = []
    for r in rows:
        spot = r.get("spot_open") or r.get("close")
        put_wall = r.get("put_wall")
        call_wall = r.get("call_wall")
        max_pain = r.get("max_pain")
        if spot is None:
            continue
        if score_fn == "midpoint":
            score = gravity_score_midpoint(spot, put_wall or 0, call_wall or spot + 1)
        elif score_fn == "max_pain":
            score = gravity_score_max_pain(spot, max_pain or spot, put_wall or 0, call_wall or spot + 1)
        else:
            score = gravity_score_hybrid(spot, put_wall or 0, call_wall or spot + 1, max_pain)
        range_pts = realized_range_pts(r.get("high"), r.get("low"))
        results.append({
            "date": r.get("date"),
            "symbol": r.get("symbol"),
            "score": score,
            "range_pts": range_pts,
            "spot_open": spot,
        })
    if not results:
        return [], []

    # Buckets: 0-25, 25-50, 50-75, 75-100
    buckets = [[] for _ in range(num_buckets)]
    for res in results:
        s = res["score"]
        idx = min(int(s / (100 / num_buckets)), num_buckets - 1)
        buckets[idx].append(res["range_pts"])

    bucket_summary = []
    for i in range(num_buckets):
        lo = i * (100 / num_buckets)
        hi = (i + 1) * (100 / num_buckets)
        vals = buckets[i]
        n = len(vals)
        if HAS_NUMPY and n > 0:
            bucket_summary.append({
                "bucket": f"{lo:.0f}-{hi:.0f}",
                "count": n,
                "mean_range": round(float(np.mean(vals)), 2),
                "median_range": round(float(np.median(vals)), 2),
                "std_range": round(float(np.std(vals)), 2),
            })
        else:
            bucket_summary.append({
                "bucket": f"{lo:.0f}-{hi:.0f}",
                "count": n,
                "mean_range": round(sum(vals) / n, 2) if n else 0.0,
                "median_range": round(sorted(vals)[n // 2], 2) if n else 0.0,
                "std_range": 0.0,
            })
    return results, bucket_summary


def main():
    parser = argparse.ArgumentParser(description="Gravity score backtest: pin vs volatile")
    parser.add_argument("--csv", type=Path, help="Path to daily CSV (date, symbol, spot_open, put_wall, call_wall, max_pain, high, low, close)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demo")
    parser.add_argument("--days", type=int, default=252, help="Days of synthetic data")
    parser.add_argument("--score", choices=["midpoint", "max_pain", "hybrid"], default="hybrid", help="Score formula")
    parser.add_argument("--buckets", type=int, default=4, help="Number of score buckets")
    args = parser.parse_args()

    if args.synthetic:
        rows = generate_synthetic_data(args.days)
        print(f"Generated {len(rows)} synthetic days")
    elif args.csv and args.csv.exists():
        rows = load_csv(args.csv)
        print(f"Loaded {len(rows)} rows from {args.csv}")
    else:
        print("Provide --csv <path> or --synthetic. See docs/TRADING_DASHBOARD_UX_DESIGN.md")
        return

    results, bucket_summary = run_backtest(rows, score_fn=args.score, num_buckets=args.buckets)
    print(f"\nScore formula: {args.score}")
    print("\nBucket summary (higher score = more pinning expected → lower range):")
    print("-" * 60)
    for b in bucket_summary:
        print(f"  Score {b['bucket']}: n={b['count']}  mean_range={b['mean_range']:.2f}  median_range={b['median_range']:.2f}")
    print("-" * 60)

    if HAS_NUMPY and len(results) >= 2:
        scores = [r["score"] for r in results]
        ranges = [r["range_pts"] for r in results]
        corr = np.corrcoef(scores, ranges)[0, 1]
        print(f"Correlation(score, range): {corr:.3f} (negative = high score → low range, as expected)")
    return


if __name__ == "__main__":
    main()
