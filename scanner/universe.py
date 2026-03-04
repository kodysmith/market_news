"""
Universe Service: candidate symbols for premarket scan (small/mid-cap, filters).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> Dict[str, Any]:
    """Load scanner/config from data/config.json."""
    path = ROOT / "data" / "config.json"
    if not path.is_file():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def get_scanner_universe_path(config: Dict[str, Any]) -> Path:
    """Path to universe list (one symbol per line or JSON list)."""
    p = config.get("SCANNER_UNIVERSE_PATH")
    if p:
        return Path(p)
    # Default: dedicated file, else fall back to sec_universe
    candidate = ROOT / "data" / "scanner_universe.txt"
    if candidate.is_file():
        return candidate
    return ROOT / "data" / "sec_universe.json"


def load_symbols_from_txt(path: Path) -> List[str]:
    symbols = []
    with open(path) as f:
        for line in f:
            s = line.strip().upper()
            if s and not s.startswith("#"):
                symbols.append(s)
    return symbols


def load_symbols_from_sec_json(path: Path) -> List[str]:
    with open(path) as f:
        data = json.load(f)
    constituents = data.get("constituents", [])
    out = []
    for c in constituents:
        t = c.get("ticker") or c.get("symbol")
        if t:
            out.append(str(t).strip().upper())
    return list(dict.fromkeys(out))


def get_candidates(
    config: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[str]:
    """
    Return list of candidate symbols for the scanner.
    V1: from cached universe file (scanner_universe.txt or sec_universe.json).
    Optional limit and price filters (applied later when we have prices).
    """
    cfg = config or load_config()
    path = get_scanner_universe_path(cfg)
    if not path.is_file():
        # Fallback: small list for testing
        return []
    if path.suffix.lower() == ".json" and "sec_universe" in path.name:
        symbols = load_symbols_from_sec_json(path)
    else:
        symbols = load_symbols_from_txt(path)
    if limit is not None and limit > 0:
        symbols = symbols[:limit]
    return symbols
