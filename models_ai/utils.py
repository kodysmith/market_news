import os
import random
import yaml
import torch
import numpy as np


class Config:
    """Nested dict-like config with attribute access."""

    def __init__(self, d=None):
        if d is None:
            d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return f"Config({self.__dict__})"

    def to_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def override(self, flat_overrides):
        """Apply dot-separated overrides, e.g. {'ssan.alpha': 0.1}."""
        for key, val in flat_overrides.items():
            parts = key.split(".")
            obj = self
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], val)
        return self

    def copy(self):
        return Config(self.to_dict())


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg_device="auto"):
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(cfg_device)


def ensure_dirs():
    for d in ["results/logs", "results/checkpoints", "results/plots", "results/diagnostics"]:
        os.makedirs(d, exist_ok=True)
