from __future__ import annotations

import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # reversal/


def path(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def load_config(file: str = "config.yaml") -> dict:
    with open(path(file), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    proxy = cfg.get("data", {}).get("proxy")
    if proxy:
        os.environ.setdefault("HTTPS_PROXY", proxy)
        os.environ.setdefault("HTTP_PROXY", proxy)
    return cfg


def override(cfg: dict, changes: dict) -> dict:
    """changes 形如 {"signal.z_max": -2.5, "portfolio.hedge": True}，返回新配置。"""
    out = copy.deepcopy(cfg)
    for k, v in changes.items():
        d = out
        keys = k.split(".")
        for kk in keys[:-1]:
            d = d.setdefault(kk, {})
        d[keys[-1]] = v
    return out
