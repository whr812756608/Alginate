"""Stage 1 model constants and file conventions."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "models.json"


def load_config(path=None):
    with open(CONFIG if path is None else path, encoding="utf-8") as handle:
        spec = json.load(handle)
    if set(spec["models"]) != {str(dp) for dp in range(3, 8)}:
        raise ValueError("Expected DP3–DP7 model metadata")
    return spec


def selected_model(dp, config=None):
    c = load_config() if config is None else config
    return c["models"][str(int(dp))]
