"""Read frozen publication-model metadata without silently filling missing choices."""
from __future__ import annotations
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.json"

def load_models(path=None):
    p = Path(path) if path is not None else CONFIG_PATH
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if set(data["models"]) != {"3", "4", "5", "6", "7"}:
        raise ValueError("Expected exactly DP3–DP7")
    for dp, spec in data["models"].items():
        if len(spec["loo_mae_pp_by_nce"]) != 3:
            raise ValueError(f"DP{dp}: expected all three NCEs")
        if spec["ratio_policy"] == "frozen" and len(spec.get("ratios") or []) != 8:
            raise ValueError(f"DP{dp}: frozen K8 definition incomplete")
    return data
