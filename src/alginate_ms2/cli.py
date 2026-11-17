"""CLI entrypoints: fail closed until the selected numerical engines pass parity."""
from __future__ import annotations
import argparse
from .config import load_models

def run_stage(stage, args=None):
    parser = argparse.ArgumentParser(description=f"Alginate publication stage: {stage}")
    parser.add_argument("--dp", nargs="+", default=["all"])
    parser.add_argument("--nce", nargs="+", default=["all"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parsed = parser.parse_args(args)
    load_models()
    parser.error(
        f"{stage}: full scientific engine is not yet integrated or verified. "
        "This branch contains the approved structure, configuration, and LOO splitter, "
        "NOT a validated reproducer of manuscript predictions."
    )
