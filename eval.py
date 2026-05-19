#!/usr/bin/env python3
"""Whale-VAD evaluation entry point.

Loads a trained checkpoint, runs the full Section 5.8 inference
pipeline (overlap-averaged tiles → per-class threshold sweep → median
filter + call merging + duration filter), and reports both frame-level
and event-level F1.  Scans multiple segment lengths so you can pick the
tiling that gives the best dev F1.

Usage::

    python eval.py \\
        --checkpoint runs/whale_vad_2026/best.pt \\
        --val-root   data/2026_BioDCASE_development_set/validation \\
        --output-dir runs/whale_vad_2026/eval \\
        --segment-lengths-s 30,45,60,75 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Tuple

from whalevad.training.evaluator import EvalConfig, Evaluator


def _parse_segments(raw: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    defaults = EvalConfig(checkpoint="", val_root="")  # for default values
    for f in fields(EvalConfig):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if f.name == "segment_lengths_s":
            parser.add_argument(
                flag,
                dest=f.name,
                type=str,
                default=",".join(str(s) for s in default),
                help=f"Comma-separated list of segment lengths in seconds (default: {default})",
            )
            continue
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                dest=f.name,
                action="store_true" if not default else "store_false",
                help=f"(default: {default})",
            )
            continue
        if isinstance(default, (int, float)):
            parser.add_argument(flag, dest=f.name, type=type(default), default=default,
                                help=f"(default: {default})")
        else:
            parser.add_argument(flag, dest=f.name, type=str, default=default,
                                help=f"(default: {default!r})")


def _build_config(ns: argparse.Namespace) -> EvalConfig:
    overrides: Dict[str, Any] = {}
    for f in fields(EvalConfig):
        val = getattr(ns, f.name)
        if f.name == "segment_lengths_s" and isinstance(val, str):
            val = _parse_segments(val)
        overrides[f.name] = val
    return EvalConfig(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Whale-VAD checkpoint.")
    _add_config_args(parser)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON file with EvalConfig overrides.",
    )
    ns = parser.parse_args()

    if ns.config:
        with open(ns.config) as fh:
            data = json.load(fh)
        for k, v in data.items():
            if hasattr(ns, k):
                setattr(ns, k, v)

    if not ns.checkpoint:
        raise SystemExit("--checkpoint is required")
    if not ns.val_root:
        raise SystemExit("--val-root is required")

    cfg = _build_config(ns)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(cfg.output_dir) / "eval.log"),
            logging.StreamHandler(),
        ],
    )
    with open(Path(cfg.output_dir) / "eval_config.json", "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    Evaluator(cfg).run()


if __name__ == "__main__":
    main()
