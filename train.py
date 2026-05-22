#!/usr/bin/env python3
"""Whale-VAD training entry point.

Usage example::

    python train.py \
        --train-root data/biodcase_development_set/train \
        --val-root   data/biodcase_development_set/validation \
        --output-dir runs/whale_vad_paper \
        --num-classes 3 --loss focal --epochs 100

The defaults follow Geldenhuys et al. (2025, DCASE).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict

from whalevad.training import TrainingConfig, Trainer


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add one CLI flag per ``TrainingConfig`` field."""
    for f in fields(TrainingConfig):
        default = getattr(TrainingConfig(), f.name)
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or isinstance(default, bool):
            parser.add_argument(
                flag,
                dest=f.name,
                action="store_true" if not default else "store_false",
                help=f"(default: {default})",
            )
            continue
        kwargs: Dict[str, Any] = {"dest": f.name, "help": f"(default: {default})"}
        if isinstance(default, (int, float)):
            kwargs["type"] = type(default)
        else:
            kwargs["type"] = str
        kwargs["default"] = default
        parser.add_argument(flag, **kwargs)


def _build_config(ns: argparse.Namespace) -> TrainingConfig:
    overrides = {f.name: getattr(ns, f.name) for f in fields(TrainingConfig)}
    return TrainingConfig(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Whale-VAD")
    _add_config_args(parser)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON file with TrainingConfig overrides.",
    )
    ns = parser.parse_args()

    if ns.config:
        with open(ns.config) as fh:
            data = json.load(fh)
        for k, v in data.items():
            if hasattr(ns, k):
                setattr(ns, k, v)

    cfg = _build_config(ns)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(Path(cfg.output_dir) / "train.log"),
            logging.StreamHandler(),
        ],
    )
    with open(Path(cfg.output_dir) / "config.json", "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    trainer = Trainer(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
