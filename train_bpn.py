#!/usr/bin/env python3
"""Train Whale-VAD with the BPN paper's recipe (Geldenhuys et al., 2510.21280v2).

Differs from the DCASE-faithful ``train.py`` in three ways, all
captured by :class:`BPNTrainingConfig` defaults:

* Upgraded depthwise aggregation block (dilation [2, 4, 8] +
  residual skips + spatial dropout, Section V.A).
* Training recipe from Section V.B.5 — AdamW with lr=1e-3,
  weight_decay=0.01, batch=48, focal loss, ~32 epochs.
* Optional BPN gating head (Section V.B); off by default because
  the proposal network is experimental.

Usage example::

    python train_bpn.py \\
        --train-root data/2026_BioDCASE_development_set/train \\
        --val-root   data/2026_BioDCASE_development_set/validation \\
        --output-dir runs/whale_vad_bpn \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict

from whalevad.bpn import BPNTrainingConfig
from whalevad.bpn.trainer import BPNTrainer


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Expose every BPNTrainingConfig field as a CLI flag."""
    defaults = BPNTrainingConfig()
    for f in fields(BPNTrainingConfig):
        default = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        # tuple fields are accepted as comma-separated strings
        if isinstance(default, tuple):
            parser.add_argument(
                flag,
                dest=f.name,
                type=str,
                default=",".join(str(x) for x in default),
                help=f"comma-separated, default {default}",
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
        kwargs: Dict[str, Any] = {"dest": f.name, "help": f"(default: {default})"}
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            kwargs["type"] = type(default)
            kwargs["default"] = default
        else:
            kwargs["type"] = str
            kwargs["default"] = default
        parser.add_argument(flag, **kwargs)


def _parse_tuple(raw: str, kind):
    if raw == "" or raw == "()":
        return ()
    return tuple(kind(x.strip()) for x in raw.split(",") if x.strip())


def _build_config(ns: argparse.Namespace) -> BPNTrainingConfig:
    defaults = BPNTrainingConfig()
    overrides: Dict[str, Any] = {}
    for f in fields(BPNTrainingConfig):
        v = getattr(ns, f.name)
        default = getattr(defaults, f.name)
        if isinstance(default, tuple):
            # all our tuple fields are int- or float-valued
            kind = type(default[0]) if default else float
            v = _parse_tuple(v if isinstance(v, str) else ",".join(map(str, v)), kind)
        overrides[f.name] = v
    return BPNTrainingConfig(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Whale-VAD with the BPN recipe.")
    _add_config_args(parser)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON file with BPNTrainingConfig overrides.",
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
        # Tuples aren't JSON-serialisable by default — convert in-place.
        cfg_dict = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in asdict(cfg).items()
        }
        json.dump(cfg_dict, fh, indent=2)
    BPNTrainer(cfg).fit()


if __name__ == "__main__":
    main()
