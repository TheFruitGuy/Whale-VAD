"""Training loop for Whale-VAD (paper-faithful).

This module wires together the dataset, the stochastic negative
sampler, the spectrogram extractor with phase information, the
``WhaleVADClassifier``, the loss functions, and the postprocessing /
threshold optimisation described in the paper.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..model import WhaleVADClassifier, WhaleVADModel
from ..spectrogram import SpectrogramExtractor
from .augmentations import NoisePerturbation, SpecAugment
from .config import TrainingConfig
from .dataset import (
    ATBFLDataset,
    SEVEN_TO_THREE,
    CLASS_MAP_3,
    CLASS_MAP_7,
    build_eval_segments,
    build_positive_segments,
    collate_segments,
    load_audio_files,
    resolve_class_names,
    map_label_to_class,
)
from .losses import (
    FocalLoss,
    MultiObjectiveLoss,
    WeightedBCELoss,
    compute_bce_pos_weight,
)
from .metrics import (
    find_optimal_thresholds,
    macro_f1,
)
from ._progress import progress
from .postprocessing import collapse_seven_to_three
from .sampling import StochasticNegativeSampler


log = logging.getLogger("whalevad.train")


# ----------------------------------------------------------------- builders


def build_spectrogram_extractor(cfg: TrainingConfig) -> SpectrogramExtractor:
    return SpectrogramExtractor(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        window_fn=cfg.window_fn,
        complex_repr=cfg.complex_repr,
        norm_features=cfg.norm_features,
        power=cfg.power,
    )


def build_classifier(cfg: TrainingConfig) -> WhaleVADClassifier:
    return WhaleVADClassifier(
        num_classes=cfg.num_classes,
        feat_channels=cfg.feat_channels,
        include_bottleneck_layers=cfg.include_bottleneck_layers,
        include_aggregation_layers=cfg.include_aggregation_layers,
        include_bounding_boxes=cfg.include_bounding_boxes,
        num_anchors=cfg.num_anchors,
    )


# -------------------------------------------------------------- the trainer


class Trainer:
    """Drives the training and validation loops described in the paper."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._setup_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # -------------------- data
        train_cache = self._resolve_index_cache_path(cfg.train_root)
        val_cache = self._resolve_index_cache_path(cfg.val_root)
        log.info("Indexing training set: %s", cfg.train_root)
        self.train_audio = load_audio_files(
            Path(cfg.train_root),
            audio_subdir=cfg.audio_subdir,
            annotation_subdir=cfg.annotation_subdir,
            audio_ext=cfg.audio_ext,
            annotation_ext=cfg.annotation_ext,
            max_workers=cfg.audio_index_workers,
            index_cache_path=train_cache,
        )
        log.info("Indexing validation set: %s", cfg.val_root)
        self.val_audio = load_audio_files(
            Path(cfg.val_root),
            audio_subdir=cfg.audio_subdir,
            annotation_subdir=cfg.annotation_subdir,
            audio_ext=cfg.audio_ext,
            annotation_ext=cfg.annotation_ext,
            max_workers=cfg.audio_index_workers,
            index_cache_path=val_cache,
        )

        self._rng = random.Random(cfg.seed)
        self.class_names = (
            tuple(cfg.class_names)
            if cfg.class_names is not None
            else resolve_class_names(cfg.num_classes)
        )

        # Positive segments are deterministic across epochs (Section 5.5).
        self.positive_segments = build_positive_segments(
            self.train_audio,
            num_classes=cfg.num_classes,
            collar_min_s=cfg.collar_min_s,
            collar_max_s=cfg.collar_max_s,
            rng=self._rng,
        )
        if len(self.positive_segments) == 0:
            raise RuntimeError(
                "No positive segments could be built — check the dataset paths and"
                " annotation parsing."
            )
        log.info("Built %d positive training segments", len(self.positive_segments))

        # Negative-segment pool generator.
        self.neg_sampler = StochasticNegativeSampler(
            self.train_audio,
            min_dur_s=cfg.neg_min_dur_s,
            max_dur_s=cfg.neg_max_dur_s,
            rng=self._rng,
        )
        if not self.neg_sampler.has_negatives:
            log.warning(
                "No usable negative regions found — training will use only positives."
            )

        # Validation set: dense tiling per Section 5.1.
        self.val_segments = build_eval_segments(
            self.val_audio,
            segment_s=cfg.eval_segment_s,
            overlap_s=cfg.eval_overlap_s,
        )
        log.info("Built %d validation tiles", len(self.val_segments))

        # -------------------- datasets / dataloaders
        self.train_dataset = ATBFLDataset(
            segments=list(self.positive_segments),
            num_classes=cfg.num_classes,
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            class_names=self.class_names,
        )
        self.val_dataset = ATBFLDataset(
            segments=self.val_segments,
            num_classes=cfg.num_classes,
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            class_names=self.class_names,
        )

        # -------------------- model + transform
        self.transform = build_spectrogram_extractor(cfg).to(self.device)
        self.classifier = build_classifier(cfg).to(self.device)
        self.model = WhaleVADModel(self.classifier, self.transform).to(self.device)

        # -------------------- augmentations (off by default; see Section 5.4)
        self.noise_aug = (
            NoisePerturbation(target_snr_db=cfg.noise_snr_db).to(self.device)
            if cfg.use_noise_perturbation
            else None
        )
        self.spec_aug = (
            SpecAugment(
                freq_mask=cfg.specaugment_freq_mask,
                time_mask=cfg.specaugment_time_mask,
                num_freq_masks=cfg.specaugment_num_freq_masks,
                num_time_masks=cfg.specaugment_num_time_masks,
            ).to(self.device)
            if cfg.use_specaugment
            else None
        )

        # -------------------- losses (Section 5.6)
        pos_weight = self._compute_pos_weight()
        bce = WeightedBCELoss(pos_weight=pos_weight.to(self.device))
        if cfg.loss_type == "bce":
            cls_loss: Module = bce
        elif cfg.loss_type == "focal":
            cls_loss = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        else:
            raise ValueError(f"Unknown loss_type={cfg.loss_type}")
        # Always keep a plain weighted BCE on hand for the validation
        # selection metric (Section 5.8 uses lowest BCE dev loss).
        self.val_bce = WeightedBCELoss(pos_weight=pos_weight.to(self.device))
        self.objective = MultiObjectiveLoss(
            cls_loss,
            regression_weight=cfg.regression_loss_weight,
        )

        # -------------------- optimiser
        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        # -------------------- state
        self.epoch = 0
        self.global_step = 0
        self.best_val_metric = math.inf  # we minimise BCE loss by default
        self.history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------- setup

    def _resolve_index_cache_path(self, dataset_root: str) -> Optional[Path]:
        """Pick the on-disk audio-index cache location for ``dataset_root``.

        Empty string -> use the dataset's own ``.whalevad_audio_index.json``;
        ``"none"`` -> disable caching; otherwise resolve as a path.  When the
        config supplies a single path for two splits we suffix it with the
        split name to keep them separate.
        """
        raw = (self.cfg.audio_index_cache_path or "").strip()
        if not raw:
            return None  # let load_audio_files default to <root>/.whalevad_audio_index.json
        if raw.lower() in {"none", "/dev/null"}:
            return Path("/dev/null")
        p = Path(raw)
        # If the same path is configured for both splits, namespace it.
        return p / f"{Path(dataset_root).name}.json" if p.is_dir() else p

    def _setup_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_pos_weight(self) -> Tensor:
        """Per-class :math:`w_c = N / P_c` weighting (Section 5.6).

        ``N`` is the total number of *negative* segments visible to the
        model in one epoch — i.e. the resampled negatives.  In the
        absence of a known per-epoch negative count we use the
        positive segment count as a rough proxy (the paper balances
        these per mini-batch).
        """
        cfg = self.cfg
        class_to_idx = {n: i for i, n in enumerate(self.class_names)}
        pos_counts = torch.zeros((cfg.num_classes,), dtype=torch.float32)
        for seg in self.positive_segments:
            for ann in seg.annotations:
                cls = map_label_to_class(ann.label, cfg.num_classes)
                if cls is None:
                    continue
                pos_counts[class_to_idx[cls]] += 1.0
        n_neg = max(1, int(len(self.positive_segments) * cfg.pos_to_neg_ratio))
        return compute_bce_pos_weight(pos_counts, n_neg)

    # ----------------------------------------------------- epoch boundary

    def _resample_negatives(self) -> None:
        cfg = self.cfg
        num_neg = int(round(cfg.pos_to_neg_ratio * len(self.positive_segments)))
        negatives = (
            self.neg_sampler.sample(num_neg)
            if self.neg_sampler.has_negatives
            else []
        )
        merged = list(self.positive_segments) + list(negatives)
        self._rng.shuffle(merged)
        self.train_dataset.set_segments(merged)
        log.info(
            "Epoch %d: %d positives + %d negatives = %d segments",
            self.epoch,
            len(self.positive_segments),
            len(negatives),
            len(merged),
        )

    def _make_dataloader(self, dataset: ATBFLDataset, *, shuffle: bool) -> DataLoader:
        cfg = self.cfg
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory and torch.cuda.is_available(),
            collate_fn=collate_segments,
            persistent_workers=cfg.num_workers > 0,
            drop_last=shuffle,
        )

    # ---------------------------------------------------------------- run

    def fit(self) -> Dict[str, Any]:
        for epoch in range(1, self.cfg.epochs + 1):
            self.epoch = epoch
            t0 = time.time()
            self._resample_negatives()
            train_loader = self._make_dataloader(self.train_dataset, shuffle=True)
            train_metrics = self._train_one_epoch(train_loader)
            val_loader = self._make_dataloader(self.val_dataset, shuffle=False)
            val_metrics = self.validate(val_loader)
            elapsed = time.time() - t0
            entry = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "elapsed_s": elapsed,
            }
            self.history.append(entry)
            self._save_history()
            self._save_checkpoint(val_metrics, latest=True)
            self._maybe_save_best(val_metrics)
            log.info(
                "[epoch %d] train_loss=%.4f val_bce=%.4f val_f1=%.4f (%.1fs)",
                epoch,
                train_metrics.get("loss", float("nan")),
                val_metrics.get("bce_loss", float("nan")),
                val_metrics.get("macro_f1", float("nan")),
                elapsed,
            )
        return {"history": self.history, "best_val_metric": self.best_val_metric}

    # -------------------------------------------------------- training

    def _train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running: Dict[str, float] = {}
        n_batches = 0
        bar = progress(
            loader,
            desc=f"train ep{self.epoch}",
            total=len(loader),
            unit="batch",
            leave=False,
        )
        for it, batch in enumerate(bar):
            self.optimizer.zero_grad(set_to_none=True)
            losses = self._forward_loss(batch, train=True)
            losses["loss"].backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.classifier.parameters(), self.cfg.grad_clip
                )
            self.optimizer.step()
            self.global_step += 1
            loss_val = float(losses["loss"].detach().item())
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach().item())
            n_batches += 1
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(loss=f"{loss_val:.4f}")
            if it % self.cfg.log_interval == 0:
                log.info("  ep%d it%d loss=%.4f", self.epoch, it, loss_val)
        return {k: v / max(1, n_batches) for k, v in running.items()}

    # ------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_probs: List[Tensor] = []
        all_targets: List[Tensor] = []
        bce_sum = 0.0
        n_batches = 0
        bar = progress(
            loader,
            desc=f"val ep{self.epoch}",
            total=len(loader),
            unit="batch",
            leave=False,
        )
        for batch in bar:
            forward = self._forward_logits(batch)
            logits = forward["logits"]
            targets = forward["targets"]
            frame_lengths = forward["frame_lengths"]
            bce = self.val_bce(logits, targets, frame_lengths)
            bce_sum += float(bce.item())
            probs = torch.sigmoid(logits).cpu()
            tgt = targets.cpu()
            for b in range(probs.size(0)):
                T = int(frame_lengths[b].item())
                all_probs.append(probs[b, :T])
                all_targets.append(tgt[b, :T])
            n_batches += 1

        if not all_probs:
            return {"bce_loss": float("nan"), "macro_f1": float("nan")}

        probs_flat = torch.cat(all_probs, dim=0)
        targets_flat = torch.cat(all_targets, dim=0)

        if (
            self.cfg.num_classes == 7
            and self.cfg.collapse_eval_to_three
        ):
            probs_flat = collapse_seven_to_three(probs_flat)
            targets_flat = _collapse_target_to_three(targets_flat)

        thresholds, per_class = find_optimal_thresholds(
            probs_flat,
            targets_flat,
            grid=self.cfg.threshold_search_grid,
        )
        f1 = macro_f1(per_class)
        out: Dict[str, float] = {
            "bce_loss": bce_sum / max(1, n_batches),
            "macro_f1": f1,
            "thresholds": thresholds.tolist(),
        }
        for i, m in enumerate(per_class):
            label = (
                CLASS_MAP_3[i]
                if (self.cfg.num_classes == 3
                    or (self.cfg.num_classes == 7 and self.cfg.collapse_eval_to_three))
                else CLASS_MAP_7[i]
            )
            out[f"precision_{label}"] = m.precision
            out[f"recall_{label}"] = m.recall
            out[f"f1_{label}"] = m.f1
        return out

    # ---------------------------------------------------------- internals

    def _forward_logits(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        audio = batch["audio"].to(self.device)
        if self.noise_aug is not None and self.model.training:
            audio = self.noise_aug(audio)

        frame_lengths = batch["frame_length"].to(self.device)
        targets = batch["frame_labels"].to(self.device)

        features, _ = self.transform(audio, lengths=frame_lengths)
        if self.spec_aug is not None and self.model.training:
            features = self.spec_aug(features)

        logits, _, opts = self.classifier(features, lab_lengths=frame_lengths)
        return {
            "logits": logits,
            "targets": targets,
            "frame_lengths": frame_lengths,
            "opts": opts,
        }

    def _forward_loss(self, batch: Dict[str, Tensor], *, train: bool) -> Dict[str, Tensor]:
        cfg = self.cfg
        forward = self._forward_logits(batch)
        kwargs: Dict[str, Any] = {
            "logits": forward["logits"],
            "targets": forward["targets"],
            "frame_lengths": forward["frame_lengths"],
        }
        if cfg.include_bounding_boxes:
            kwargs["bbox_pred"] = forward["opts"].get("bounding_box_reg")
            kwargs["bbox_conf"] = forward["opts"].get("bounding_box_conf")
            kwargs["bbox_target"] = batch["bboxes"].to(self.device)
            kwargs["bbox_mask"] = batch["bbox_mask"].to(self.device)
        return self.objective(**kwargs)

    # ---------------------------------------------------------- I/O

    def _save_checkpoint(self, val_metrics: Dict[str, float], *, latest: bool) -> Path:
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "classifier_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "val_metrics": val_metrics,
        }
        path = self.output_dir / ("latest.pt" if latest else f"epoch_{self.epoch}.pt")
        torch.save(ckpt, path)
        return path

    def _maybe_save_best(self, val_metrics: Dict[str, float]) -> None:
        # Section 5.8: "the best model was chosen based on the lowest BCE
        # development loss".  ``select_metric`` allows overriding for
        # experimentation but defaults to bce_loss.
        metric_name = self.cfg.select_metric
        if metric_name == "bce_loss":
            value = val_metrics.get("bce_loss", math.inf)
            improved = value < self.best_val_metric
        elif metric_name == "f1":
            value = -val_metrics.get("macro_f1", -math.inf)
            improved = value < self.best_val_metric
        else:
            raise ValueError(f"Unknown select_metric={metric_name}")
        if improved:
            self.best_val_metric = value
            best = self.output_dir / "best.pt"
            torch.save(
                {
                    "epoch": self.epoch,
                    "classifier_state_dict": self.classifier.state_dict(),
                    "config": asdict(self.cfg),
                    "val_metrics": val_metrics,
                },
                best,
            )
            log.info(
                "  new best %s=%.4f -> %s",
                metric_name,
                val_metrics.get(metric_name, value),
                best,
            )

    def _save_history(self) -> None:
        with open(self.output_dir / "history.json", "w") as fh:
            json.dump(self.history, fh, indent=2)


# -------------------------------------------------------- helper functions


def _collapse_target_to_three(targets: Tensor) -> Tensor:
    """Reduce a 7-class binary target tensor to the 3-class problem."""
    if targets.size(-1) != 7:
        raise ValueError("Expected 7-class target tensor")
    out_channels: List[Tensor] = []
    for cls in CLASS_MAP_3:
        members = [i for i, name in enumerate(CLASS_MAP_7) if SEVEN_TO_THREE[name] == cls]
        out_channels.append(targets[..., members].max(dim=-1).values)
    return torch.stack(out_channels, dim=-1)
