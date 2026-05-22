"""Evaluation entry point for trained Whale-VAD checkpoints.

Implements the full inference + scoring pipeline described in Section 5.8
of the paper:

1. Tile each validation recording into fixed-length segments with a
   configurable overlap.
2. Run the model to obtain per-frame call probabilities.
3. Average overlapping windows per recording (Section 5.1).
4. Sweep per-class thresholds to maximise frame-level F1 over the
   whole dev set (Section 5.8).
5. Run the median-filter + threshold + merge + duration-filter
   pipeline (Section 5.8) to obtain detected call events.
6. Compute both frame-level and event-level F1.

Used by ``eval.py`` to scan multiple ``segment_s`` settings (e.g.
30 / 45 / 60 / 75 s) so the practitioner can pick the tiling that
gives the best dev F1.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..model import WhaleVADClassifier
from ..spectrogram import SpectrogramExtractor
from ._progress import progress
from .dataset import (
    ATBFLDataset,
    AudioFile,
    CLASS_MAP_3,
    build_eval_segments,
    collate_segments,
    load_audio_files,
    map_label_to_class,
    resolve_class_names,
)
from .metrics import find_optimal_thresholds
from .postprocessing import (
    CallEvent,
    collapse_seven_to_three,
    merge_overlapping_windows,
    probabilities_to_calls,
)


log = logging.getLogger("whalevad.eval")


# --------------------------------------------------------------- config


@dataclass
class EvalConfig:
    checkpoint: str
    val_root: str
    output_dir: str = "runs/eval"

    audio_subdir: str = "audio"
    annotation_subdir: str = "annotations"
    audio_ext: str = ".wav"
    annotation_ext: str = ".csv"

    segment_lengths_s: Tuple[float, ...] = (30.0, 45.0, 60.0, 75.0)
    overlap_s: float = 2.0

    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"
    threshold_search_grid: int = 101

    # Event-matching IoU threshold for the event-level F1.  0.3 mirrors
    # the BioDCASE default, but it's worth scanning higher values too.
    event_iou_threshold: float = 0.3

    # Postprocessing knobs (defaults match the paper).
    median_kernel_ms: float = 500.0
    merge_gap_s: float = 0.5
    min_call_duration_s: float = 0.5
    max_call_duration_s: float = 30.0

    audio_index_workers: int = 16
    audio_index_cache_path: str = ""


# ---------------------------------------------------- result containers


@dataclass
class PerClassResult:
    label: str
    threshold: float
    frame_tp: int
    frame_fp: int
    frame_fn: int
    frame_precision: float
    frame_recall: float
    frame_f1: float
    event_tp: int
    event_fp: int
    event_fn: int
    event_precision: float
    event_recall: float
    event_f1: float


@dataclass
class SegmentLengthResult:
    segment_s: float
    overlap_s: float
    per_class: List[PerClassResult] = field(default_factory=list)

    @property
    def frame_macro_f1(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(c.frame_f1 for c in self.per_class) / len(self.per_class)

    @property
    def event_macro_f1(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(c.event_f1 for c in self.per_class) / len(self.per_class)


# --------------------------------------------------- ground-truth events


def _ground_truth_events_for_file(
    af: AudioFile, *, num_classes: int, eval_class_names: Sequence[str]
) -> List[CallEvent]:
    events: List[CallEvent] = []
    for ann in af.annotations:
        cls = map_label_to_class(ann.label, num_classes)
        if cls is None:
            continue
        if num_classes == 7 and len(eval_class_names) == 3:
            # Map from 7-class label down to the 3-class evaluation label.
            cls = map_label_to_class(ann.label, 3) or cls
        if cls not in eval_class_names:
            continue
        events.append(
            CallEvent(
                onset_s=ann.onset_s,
                offset_s=ann.offset_s,
                label=cls,
            )
        )
    return events


# ------------------------------------------------------ event matching


def _event_iou(a: CallEvent, b: CallEvent) -> float:
    inter = max(0.0, min(a.offset_s, b.offset_s) - max(a.onset_s, b.onset_s))
    if inter <= 0:
        return 0.0
    union = (a.offset_s - a.onset_s) + (b.offset_s - b.onset_s) - inter
    return inter / union if union > 0 else 0.0


def event_level_metrics(
    detected: Sequence[CallEvent],
    truth: Sequence[CallEvent],
    *,
    label: str,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    """Greedy IoU-matching: each ground-truth event consumes its best
    detected event whose IoU passes ``iou_threshold``.  Returns
    ``(tp, fp, fn)`` for the supplied label.
    """
    det = [e for e in detected if e.label == label]
    tru = [e for e in truth if e.label == label]
    used = [False] * len(det)
    tp = 0
    for g in tru:
        best_j, best_iou = -1, 0.0
        for j, d in enumerate(det):
            if used[j]:
                continue
            iou = _event_iou(g, d)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    fn = len(tru) - tp
    fp = sum(1 for u in used if not u)
    return tp, fp, fn


# -------------------------------------------------- the Evaluator class


class Evaluator:
    """Runs the full eval pipeline for a saved checkpoint."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading checkpoint: %s", cfg.checkpoint)
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        self._ckpt_config: Dict[str, Any] = ckpt.get("config", {})
        if not self._ckpt_config:
            raise RuntimeError(
                "Checkpoint has no embedded config — was it produced by this trainer?"
            )

        # Resolve class layout.
        num_classes = int(self._ckpt_config.get("num_classes", 3))
        self.num_classes = num_classes
        self.train_class_names = resolve_class_names(num_classes)
        # The paper always reports on the 3-class problem.
        self.eval_class_names: Tuple[str, ...] = CLASS_MAP_3
        self.collapse_eval_to_three = (
            num_classes == 7
            and bool(self._ckpt_config.get("collapse_eval_to_three", True))
        )

        # Audio + spectrogram settings come from the trained config so
        # the eval pipeline matches what the model was trained on.
        self.sample_rate = int(self._ckpt_config["sample_rate"])
        self.n_fft = int(self._ckpt_config["n_fft"])
        self.hop_length = int(self._ckpt_config["hop_length"])
        self.hop_s = self.hop_length / float(self.sample_rate)

        self.transform = SpectrogramExtractor(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=int(self._ckpt_config.get("win_length", self.n_fft)),
            hop_length=self.hop_length,
            window_fn=self._ckpt_config.get("window_fn", "hann_window"),
            complex_repr=self._ckpt_config.get("complex_repr", "trig"),
            norm_features=self._ckpt_config.get("norm_features", "demean"),
            power=self._ckpt_config.get("power", None),
        ).to(self.device)

        self.classifier = WhaleVADClassifier(
            num_classes=num_classes,
            feat_channels=int(self._ckpt_config.get("feat_channels", 3)),
            include_bottleneck_layers=bool(
                self._ckpt_config.get("include_bottleneck_layers", True)
            ),
            include_aggregation_layers=bool(
                self._ckpt_config.get("include_aggregation_layers", True)
            ),
            include_bounding_boxes=bool(
                self._ckpt_config.get("include_bounding_boxes", False)
            ),
            num_anchors=int(self._ckpt_config.get("num_anchors", 64)),
        ).to(self.device)
        self.classifier.load_state_dict(ckpt["classifier_state_dict"])
        self.classifier.eval()

        log.info("Indexing validation set: %s", cfg.val_root)
        self.val_audio = load_audio_files(
            Path(cfg.val_root),
            audio_subdir=cfg.audio_subdir,
            annotation_subdir=cfg.annotation_subdir,
            audio_ext=cfg.audio_ext,
            annotation_ext=cfg.annotation_ext,
            max_workers=cfg.audio_index_workers,
        )

    # ------------------------------------------------------------ public

    def run(self) -> Dict[float, SegmentLengthResult]:
        results: Dict[float, SegmentLengthResult] = {}
        for seg_s in self.cfg.segment_lengths_s:
            log.info("=== Evaluating with segment_s=%.1f ===", seg_s)
            results[seg_s] = self._evaluate_one_segment_length(
                segment_s=seg_s, overlap_s=self.cfg.overlap_s
            )
        # Save full results.
        with open(self.output_dir / "eval_results.json", "w") as fh:
            json.dump(
                {
                    "config": asdict(self.cfg),
                    "checkpoint_config": self._ckpt_config,
                    "results": {
                        str(s): {
                            "segment_s": r.segment_s,
                            "overlap_s": r.overlap_s,
                            "frame_macro_f1": r.frame_macro_f1,
                            "event_macro_f1": r.event_macro_f1,
                            "per_class": [asdict(c) for c in r.per_class],
                        }
                        for s, r in results.items()
                    },
                },
                fh,
                indent=2,
            )
        self._print_summary(results)
        return results

    # ------------------------------------------------------ inner steps

    def _evaluate_one_segment_length(
        self, *, segment_s: float, overlap_s: float
    ) -> SegmentLengthResult:
        # 1. Tile recordings.
        segments = build_eval_segments(
            self.val_audio,
            segment_s=segment_s,
            overlap_s=overlap_s,
        )
        # 2. Group segments by source recording so we can stitch them back
        # together with overlap averaging per Section 5.1.
        seg_index_by_file: Dict[Path, List[int]] = {}
        for i, seg in enumerate(segments):
            seg_index_by_file.setdefault(seg.audio_path, []).append(i)

        # 3. Forward all segments to get per-segment frame probabilities.
        dataset = ATBFLDataset(
            segments=segments,
            num_classes=self.num_classes,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            class_names=self.train_class_names,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_segments,
            persistent_workers=False,
        )

        # Storage: per-segment frame probabilities + valid frame lengths.
        per_seg_probs: List[Tensor] = [None] * len(segments)  # type: ignore[list-item]
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                progress(
                    loader,
                    desc=f"infer seg={segment_s:.0f}s",
                    total=len(loader),
                    unit="batch",
                    leave=False,
                )
            ):
                audio = batch["audio"].to(self.device)
                frame_lengths = batch["frame_length"].to(self.device)
                features, _ = self.transform(audio, lengths=frame_lengths)
                logits, _, _ = self.classifier(features, lab_lengths=frame_lengths)
                probs = torch.sigmoid(logits).cpu()
                for b in range(probs.size(0)):
                    seg_i = batch_idx * self.cfg.batch_size + b
                    if seg_i >= len(segments):
                        break
                    T = int(frame_lengths[b].item())
                    p = probs[b, :T]
                    if self.collapse_eval_to_three:
                        p = collapse_seven_to_three(p)
                    per_seg_probs[seg_i] = p

        # 4. Stitch per-recording with overlap averaging.
        per_file_probs: Dict[Path, Tensor] = {}
        for af in self.val_audio:
            seg_idx = seg_index_by_file.get(af.path, [])
            if not seg_idx:
                continue
            wins = [per_seg_probs[i] for i in seg_idx]
            starts = [segments[i].start_s for i in seg_idx]
            stitched = merge_overlapping_windows(
                wins,
                starts,
                hop_s=self.hop_s,
                total_duration_s=af.duration_s,
            )
            per_file_probs[af.path] = stitched

        # 5. Build per-file frame-level targets at the same resolution.
        per_file_targets: Dict[Path, Tensor] = {}
        num_eval_classes = len(self.eval_class_names)
        for af in self.val_audio:
            if af.path not in per_file_probs:
                continue
            num_frames = per_file_probs[af.path].size(0)
            tgt = torch.zeros(
                (num_frames, num_eval_classes), dtype=torch.float32
            )
            for ann in af.annotations:
                # Use 3-class label for eval, regardless of training mode.
                cls = map_label_to_class(ann.label, 3)
                if cls is None or cls not in self.eval_class_names:
                    continue
                c = self.eval_class_names.index(cls)
                # Floor-based assignment, consistent with build_frame_labels.
                start_frame = max(0, int(math.floor(ann.onset_s / self.hop_s)))
                end_frame = min(num_frames, int(math.floor(ann.offset_s / self.hop_s)))
                if end_frame > start_frame:
                    tgt[start_frame:end_frame, c] = 1.0
            per_file_targets[af.path] = tgt

        # 6. Concatenate across recordings and find optimal thresholds.
        all_probs = torch.cat([per_file_probs[p] for p in per_file_probs], dim=0)
        all_targets = torch.cat([per_file_targets[p] for p in per_file_probs], dim=0)
        thresholds, frame_metrics = find_optimal_thresholds(
            all_probs, all_targets, grid=self.cfg.threshold_search_grid
        )

        # 7. Postprocess each recording at the chosen thresholds and run
        # event-level matching.
        event_counts = {
            name: {"tp": 0, "fp": 0, "fn": 0} for name in self.eval_class_names
        }
        for af in self.val_audio:
            if af.path not in per_file_probs:
                continue
            detected = probabilities_to_calls(
                per_file_probs[af.path],
                thresholds,
                hop_s=self.hop_s,
                class_names=self.eval_class_names,
                median_kernel_ms=self.cfg.median_kernel_ms,
                merge_gap_s=self.cfg.merge_gap_s,
                min_duration_s=self.cfg.min_call_duration_s,
                max_duration_s=self.cfg.max_call_duration_s,
            )
            truth = _ground_truth_events_for_file(
                af,
                num_classes=self.num_classes,
                eval_class_names=self.eval_class_names,
            )
            for name in self.eval_class_names:
                tp, fp, fn = event_level_metrics(
                    detected,
                    truth,
                    label=name,
                    iou_threshold=self.cfg.event_iou_threshold,
                )
                event_counts[name]["tp"] += tp
                event_counts[name]["fp"] += fp
                event_counts[name]["fn"] += fn

        # 8. Pack per-class results.
        result = SegmentLengthResult(segment_s=segment_s, overlap_s=overlap_s)
        for c, name in enumerate(self.eval_class_names):
            fm = frame_metrics[c]
            ev = event_counts[name]
            event_precision = (
                ev["tp"] / (ev["tp"] + ev["fp"])
                if (ev["tp"] + ev["fp"]) > 0
                else 0.0
            )
            event_recall = (
                ev["tp"] / (ev["tp"] + ev["fn"])
                if (ev["tp"] + ev["fn"]) > 0
                else 0.0
            )
            event_f1 = (
                2 * event_precision * event_recall / (event_precision + event_recall)
                if (event_precision + event_recall) > 0
                else 0.0
            )
            result.per_class.append(
                PerClassResult(
                    label=name,
                    threshold=float(thresholds[c].item()),
                    frame_tp=fm.tp,
                    frame_fp=fm.fp,
                    frame_fn=fm.fn,
                    frame_precision=fm.precision,
                    frame_recall=fm.recall,
                    frame_f1=fm.f1,
                    event_tp=ev["tp"],
                    event_fp=ev["fp"],
                    event_fn=ev["fn"],
                    event_precision=event_precision,
                    event_recall=event_recall,
                    event_f1=event_f1,
                )
            )
        return result

    # ----------------------------------------------------------- I/O

    def _print_summary(self, results: Dict[float, SegmentLengthResult]) -> None:
        log.info("=== Eval summary ===")
        log.info(
            "%-8s %-10s %-10s   %-10s %-10s",
            "seg_s",
            "frame_F1",
            "event_F1",
            "best_class",
            "worst_class",
        )
        for seg_s, r in results.items():
            best = max(r.per_class, key=lambda c: c.event_f1)
            worst = min(r.per_class, key=lambda c: c.event_f1)
            log.info(
                "%-8.1f %-10.4f %-10.4f   %-10s %-10s",
                seg_s,
                r.frame_macro_f1,
                r.event_macro_f1,
                f"{best.label}={best.event_f1:.3f}",
                f"{worst.label}={worst.event_f1:.3f}",
            )
