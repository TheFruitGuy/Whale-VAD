"""Evaluation metrics and threshold optimisation for Whale-VAD.

The paper selects a per-class threshold :math:`\\theta_c` that maximises
the development F1 score (Section 5.8).  This module implements both
the per-class F1 sweep and frame-level TP/FP/FN accounting used
throughout training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch import Tensor


@dataclass
class FrameMetrics:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_frame_metrics(
    probs: Tensor,  # (..., C)
    targets: Tensor,  # (..., C)
    thresholds: Tensor,  # (C,)
    *,
    mask: Tensor | None = None,  # (..., ) bool
) -> List[FrameMetrics]:
    """Compute per-class TP/FP/FN at the supplied thresholds."""
    num_classes = probs.size(-1)
    if thresholds.numel() != num_classes:
        raise ValueError("thresholds shape mismatch")
    pred = (probs >= thresholds.to(probs)).to(torch.bool)
    tgt = targets.to(torch.bool)
    if mask is not None:
        m = mask.unsqueeze(-1)
        pred = pred & m
        tgt = tgt & m
    tp = (pred & tgt).reshape(-1, num_classes).sum(dim=0)
    fp = (pred & ~tgt).reshape(-1, num_classes).sum(dim=0)
    fn = (~pred & tgt).reshape(-1, num_classes).sum(dim=0)
    if mask is not None:
        # ensure padded frames don't contribute to fn either
        valid = mask.reshape(-1).sum().item()
        del valid
    return [
        FrameMetrics(int(tp[c].item()), int(fp[c].item()), int(fn[c].item()))
        for c in range(num_classes)
    ]


def find_optimal_thresholds(
    probs: Tensor,  # (N, C) flattened over valid frames
    targets: Tensor,  # (N, C)
    *,
    grid: int = 101,
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
) -> Tuple[Tensor, List[FrameMetrics]]:
    """Grid-search per-class thresholds that maximise frame-level F1.

    Returns the optimal thresholds (one per class) and the
    :class:`FrameMetrics` at those thresholds.
    """
    if probs.dim() != 2 or targets.dim() != 2 or probs.shape != targets.shape:
        raise ValueError("probs and targets must be (N, C) tensors of equal shape")
    num_classes = probs.size(-1)
    thresholds = torch.linspace(min_threshold, max_threshold, grid)
    best_th = torch.full((num_classes,), 0.5)
    best_metrics: List[FrameMetrics] = [FrameMetrics(0, 0, 0) for _ in range(num_classes)]

    tgt_bool = targets.to(torch.bool)
    for c in range(num_classes):
        p = probs[:, c]
        t = tgt_bool[:, c]
        best_f1 = -1.0
        for th in thresholds.tolist():
            pred = p >= th
            tp = int((pred & t).sum().item())
            fp = int((pred & ~t).sum().item())
            fn = int((~pred & t).sum().item())
            denom_p = tp + fp
            denom_r = tp + fn
            precision = tp / denom_p if denom_p > 0 else 0.0
            recall = tp / denom_r if denom_r > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_th[c] = th
                best_metrics[c] = FrameMetrics(tp, fp, fn)
    return best_th, best_metrics


def macro_f1(metrics: Sequence[FrameMetrics]) -> float:
    if not metrics:
        return 0.0
    return float(sum(m.f1 for m in metrics) / len(metrics))


def gather_valid_frames(
    probs: Tensor, targets: Tensor, frame_lengths: Tensor
) -> Tuple[Tensor, Tensor]:
    """Flatten ``(B, T, C)`` predictions, dropping padded frames."""
    B, T, C = probs.shape
    idx = torch.arange(T, device=frame_lengths.device).unsqueeze(0)
    mask = idx < frame_lengths.unsqueeze(1)  # (B, T)
    return probs[mask].view(-1, C), targets[mask].view(-1, C)
