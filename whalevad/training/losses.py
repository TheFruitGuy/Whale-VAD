"""Loss functions for Whale-VAD training (Section 5.6, 5.7).

This module implements:

* ``WeightedBCELoss`` — weighted binary cross-entropy with the class
  weighting :math:`w_c = N / P_c` described in Section 5.6.
* ``FocalLoss`` — sigmoid focal loss with ``alpha=0.25`` and
  ``gamma=2`` (the recommendation from Lin et al., 2018, used here per
  the paper).
* ``MultiObjectiveLoss`` — joint classification + bounding-box L1
  regression loss for the multi-objective regression experiment
  described in Section 5.7.

All classification losses operate on per-frame logits and natively
respect padding via ``frame_lengths``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


# ----------------------------------------------------------------- helpers


def _frame_mask(frame_lengths: Tensor, max_frames: int) -> Tensor:
    """Build a (batch, frames) boolean mask of valid (non-padding) frames."""
    device = frame_lengths.device
    idx = torch.arange(max_frames, device=device).unsqueeze(0)  # (1, T)
    return idx < frame_lengths.unsqueeze(1)


def compute_bce_pos_weight(
    pos_counts: Tensor, neg_segment_count: int, *, eps: float = 1.0
) -> Tensor:
    """Per-class positive weight :math:`w_c = N / P_c` (Section 5.6).

    Args:
        pos_counts: shape ``(num_classes,)`` — number of positive
            segments per class.
        neg_segment_count: scalar ``N`` — number of negative segments.
        eps: lower bound on ``P_c`` to avoid division by zero.
    """
    pos = pos_counts.clamp_min(eps).float()
    return float(neg_segment_count) / pos


# ----------------------------------------------------------- weighted BCE


class WeightedBCELoss(Module):
    """Frame-level weighted BCE with optional padding mask.

    Args:
        pos_weight: tensor of shape ``(num_classes,)`` with per-class
            positive sample weights (typically :math:`w_c = N / P_c`).
        reduction: reduction over valid frames — ``"mean"`` averages
            over (valid_frame * class) elements.
    """

    def __init__(
        self,
        pos_weight: Optional[Tensor] = None,
        *,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None  # type: ignore[assignment]
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,  # (B, T, C)
        targets: Tensor,  # (B, T, C)
        frame_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        if frame_lengths is not None:
            mask = _frame_mask(frame_lengths, logits.size(1)).unsqueeze(-1)
            loss = loss * mask
            denom = mask.sum().clamp_min(1).float() * loss.size(-1)
        else:
            denom = torch.tensor(
                float(loss.numel()), device=loss.device, dtype=loss.dtype
            )
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.sum() / denom


# -------------------------------------------------------------- focal loss


class FocalLoss(Module):
    """Sigmoid focal loss (Lin et al., 2018).

    Uses ``alpha=0.25`` and ``gamma=2`` by default, matching Section 5.6.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,  # (B, T, C)
        targets: Tensor,  # (B, T, C)
        frame_lengths: Optional[Tensor] = None,
    ) -> Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * ce
        if frame_lengths is not None:
            mask = _frame_mask(frame_lengths, logits.size(1)).unsqueeze(-1)
            loss = loss * mask
            denom = mask.sum().clamp_min(1).float() * loss.size(-1)
        else:
            denom = torch.tensor(
                float(loss.numel()), device=loss.device, dtype=loss.dtype
            )
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.sum() / denom


# ------------------------------------------- multi-objective (cls + bbox)


class MultiObjectiveLoss(Module):
    """Combines a classification loss with bounding-box regression.

    The bounding-box branch (Section 5.7) is optional.  Anchors that do
    not correspond to ground-truth boxes are supervised through the
    confidence head with a BCE term, while present anchors are matched
    to the available annotations and supervised with smoothed L1 loss
    on box coordinates plus BCE on the confidence.
    """

    def __init__(
        self,
        classifier_loss: Module,
        *,
        regression_weight: float = 1.0,
        confidence_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.classifier_loss = classifier_loss
        self.regression_weight = float(regression_weight)
        self.confidence_weight = float(confidence_weight)
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(
        self,
        *,
        logits: Tensor,
        targets: Tensor,
        frame_lengths: Tensor,
        bbox_pred: Optional[Tensor] = None,
        bbox_conf: Optional[Tensor] = None,
        bbox_target: Optional[Tensor] = None,
        bbox_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        cls_loss = self.classifier_loss(logits, targets, frame_lengths)
        total = cls_loss
        out: Dict[str, Tensor] = {"cls_loss": cls_loss.detach()}

        if (
            bbox_pred is not None
            and bbox_conf is not None
            and bbox_target is not None
            and bbox_mask is not None
        ):
            # Confidence: BCE between predicted anchor confidence and the
            # presence mask of the ground-truth boxes.
            conf_target = bbox_mask.float()
            conf_loss = F.binary_cross_entropy_with_logits(
                bbox_conf, conf_target, reduction="mean"
            )

            # Regression: only on anchors that have a ground-truth box.
            if bbox_mask.any():
                m = bbox_mask
                reg_loss = F.smooth_l1_loss(
                    bbox_pred[m], bbox_target[m], beta=self.smooth_l1_beta
                )
            else:
                reg_loss = bbox_pred.new_zeros(())

            total = (
                cls_loss
                + self.regression_weight * reg_loss
                + self.confidence_weight * conf_loss
            )
            out["bbox_reg_loss"] = reg_loss.detach()
            out["bbox_conf_loss"] = conf_loss.detach()
        out["loss"] = total
        return out
