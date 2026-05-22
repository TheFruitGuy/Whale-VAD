"""WhaleVAD-BPN model (Geldenhuys et al., arXiv:2510.21280v2, Oct 2025).

This module implements the *upgraded* WhaleVAD baseline described in
Section V.A of the follow-up BPN paper.  Two concrete differences from
the DCASE 2025 paper's WhaleVAD model live here:

1.  The three-layer depthwise convolutional aggregation block now uses
    increasing dilation :math:`d \\in \\{2, 4, 8\\}` along the time axis,
    so the temporal receptive field expands without further pooling.
2.  Residual connections are added *between* each depthwise layer
    (the original paper only put a single residual around the whole
    block), and spatial dropout (``Dropout2d``) is used throughout the
    block in place of conventional dropout.

The optional :class:`BoundaryProposalNetwork` is the BPN gating
mechanism from Section V.B — multiple intermediate feature maps are
projected, concatenated, processed by a small proposal network, and
combined by a learned weighted mean into a per-frame multiplicative
mask that suppresses spurious detections.

The training recipe explicitly reported in Section V.B.5 is encoded
in :class:`whalevad.bpn.config.BPNTrainingConfig`; see that file for
the values (``lr=1e-3``, ``weight_decay=0.01``, batch=48, ~32 epochs).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Dropout,
    Dropout2d,
    GELU,
    LSTM,
    Linear,
    MaxPool2d,
    Module,
    ModuleList,
    Parameter,
    Sequential,
)

from ..model import ResidualBlock, WhaleVADClassifier


__all__ = [
    "DilatedDepthwiseLayer",
    "make_upgraded_depthwise_block",
    "WhaleVADBPN",
    "BoundaryProposalNetwork",
    "WhaleVADBPNWrapper",
]


# ---------------------------------------------------------------- helpers


class DilatedDepthwiseLayer(Module):
    """One depthwise 3×3 conv with time-axis dilation + spatial dropout + BN + GELU.

    ``Dropout2d`` is applied *before* the convolution so that whole
    channels are zeroed during training (consistent with the paper's
    description of spatial dropout in the aggregation block).
    Padding is set so the spatial dims are preserved for every choice
    of ``dilation_t``.
    """

    def __init__(
        self,
        channels: int = 128,
        *,
        dilation_t: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = Dropout2d(dropout)
        self.conv = Conv2d(
            channels,
            channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            # padding=(1, dilation_t) keeps both freq and time dims fixed.
            padding=(1, dilation_t),
            dilation=(1, dilation_t),
            groups=channels,  # depthwise
        )
        self.bn = BatchNorm2d(channels)
        self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(self.dropout(x))))


def make_upgraded_depthwise_block(
    channels: int = 128,
    *,
    dilations: Sequence[int] = (2, 4, 8),
    dropout: float = 0.2,
) -> Module:
    """Build the BPN paper's upgraded depthwise aggregation block.

    Three dilated-depthwise layers wrapped in a :class:`ResidualBlock`
    so that each layer's output is residually summed onto the running
    activations, as described in Section V.A.
    """
    return ResidualBlock(
        *[
            DilatedDepthwiseLayer(channels, dilation_t=d, dropout=dropout)
            for d in dilations
        ]
    )


# ---------------------------------------------------------- WhaleVAD-BPN


class WhaleVADBPN(WhaleVADClassifier):
    """WhaleVAD classifier with the BPN paper's depthwise block.

    Inherits everything from :class:`WhaleVADClassifier` and only
    replaces the third stage of ``cnn_blocks`` (the residual
    bottleneck + aggregation pair) with one that uses
    :func:`make_upgraded_depthwise_block` for the aggregation half.

    Extra constructor arguments
    ---------------------------
    depthwise_dilations:
        Dilation factors for the three depthwise layers along the time
        axis.  Defaults to ``(2, 4, 8)`` per the paper.
    depthwise_dropout:
        Spatial-dropout rate for ``DilatedDepthwiseLayer``.  The paper
        does not specify a value; ``0.2`` matches the original
        WhaleVAD's ``Dropout2d`` at the aggregation entry.
    """

    def __init__(
        self,
        *,
        depthwise_dilations: Sequence[int] = (2, 4, 8),
        depthwise_dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._upgrade_aggregation_block(
            dilations=depthwise_dilations, dropout=depthwise_dropout
        )

    # -------------------------------------------------------- private

    def _upgrade_aggregation_block(
        self, *, dilations: Sequence[int], dropout: float
    ) -> None:
        """Swap the parent's aggregation block for the dilated/residual one.

        ``cnn_blocks`` layout (see ``model.py``)::

            Sequential(
                conv_block_feat_extractor,           # idx 0
                ResidualBlock(
                    [conv_block_bottleneck,          # block 0
                     conv_block_feat_agg]            # block 1  <- replace
                ),                                   # idx 1
            )

        We only patch the second block of the inner ``ResidualBlock``.
        """
        outer = self.cnn_blocks
        inner = outer[1]  # ResidualBlock
        new_aggregation = make_upgraded_depthwise_block(
            channels=128, dilations=dilations, dropout=dropout
        )
        # Preserve any other blocks (e.g. bottleneck) before the
        # aggregation, so users can still disable the bottleneck via the
        # parent's flag.
        old_blocks = list(inner.blocks)
        if not old_blocks:
            old_blocks = [new_aggregation]
        else:
            old_blocks[-1] = new_aggregation
        # Re-wrap with the same residual semantics as the parent.
        new_inner = ResidualBlock(
            *old_blocks,
            output_residuals=inner.output_residuals,
        )
        # Reconstruct the Sequential so the new ResidualBlock is in place.
        new_outer = Sequential(outer[0], new_inner)
        self.cnn_blocks = new_outer


# ---------------------------------------------------- Boundary Proposal Net


class _IntermediateProjectionHead(Module):
    """1×1 conv → BN → GELU → maxpool, per Table III of the paper."""

    def __init__(self, in_channels: int, out_channels: int = 128) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1)
        )
        self.bn = BatchNorm2d(out_channels)
        self.act = GELU()
        self.pool = MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class BoundaryProposalNetwork(Module):
    """The BPN proposal head + BiLSTM gating (Section V.B, Table III).

    Inputs
    ------
    intermediates:
        List of :math:`H` intermediate feature maps from the classifier,
        each with shape ``(B, C_in_h, F_h, T)``.  All must share the
        time dimension ``T``.

    Output
    ------
    mask: (B, T, num_classes)
        Sigmoid-valued gating mask, broadcast-multiplied with the
        classifier's per-frame probabilities to suppress false
        positives.

    Architecture (BPN-multi variant, which the paper found best)
    -----------
    For each intermediate:
        - 1×1 Conv → BN → GELU → MaxPool(3×1)  (=> "projection head")
    Stack the heads along a new H dimension and average-pool away
    frequency so each head reduces to a ``(B, H, C, T)`` map.  The
    proposal network then runs two transpose convs that re-arrange
    those into ``R`` candidate ROI vectors per time step
    (the transpose-conv-with-stride-1 expansion described in
    Table III), each of dimension ``C_bpn``.  Every ROI is processed
    by a shared BiLSTM, projected to ``num_classes``, and the
    R-axis is collapsed by a *learned* weighted mean to obtain the
    final mask.
    """

    def __init__(
        self,
        intermediate_channels: Sequence[int],
        *,
        proj_channels: int = 128,
        bpn_channels: int = 64,
        num_rois: int = 4,
        lstm_hidden: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not intermediate_channels:
            raise ValueError("BPN needs at least one intermediate feature map")
        self.num_heads = len(intermediate_channels)
        self.num_rois = int(num_rois)
        self.num_classes = int(num_classes)

        self.heads = ModuleList(
            [
                _IntermediateProjectionHead(c, out_channels=proj_channels)
                for c in intermediate_channels
            ]
        )

        # Proposal network: two transpose convs.  K=(4,1) and K=(5,1)
        # with stride 1 expand the head-axis from H to R distinct
        # ROIs per time step.  We treat the H dimension as the
        # "spatial" axis the transpose convs operate on.
        self.proposal = Sequential(
            ConvTranspose2d(
                proj_channels, proj_channels,
                kernel_size=(4, 1), stride=(1, 1),
            ),
            BatchNorm2d(proj_channels),
            GELU(),
            Dropout2d(dropout),
            ConvTranspose2d(
                proj_channels, bpn_channels,
                kernel_size=(5, 1), stride=(1, 1),
            ),
            BatchNorm2d(bpn_channels),
            GELU(),
            Dropout2d(dropout),
        )
        # Output of the transpose convs along H is ``H + 3 + 4 = H + 7``;
        # we adaptively pool to ``num_rois`` so the rest of the BPN
        # works at a fixed shape.

        self.roi_lstm = LSTM(
            input_size=bpn_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.roi_classifier = Linear(2 * lstm_hidden, num_classes)

        # Learned per-ROI weights for the weighted mean across ROIs.
        # Initialise uniform so the BPN starts as a simple average.
        self.roi_weights = Parameter(torch.ones(self.num_rois) / self.num_rois)

    def forward(self, intermediates: Sequence[Tensor]) -> Tensor:
        """Compute the per-frame gating mask.

        Parameters
        ----------
        intermediates
            One feature map per projection head, each shaped
            ``(B, C_h, F_h, T)``.  All must share the same ``B`` and
            ``T``.
        """
        if len(intermediates) != self.num_heads:
            raise ValueError(
                f"BPN expects {self.num_heads} intermediates, got {len(intermediates)}"
            )
        proj: List[Tensor] = []
        T_ref: Optional[int] = None
        for x, head in zip(intermediates, self.heads):
            y = head(x)                       # (B, P, F', T)
            # Collapse frequency by averaging — gives one vector per
            # head per time step.
            y = y.mean(dim=2, keepdim=False)  # (B, P, T)
            if T_ref is None:
                T_ref = y.size(-1)
            elif y.size(-1) != T_ref:
                # Defensive: crop to the smallest T across heads.
                y = y[..., :T_ref]
            proj.append(y)
        # Stack heads along a new "H" axis: (B, P, H, T).
        stacked = torch.stack(proj, dim=2)
        # Treat H as the spatial axis the transpose convs operate on.
        # Permute to (B, P, H, T) then run ConvTranspose2d which expects
        # NCHW; here C=P, H=H, W=T.  The convs only act on H (kernel
        # (K, 1)) so T is untouched.
        roi = self.proposal(stacked)          # (B, C_bpn, H + 7, T)

        # Reduce H + 7 -> num_rois via adaptive avg pool over H.
        roi = roi.mean(dim=2) if self.num_rois == 1 else self._reduce_roi(roi)
        # roi shape: (B, num_rois, C_bpn, T)
        # For num_rois == 1 we squeezed out the H axis; re-add it.
        if self.num_rois == 1:
            roi = roi.unsqueeze(1)

        # Per-ROI BiLSTM over time.  Flatten ROI into batch.
        B, R, C_bpn, T = roi.shape
        roi_seq = roi.permute(0, 1, 3, 2).reshape(B * R, T, C_bpn)
        seq_out, _ = self.roi_lstm(roi_seq)   # (B*R, T, 2*H)
        roi_logits = self.roi_classifier(seq_out)         # (B*R, T, num_classes)
        roi_logits = roi_logits.view(B, R, T, self.num_classes)

        # Sigmoid then learned weighted mean across ROIs.
        roi_probs = torch.sigmoid(roi_logits)             # (B, R, T, C)
        weights = torch.softmax(self.roi_weights, dim=0).view(1, R, 1, 1)
        mask = (roi_probs * weights).sum(dim=1)           # (B, T, C)
        return mask

    @staticmethod
    def _reduce_roi(roi: Tensor) -> Tensor:
        """Adaptive-average-pool the H axis to fixed ``num_rois`` slots.

        ``roi`` has shape ``(B, C, H_eff, T)``; the BPN expects a fixed
        number of ROIs regardless of how many intermediate heads the
        user supplied, so we collapse H_eff via an adaptive pool.
        """
        # Move T to be the batch component of an adaptive pool, then back.
        B, C, H_eff, T = roi.shape
        # adaptive_avg_pool2d treats trailing two dims as spatial.
        return torch.nn.functional.adaptive_avg_pool2d(
            roi, output_size=(roi.size(0) * 0 + 4, T)
        ).reshape(B, C, 4, T).permute(0, 2, 1, 3)


# ----------------------------------------------------- the final wrapper


class WhaleVADBPNWrapper(Module):
    """End-to-end model = upgraded backbone + BPN gating.

    The wrapper exposes the same forward signature as
    :class:`WhaleVADClassifier` (``features, lab_lengths``) but the
    returned probabilities are *gated* by the BPN mask.  When
    ``include_bpn=False`` the wrapper degenerates to the plain
    upgraded backbone, so the same model class can be used to train
    and ablate.

    The classifier must be constructed with
    ``include_intermediate_features=True`` so the residual block
    exposes the intermediate maps needed by the BPN.
    """

    def __init__(
        self,
        classifier: WhaleVADClassifier,
        *,
        bpn: Optional[BoundaryProposalNetwork] = None,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.bpn = bpn

    @property
    def include_bpn(self) -> bool:
        return self.bpn is not None

    def forward(
        self,
        features: Tensor,
        lab_lengths: Optional[Tensor] = None,
        **opts,
    ) -> Tuple[Tensor, Tensor, dict]:
        logits, probs, ret = self.classifier(
            features, lab_lengths=lab_lengths, **opts
        )
        if self.bpn is None:
            return logits, probs, ret
        intermediates = ret.get("intermediate_features") or []
        if not intermediates:
            # Fallback: caller forgot to enable intermediate features.
            return logits, probs, ret
        mask = self.bpn(intermediates)
        # Gate the probabilities, then re-derive logits so downstream
        # losses based on logits remain consistent.
        gated_probs = probs * mask
        gated_probs = gated_probs.clamp(min=1e-7, max=1.0 - 1e-7)
        gated_logits = torch.log(gated_probs / (1.0 - gated_probs))
        ret = dict(ret)
        ret["bpn_mask"] = mask
        return gated_logits, gated_probs, ret
