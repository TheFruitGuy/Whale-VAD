"""Postprocessing additions for the BPN paper reproduction.

The DCASE paper used a single per-class threshold and a single
500 ms median-filter kernel; the BPN paper (Section II and Table IV)
generalises this to a richer set of post-processing knobs and shows
that careful tuning of these alone produces large F1 gains:

* **Class-specific median filter** (kernel size in frames).
* **Hysteresis thresholding** with separate on/off thresholds — the
  detector enters an "active" state when the smoothed probability
  exceeds the on-threshold, and only exits when it drops below the
  off-threshold.
* **Hangover** — an M-out-of-N majority vote over the binary
  detection sequence, equivalent to a median filter applied to the
  binary stream.
* **Class-specific event-level filters** (min/max event duration,
  minimum inter-event gap).

This module implements those operations and exposes a single
:func:`probabilities_to_calls_v2` entry point that runs the full
pipeline.  The existing :func:`probabilities_to_calls` in
``whalevad.training.postprocessing`` is unchanged so the DCASE
pipeline continues to behave the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ..training.postprocessing import (
    CallEvent,
    median_filter_1d,
)


__all__ = [
    "PostProcessV2Config",
    "hysteresis_threshold",
    "hangover_filter",
    "probabilities_to_calls_v2",
]


# ---------------------------------------------------------------- types


_Listable = Union[float, int, Sequence[float], Sequence[int], None]


def _per_class(value: _Listable, num_classes: int, default):
    """Expand a single value or short list to one entry per class."""
    if value is None:
        return [default] * num_classes
    if isinstance(value, (int, float)):
        return [float(value)] * num_classes
    out = list(value)
    if len(out) == 1:
        return out * num_classes
    if len(out) != num_classes:
        raise ValueError(
            f"Expected length {num_classes} or 1, got {len(out)}: {value!r}"
        )
    return out


@dataclass
class PostProcessV2Config:
    """All post-processing knobs as class-specific values."""

    on_thresholds: Sequence[float] = (0.5, 0.5, 0.5)
    off_thresholds: Optional[Sequence[float]] = None  # None -> same as on
    median_kernels: Optional[Sequence[Optional[int]]] = None  # frames
    hangover_kernels: Optional[Sequence[Optional[int]]] = None  # frames
    min_event_durations_s: Sequence[float] = (0.5, 0.5, 0.5)
    max_event_durations_s: Sequence[float] = (30.0, 30.0, 30.0)
    min_inter_event_s: Sequence[float] = (0.5, 0.5, 0.5)


# ------------------------------------------------------- filter helpers


def _per_class_median_filter(
    probs: Tensor, kernels: Sequence[Optional[int]]
) -> Tensor:
    """Apply a separate median-filter kernel to each class channel.

    ``probs`` is ``(T, C)``.  A ``None`` (or non-positive) kernel
    leaves the channel unchanged.
    """
    if probs.dim() != 2:
        raise ValueError("expected (T, C) probabilities")
    T, C = probs.shape
    if len(kernels) != C:
        raise ValueError(f"need {C} kernels, got {len(kernels)}")
    out_channels: List[Tensor] = []
    for c, k in enumerate(kernels):
        col = probs[:, c : c + 1]  # (T, 1)
        if k is None or k <= 1:
            out_channels.append(col)
            continue
        smoothed = median_filter_1d(col.unsqueeze(0), kernel_size=int(k))
        out_channels.append(smoothed.squeeze(0))
    return torch.cat(out_channels, dim=-1)


def hysteresis_threshold(
    probs: Tensor, on_t: float, off_t: float
) -> Tensor:
    """Hysteresis thresholding on a 1-D probability sequence.

    The detector enters the "active" state when ``probs[i] >= on_t``
    and remains active until ``probs[i] < off_t``.  Returns a bool
    tensor of the same length.

    With ``on_t == off_t`` this collapses to a plain threshold.
    """
    if probs.dim() != 1:
        raise ValueError("expected 1-D probabilities")
    if on_t < off_t:
        # Defensive: paper specifies on >= off.  Swap to recover sane
        # behaviour rather than producing a no-op.
        on_t, off_t = off_t, on_t
    n = probs.size(0)
    out = torch.zeros(n, dtype=torch.bool, device=probs.device)
    active = False
    vals = probs.tolist()
    for i, v in enumerate(vals):
        if active:
            if v < off_t:
                active = False
        else:
            if v >= on_t:
                active = True
        out[i] = active
    return out


def hangover_filter(binary: Tensor, kernel: int) -> Tensor:
    """M-out-of-N majority vote on a 1-D boolean detection sequence.

    A length-``kernel`` sliding window is centred on each frame and
    the frame is set to active iff at least ``ceil(kernel/2)`` of the
    window samples are active.  This is the median filter of the
    binary stream, as per equation (1) in the BPN paper.
    """
    if binary.dim() != 1:
        raise ValueError("expected 1-D detection sequence")
    if kernel is None or kernel <= 1:
        return binary
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    x = binary.float().unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    x_p = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    windows = x_p.unfold(-1, kernel, 1).squeeze(0).squeeze(0)  # (T, K)
    votes = windows.sum(dim=-1)
    return votes > (kernel / 2.0)


# --------------------------------------------------------- event helpers


def _binary_to_events(
    binary: Tensor, *, hop_s: float, label: str
) -> List[CallEvent]:
    events: List[CallEvent] = []
    n = binary.size(0)
    on = False
    start = 0
    bools = binary.tolist()
    for i, v in enumerate(bools):
        if v and not on:
            on = True
            start = i
        elif not v and on:
            on = False
            events.append(
                CallEvent(start * hop_s, i * hop_s, label=label)
            )
    if on:
        events.append(CallEvent(start * hop_s, n * hop_s, label=label))
    return events


def _merge_with_min_gap(
    events: Sequence[CallEvent], *, min_gap_s: float
) -> List[CallEvent]:
    if not events:
        return []
    out: List[CallEvent] = []
    cur = events[0]
    for ev in events[1:]:
        if ev.onset_s - cur.offset_s <= min_gap_s:
            cur = CallEvent(
                onset_s=cur.onset_s,
                offset_s=max(cur.offset_s, ev.offset_s),
                label=cur.label,
                score=max(cur.score, ev.score),
            )
        else:
            out.append(cur)
            cur = ev
    out.append(cur)
    return out


def probabilities_to_calls_v2(
    probs: Tensor,
    *,
    hop_s: float,
    class_names: Sequence[str],
    config: PostProcessV2Config,
) -> List[CallEvent]:
    """Full BPN-paper postprocessing pipeline.

    Steps applied in order, all class-specific:

    1. Median filter the probabilities for each class with its own
       kernel (Section II.C.1).
    2. Hysteresis-threshold to binary detections (Section II.C.1).
    3. Hangover (majority-vote) on the binary detections (Section II.C.1
       and equation 1).
    4. Convert to call events (one per contiguous run of True).
    5. Merge events of the same class with inter-event gap below the
       per-class minimum (Section II.C.2).
    6. Drop events shorter than the per-class min or longer than the
       per-class max duration (Section II.C.2).
    """
    if probs.dim() != 2:
        raise ValueError("expected (T, C) probabilities")
    C = probs.size(-1)
    if len(class_names) != C:
        raise ValueError(
            f"class_names size {len(class_names)} != probs class dim {C}"
        )

    median_kernels = _per_class(config.median_kernels, C, default=None)
    on_thresholds = _per_class(config.on_thresholds, C, default=0.5)
    off_thresholds = _per_class(
        config.off_thresholds if config.off_thresholds is not None
        else config.on_thresholds,
        C,
        default=0.5,
    )
    hangover_kernels = _per_class(config.hangover_kernels, C, default=None)
    min_durs = _per_class(config.min_event_durations_s, C, default=0.5)
    max_durs = _per_class(config.max_event_durations_s, C, default=30.0)
    min_gaps = _per_class(config.min_inter_event_s, C, default=0.5)

    # 1. Class-specific median filter
    smoothed = _per_class_median_filter(probs, median_kernels)

    # 2-4. Per class: hysteresis -> hangover -> events
    all_events: List[CallEvent] = []
    for c, name in enumerate(class_names):
        col = smoothed[:, c]
        binary = hysteresis_threshold(
            col, on_t=float(on_thresholds[c]), off_t=float(off_thresholds[c])
        )
        hk = hangover_kernels[c]
        if hk is not None and hk > 1:
            binary = hangover_filter(binary, kernel=int(hk))
        events = _binary_to_events(binary, hop_s=hop_s, label=name)
        # Attach mean smoothed probability as the event score.
        scored: List[CallEvent] = []
        for ev in events:
            i0 = max(0, int(ev.onset_s / hop_s))
            i1 = min(col.size(0), int(ev.offset_s / hop_s))
            score = float(col[i0:i1].mean().item()) if i1 > i0 else 0.0
            scored.append(
                CallEvent(ev.onset_s, ev.offset_s, ev.label, score=score)
            )
        # 5. Class-specific inter-event merge
        merged = _merge_with_min_gap(scored, min_gap_s=float(min_gaps[c]))
        # 6. Class-specific min/max duration filter
        kept = [
            ev for ev in merged
            if float(min_durs[c]) <= ev.duration_s <= float(max_durs[c])
        ]
        all_events.extend(kept)
    return all_events
