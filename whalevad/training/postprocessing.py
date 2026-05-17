"""Postprocessing utilities (Section 5.8 of the paper).

The pipeline implemented here matches the paper exactly:

1.  Median-filter the per-frame call probabilities with a 500 ms
    kernel.
2.  Threshold per class with a value :math:`\\theta_c` selected to
    maximise the development F1 score.
3.  Convert the binary frame sequence into call segments.
4.  Refine: merge overlapping calls of the same type, eliminate
    duplicates, join calls separated by less than 500 ms, and discard
    calls longer than 30 s or shorter than 500 ms.

The module also provides :func:`collapse_seven_to_three` which is used
when training on the 7-class problem and evaluating on the 3-class
problem (Section 5.8: "the 7-class classifier output is collapsed into
the 3-class variant").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .dataset import CLASS_MAP_3, CLASS_MAP_7, SEVEN_TO_THREE


# ----------------------------------------------------------------- filter


def median_filter_1d(x: Tensor, *, kernel_size: int) -> Tensor:
    """Apply a 1-D median filter along the time dimension of ``x``.

    ``x`` is expected with shape ``(..., T, C)`` — the time dimension is
    the second-to-last.  Edges are handled by replicate-padding so the
    output preserves length.
    """
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    orig_shape = x.shape
    T, C = orig_shape[-2], orig_shape[-1]
    flat = x.reshape(-1, T, C)
    # Move time to last so we can unfold easily.
    flat_t = flat.transpose(1, 2)  # (N, C, T)
    flat_p = F.pad(flat_t, (pad, pad), mode="replicate")
    windows = flat_p.unfold(-1, kernel_size, 1)  # (N, C, T, K)
    med, _ = windows.median(dim=-1)
    med = med.transpose(1, 2)  # (N, T, C)
    return med.reshape(orig_shape)


# -------------------------------------------------------------- collapse


def collapse_seven_to_three(probs: Tensor) -> Tensor:
    """Reduce a 7-class probability tensor to the 3-class problem.

    For each 3-class target we take the maximum probability across the
    7-class members that map into it (Section 4 mapping).
    """
    if probs.size(-1) != 7:
        raise ValueError("Expected 7-class probability tensor")
    out_channels: List[Tensor] = []
    for cls in CLASS_MAP_3:
        members = [i for i, name in enumerate(CLASS_MAP_7) if SEVEN_TO_THREE[name] == cls]
        out_channels.append(probs[..., members].max(dim=-1).values)
    return torch.stack(out_channels, dim=-1)


# ------------------------------------------------------------------ calls


@dataclass
class CallEvent:
    """A detected call segment in seconds with the class label."""

    onset_s: float
    offset_s: float
    label: str
    score: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s


def _binary_to_intervals(binary: Tensor, hop_s: float, class_name: str) -> List[CallEvent]:
    """Convert a 1-D boolean frame sequence to time intervals."""
    if binary.dim() != 1:
        raise ValueError("binary must be 1-D")
    events: List[CallEvent] = []
    on = False
    start_idx = 0
    n = binary.size(0)
    bools = binary.tolist()
    for i, v in enumerate(bools):
        if v and not on:
            on = True
            start_idx = i
        elif not v and on:
            on = False
            events.append(
                CallEvent(
                    onset_s=start_idx * hop_s,
                    offset_s=i * hop_s,
                    label=class_name,
                )
            )
    if on:
        events.append(
            CallEvent(
                onset_s=start_idx * hop_s,
                offset_s=n * hop_s,
                label=class_name,
            )
        )
    return events


def _merge_and_filter(
    events: Sequence[CallEvent],
    *,
    merge_gap_s: float,
    min_duration_s: float,
    max_duration_s: float,
) -> List[CallEvent]:
    """Merge overlapping/adjacent same-class events and apply duration filters."""
    if not events:
        return []
    by_label: Dict[str, List[CallEvent]] = {}
    for ev in events:
        by_label.setdefault(ev.label, []).append(ev)
    merged: List[CallEvent] = []
    for label, evs in by_label.items():
        evs = sorted(evs, key=lambda e: e.onset_s)
        cur = evs[0]
        for ev in evs[1:]:
            if ev.onset_s - cur.offset_s <= merge_gap_s:
                cur = CallEvent(
                    onset_s=cur.onset_s,
                    offset_s=max(cur.offset_s, ev.offset_s),
                    label=label,
                    score=max(cur.score, ev.score),
                )
            else:
                merged.append(cur)
                cur = ev
        merged.append(cur)
    return [
        e for e in merged
        if min_duration_s <= e.duration_s <= max_duration_s
    ]


def probabilities_to_calls(
    probs: Tensor,  # (T, C)
    thresholds: Tensor,  # (C,)
    *,
    hop_s: float,
    class_names: Sequence[str],
    median_kernel_ms: float = 500.0,
    merge_gap_s: float = 0.5,
    min_duration_s: float = 0.5,
    max_duration_s: float = 30.0,
) -> List[CallEvent]:
    """Run the full postprocessing pipeline on per-frame probabilities."""
    if probs.dim() != 2:
        raise ValueError("probs must be a (T, C) tensor")
    if hop_s <= 0:
        raise ValueError("hop_s must be positive")
    # Median filter
    kernel = max(1, int(round(median_kernel_ms / 1000.0 / hop_s)))
    smoothed = median_filter_1d(probs.unsqueeze(0), kernel_size=kernel).squeeze(0)

    # Threshold and convert to intervals
    binary = smoothed >= thresholds.to(smoothed).unsqueeze(0)
    events: List[CallEvent] = []
    for c, name in enumerate(class_names):
        events.extend(_binary_to_intervals(binary[:, c], hop_s, name))

    # Attach mean score for each event
    scored: List[CallEvent] = []
    for ev in events:
        c = list(class_names).index(ev.label)
        i0 = max(0, int(ev.onset_s / hop_s))
        i1 = min(smoothed.size(0), int(ev.offset_s / hop_s))
        score = float(smoothed[i0:i1, c].mean().item()) if i1 > i0 else 0.0
        scored.append(
            CallEvent(
                onset_s=ev.onset_s,
                offset_s=ev.offset_s,
                label=ev.label,
                score=score,
            )
        )

    return _merge_and_filter(
        scored,
        merge_gap_s=merge_gap_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
    )


def merge_overlapping_windows(
    frame_probs: Sequence[Tensor],
    frame_starts_s: Sequence[float],
    *,
    hop_s: float,
    total_duration_s: float,
) -> Tensor:
    """Average probabilities across overlapping eval windows (Section 5.1).

    Given the per-window per-frame probabilities and their absolute start
    times within the source recording, build a single contiguous
    probability tensor by averaging the overlap region.
    """
    if len(frame_probs) == 0:
        raise ValueError("No frame probabilities supplied")
    num_classes = frame_probs[0].size(-1)
    total_frames = int(round(total_duration_s / hop_s))
    acc = torch.zeros((total_frames, num_classes), dtype=frame_probs[0].dtype)
    cnt = torch.zeros((total_frames,), dtype=torch.float32)
    for window, start_s in zip(frame_probs, frame_starts_s):
        if window.dim() != 2:
            raise ValueError("each window must be (T, C)")
        start_frame = int(round(start_s / hop_s))
        end_frame = min(total_frames, start_frame + window.size(0))
        usable = end_frame - start_frame
        if usable <= 0:
            continue
        acc[start_frame:end_frame] += window[:usable].to(acc.dtype)
        cnt[start_frame:end_frame] += 1.0
    cnt = cnt.clamp_min(1.0).unsqueeze(-1)
    return acc / cnt
