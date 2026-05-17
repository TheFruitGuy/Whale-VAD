"""Stochastic negative mini-batch undersampling (Section 5.5).

Whale vocalisations occupy roughly 5 % of the training audio.  Following
the paper, we resample a fresh subset of negative (no-call) segments at
the start of each training epoch, sized to approximately match the
number of positive segments.  Positive segments themselves remain
unchanged across epochs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .dataset import AudioFile, Segment


@dataclass
class _Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def _negative_intervals(af: AudioFile, *, guard_s: float = 0.0) -> List[_Interval]:
    """Compute disjoint negative intervals — regions with no annotation."""
    if af.duration_s <= 0:
        return []
    occupied: List[_Interval] = []
    for ann in af.annotations:
        s = max(0.0, ann.onset_s - guard_s)
        e = min(af.duration_s, ann.offset_s + guard_s)
        if e > s:
            occupied.append(_Interval(s, e))
    if not occupied:
        return [_Interval(0.0, af.duration_s)]
    occupied.sort(key=lambda iv: iv.start)
    merged: List[_Interval] = [occupied[0]]
    for iv in occupied[1:]:
        last = merged[-1]
        if iv.start <= last.end:
            last.end = max(last.end, iv.end)
        else:
            merged.append(iv)
    negatives: List[_Interval] = []
    t = 0.0
    for iv in merged:
        if iv.start > t:
            negatives.append(_Interval(t, iv.start))
        t = iv.end
    if t < af.duration_s:
        negatives.append(_Interval(t, af.duration_s))
    return [n for n in negatives if n.duration > 0]


class StochasticNegativeSampler:
    """Generate a fresh list of negative segments per epoch.

    The sampler weights each audio file by its total negative duration so
    that recordings with more "silent" time contribute more negative
    examples.  Within a chosen file, a random sub-window is drawn from
    one of its negative intervals.
    """

    def __init__(
        self,
        audio_files: Sequence[AudioFile],
        *,
        min_dur_s: float,
        max_dur_s: float,
        guard_s: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        if min_dur_s <= 0 or max_dur_s < min_dur_s:
            raise ValueError("Invalid negative duration range")
        self.audio_files = list(audio_files)
        self.min_dur_s = min_dur_s
        self.max_dur_s = max_dur_s
        self.guard_s = guard_s
        self._rng = rng or random.Random()
        self._neg_intervals: List[Tuple[AudioFile, List[_Interval], float]] = []
        total_neg_duration = 0.0
        for af in self.audio_files:
            negs = _negative_intervals(af, guard_s=guard_s)
            usable = [n for n in negs if n.duration >= min_dur_s]
            file_total = sum(n.duration for n in usable)
            if not usable:
                continue
            self._neg_intervals.append((af, usable, file_total))
            total_neg_duration += file_total
        self._total_neg_duration = total_neg_duration
        self._file_weights = [t / total_neg_duration for _, _, t in self._neg_intervals]

    @property
    def total_negative_duration_s(self) -> float:
        return self._total_neg_duration

    @property
    def has_negatives(self) -> bool:
        return len(self._neg_intervals) > 0

    def sample(self, count: int) -> List[Segment]:
        """Draw ``count`` negative segments from the underlying audio pool."""
        if count <= 0 or not self.has_negatives:
            return []
        segments: List[Segment] = []
        files = [f for f, _, _ in self._neg_intervals]
        while len(segments) < count:
            af = self._rng.choices(files, weights=self._file_weights, k=1)[0]
            intervals = next(
                ivs for f, ivs, _ in self._neg_intervals if f is af
            )
            interval_weights = [iv.duration for iv in intervals]
            iv = self._rng.choices(intervals, weights=interval_weights, k=1)[0]
            dur = self._rng.uniform(self.min_dur_s, self.max_dur_s)
            dur = min(dur, iv.duration)
            max_start = iv.end - dur
            start = self._rng.uniform(iv.start, max_start) if max_start > iv.start else iv.start
            end = start + dur
            segments.append(
                Segment(
                    audio_path=af.path,
                    start_s=start,
                    end_s=end,
                    annotations=(),  # negative by construction
                    is_positive=False,
                )
            )
        return segments
