"""Dataset utilities for Whale-VAD training.

Implements the segment-based data pipeline described in Section 5.1
of the paper.  Each segment corresponds to a human annotation extended
with a randomly sampled collar.  An associated discrete classification
target is constructed at the 20 ms frame resolution used by the model.
Negative segments (no calls) are produced separately via the
``StochasticNegativeSampler`` (Section 5.5).
"""

from __future__ import annotations

import csv
import math
import random
import re
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ._progress import progress


# ---------------------------------------------------------------------- Classes

CLASS_MAP_7: Tuple[str, ...] = (
    "BmA",
    "BmB",
    "BmZ",
    "BmD",
    "BpD",
    "Bp20",
    "Bp20plus",
)
"""Original 7-call labels from the ATBFL dataset."""

CLASS_MAP_3: Tuple[str, ...] = ("bmabz", "d", "bp")
"""Three-class collapsed labels used at challenge evaluation."""

SEVEN_TO_THREE: Mapping[str, str] = {
    "BmA": "bmabz",
    "BmB": "bmabz",
    "BmZ": "bmabz",
    "BmD": "d",
    "BpD": "d",
    "Bp20": "bp",
    "Bp20plus": "bp",
}
"""Mapping from the 7-class labels to the 3-class problem (Section 4)."""


def class_to_index(class_names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def resolve_class_names(num_classes: int) -> Tuple[str, ...]:
    if num_classes == 7:
        return CLASS_MAP_7
    if num_classes == 3:
        return CLASS_MAP_3
    raise ValueError(f"Unsupported num_classes={num_classes}")


# ----------------------------------------------------------------- Annotations


@dataclass(frozen=True)
class Annotation:
    """A single whale-call annotation.

    Onsets and offsets are in seconds relative to the start of the audio
    file.  ``low_freq``/``high_freq`` are optional Hz bounding-box edges
    used by the multi-objective regression head (Section 5.7).
    """

    onset_s: float
    offset_s: float
    label: str
    low_freq_hz: Optional[float] = None
    high_freq_hz: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.offset_s - self.onset_s)


@dataclass
class AudioFile:
    """Pointer to a single audio recording with its annotations."""

    path: Path
    duration_s: float
    sample_rate: int
    annotations: List[Annotation] = field(default_factory=list)

    @property
    def num_annotations(self) -> int:
        return len(self.annotations)


@dataclass(frozen=True)
class Segment:
    """A contiguous interval of audio to be presented to the model."""

    audio_path: Path
    start_s: float
    end_s: float
    annotations: Tuple[Annotation, ...] = ()
    is_positive: bool = True

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


# ------------------------------------------------------------- CSV annotations


_DEFAULT_LABEL_KEYS = (
    "label", "tag", "class", "event_label", "annotation",
    "Label", "Tag", "Annotation",
)
_DEFAULT_ONSET_KEYS = (
    "onset", "start", "start_time", "begin_time",
    "start_datetime", "Begin Time (s)",
)
_DEFAULT_OFFSET_KEYS = (
    "offset", "end", "end_time", "stop", "stop_time",
    "end_datetime", "End Time (s)",
)
_DEFAULT_LOW_FREQ_KEYS = ("low_frequency", "low_freq", "freq_low", "Low Freq (Hz)")
_DEFAULT_HIGH_FREQ_KEYS = ("high_frequency", "high_freq", "freq_high", "High Freq (Hz)")


def _pick(row: Mapping[str, str], keys: Sequence[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return None


# Filenames like ``2015-02-04T03-00-00_000.wav`` encode the recording's start
# datetime.  We accept the ISO date, dash-separated time, and optional
# fractional seconds (treated as milliseconds when 3 digits).
_FILENAME_DATETIME_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})(?:[_.](\d+))?"
)


def parse_audio_file_start_datetime(filename: str) -> Optional[datetime]:
    """Extract the recording start datetime from a BioDCASE-style filename.

    Returns ``None`` when the filename doesn't match the expected pattern.
    """
    m = _FILENAME_DATETIME_RE.search(Path(filename).stem)
    if not m:
        return None
    date, hh, mm, ss, fraction = m.groups()
    fraction_str = fraction or "0"
    # Treat fractional seconds as milliseconds (typical BioDCASE convention).
    micros = (fraction_str + "000000")[:6]
    try:
        return datetime.fromisoformat(f"{date}T{hh}:{mm}:{ss}.{micros}")
    except ValueError:
        return None


def _annotation_time_to_seconds(
    value: Optional[str], file_start: Optional[datetime]
) -> Optional[float]:
    """Convert an annotation time field to seconds-from-file-start.

    Accepts both numeric strings (already in seconds) and ISO datetime
    strings.  For the datetime variant, ``file_start`` is required.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if file_start is None:
        return None
    return (dt - file_start).total_seconds()


def load_annotations_csv(
    csv_path: Path,
    *,
    file_start: Optional[datetime] = None,
) -> List[Annotation]:
    """Parse a CSV annotation file.

    Supports several common header conventions:

    * ``onset/offset/label`` (numeric seconds, used by ATBFL 2025)
    * ``Begin Time (s)/End Time (s)/Tag`` (Raven Pro)
    * ``start_datetime/end_datetime/annotation`` (BioDCASE 2026, where
      times are ISO datetimes that must be converted to seconds using
      ``file_start`` — the recording's start datetime).
    """
    rows = _read_rows(csv_path)
    resolver = (lambda _fname: file_start) if file_start is not None else None
    return _rows_to_annotations(rows, file_start_resolver=resolver)


# ------------------------------------------------------- Dataset construction


def _find_paired_files(
    root: Path,
    audio_subdir: str,
    annotation_subdir: str,
    audio_ext: str,
    annotation_ext: str,
) -> List[Tuple[Path, Optional[Path]]]:
    """Discover audio files and pair them with annotation CSVs.

    Layouts supported (probed in order):

    1. ``root/{audio_subdir}/<site>/<file>.wav`` paired with
       ``root/{annotation_subdir}/<site>/<file>.csv`` (per-file CSVs).
    2. ``root/{audio_subdir}/<site>/<file>.wav`` paired with
       ``root/{annotation_subdir}/<site>.csv`` (one CSV per site).
    3. ``root/{site}/<file>.wav`` with ``root/{site}/annotations.csv``
       carrying all calls for that site (with a ``filename`` column).

    For layouts (2) and (3) the loader filters per-audio annotations
    later using the ``filename`` column in the CSV.
    """
    pairs: List[Tuple[Path, Optional[Path]]] = []
    audio_dir = root / audio_subdir if (root / audio_subdir).exists() else root
    ann_dir = (
        root / annotation_subdir
        if (root / annotation_subdir).exists()
        else audio_dir
    )

    for audio_path in sorted(audio_dir.rglob(f"*{audio_ext}")):
        rel = audio_path.relative_to(audio_dir)
        # Layout 1: per-file CSV mirroring the audio tree.
        candidate = ann_dir / rel.with_suffix(annotation_ext)
        if candidate.exists():
            pairs.append((audio_path, candidate))
            continue
        # Layout 2: one CSV per site in the annotations folder.
        if len(rel.parts) >= 2:
            site = rel.parts[0]
            site_index = ann_dir / f"{site}{annotation_ext}"
            if site_index.exists():
                pairs.append((audio_path, site_index))
                continue
        # Layout 3: per-site annotations.csv co-located with the audio.
        site_csv = audio_path.parent / f"annotations{annotation_ext}"
        if site_csv.exists():
            pairs.append((audio_path, site_csv))
            continue
        pairs.append((audio_path, None))
    return pairs


def _filter_annotations_for_file(
    all_anns: List[Annotation],
    filename_filter: Optional[str],
    rows: Optional[List[Mapping[str, str]]] = None,
) -> List[Annotation]:
    # When the CSV is a per-site index with a filename column we need to
    # filter; otherwise return all annotations.
    if filename_filter is None or rows is None:
        return all_anns
    keep: List[Annotation] = []
    for ann, row in zip(all_anns, rows):
        fname = row.get("filename") or row.get("file") or row.get("audio_file") or ""
        if Path(fname).name == filename_filter:
            keep.append(ann)
    return keep


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with open(csv_path, "r", newline="") as fh:
        try:
            sniff = csv.Sniffer().sniff(fh.read(4096))
            fh.seek(0)
            reader = csv.DictReader(fh, dialect=sniff)
        except csv.Error:
            fh.seek(0)
            reader = csv.DictReader(fh)
        return [dict(r) for r in reader]


def load_audio_files(
    root: Path,
    *,
    audio_subdir: str = "audio",
    annotation_subdir: str = "annotations",
    audio_ext: str = ".wav",
    annotation_ext: str = ".csv",
    show_progress: bool = True,
) -> List[AudioFile]:
    """Discover and index all audio files plus their annotations under ``root``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    pairs = _find_paired_files(
        root, audio_subdir, annotation_subdir, audio_ext, annotation_ext
    )
    if not pairs:
        raise RuntimeError(f"No audio files found under {root}")

    audio_files: List[AudioFile] = []
    # Cache per-site CSV rows so we only parse each file once.
    site_csv_cache: Dict[Path, List[Dict[str, str]]] = {}

    iterator = progress(
        pairs,
        desc=f"Indexing {root.name}",
        total=len(pairs),
        disable=not show_progress,
        unit="file",
    )
    for audio_path, csv_path in iterator:
        sample_rate, num_frames = _audio_info(audio_path)
        duration_s = num_frames / float(sample_rate)

        anns: List[Annotation] = []
        if csv_path is not None and csv_path.exists():
            # Treat the CSV as a per-site index whenever its stem does not
            # match the audio file name (i.e. it cannot be a per-file CSV).
            is_per_site_index = csv_path.stem != audio_path.stem
            if is_per_site_index:
                rows = site_csv_cache.setdefault(csv_path, _read_rows(csv_path))
                all_anns = _rows_to_annotations(
                    rows, file_start_resolver=parse_audio_file_start_datetime
                )
                anns = _filter_annotations_for_file(all_anns, audio_path.name, rows)
            else:
                anns = load_annotations_csv(
                    csv_path,
                    file_start=parse_audio_file_start_datetime(audio_path.name),
                )
        audio_files.append(
            AudioFile(
                path=audio_path,
                duration_s=duration_s,
                sample_rate=sample_rate,
                annotations=anns,
            )
        )
    return audio_files


def _rows_to_annotations(
    rows: Sequence[Mapping[str, str]],
    *,
    file_start_resolver: Optional[Callable[[str], Optional[datetime]]] = None,
) -> List[Annotation]:
    """Convert CSV rows into ``Annotation`` objects.

    When ``file_start_resolver`` is supplied, datetime-valued onsets/offsets
    are converted to seconds-from-file-start using the per-row filename
    (taken from the ``filename``/``file``/``audio_file`` column).
    """
    out: List[Annotation] = []
    for row in rows:
        label = _pick(row, _DEFAULT_LABEL_KEYS)
        onset_raw = _pick(row, _DEFAULT_ONSET_KEYS)
        offset_raw = _pick(row, _DEFAULT_OFFSET_KEYS)
        if label is None or onset_raw is None or offset_raw is None:
            continue

        file_start: Optional[datetime] = None
        if file_start_resolver is not None:
            fname = (
                row.get("filename")
                or row.get("file")
                or row.get("audio_file")
                or ""
            )
            fname = Path(fname).name
            if fname:
                file_start = file_start_resolver(fname)

        onset = _annotation_time_to_seconds(onset_raw, file_start)
        offset = _annotation_time_to_seconds(offset_raw, file_start)
        if onset is None or offset is None or offset <= onset:
            continue

        low_raw = _pick(row, _DEFAULT_LOW_FREQ_KEYS)
        high_raw = _pick(row, _DEFAULT_HIGH_FREQ_KEYS)
        try:
            low_freq = float(low_raw) if low_raw is not None else None
            high_freq = float(high_raw) if high_raw is not None else None
        except ValueError:
            low_freq = None
            high_freq = None
        out.append(
            Annotation(
                onset_s=onset,
                offset_s=offset,
                label=label.strip(),
                low_freq_hz=low_freq,
                high_freq_hz=high_freq,
            )
        )
    return out


# ----------------------------------------------------------- Segment helpers


# Case-insensitive lookup tables so that BioDCASE-2026 lowercase labels
# (``bma``, ``bmb``, ...) map to the canonical mixed-case names used here.
_CLASS_MAP_7_LOWER = {name.lower(): name for name in CLASS_MAP_7}
_CLASS_MAP_3_LOWER = {name.lower(): name for name in CLASS_MAP_3}
_SEVEN_TO_THREE_LOWER = {k.lower(): v for k, v in SEVEN_TO_THREE.items()}


def map_label_to_class(label: str, num_classes: int) -> Optional[str]:
    """Map a raw annotation label onto the configured class space.

    Case-insensitive; returns ``None`` for unknown labels.
    """
    if not label:
        return None
    key = label.strip().lower()
    if num_classes == 7:
        return _CLASS_MAP_7_LOWER.get(key)
    if num_classes == 3:
        if key in _CLASS_MAP_3_LOWER:
            return _CLASS_MAP_3_LOWER[key]
        return _SEVEN_TO_THREE_LOWER.get(key)
    raise ValueError(f"Unsupported num_classes={num_classes}")


def _annotations_for_segment(
    annotations: Sequence[Annotation],
    start_s: float,
    end_s: float,
) -> Tuple[Annotation, ...]:
    return tuple(a for a in annotations if a.offset_s > start_s and a.onset_s < end_s)


def build_positive_segments(
    audio_files: Sequence[AudioFile],
    *,
    num_classes: int,
    collar_min_s: float,
    collar_max_s: float,
    rng: random.Random,
    show_progress: bool = True,
) -> List[Segment]:
    """Build one segment per annotation with a random collar (Section 5.1)."""
    segments: List[Segment] = []
    iterator = progress(
        audio_files,
        desc="Building positive segments",
        total=len(audio_files),
        disable=not show_progress,
        unit="file",
    )
    for af in iterator:
        for ann in af.annotations:
            if map_label_to_class(ann.label, num_classes) is None:
                continue
            left = rng.uniform(collar_min_s, collar_max_s)
            right = rng.uniform(collar_min_s, collar_max_s)
            start = max(0.0, ann.onset_s - left)
            end = min(af.duration_s, ann.offset_s + right)
            if end <= start:
                continue
            segments.append(
                Segment(
                    audio_path=af.path,
                    start_s=start,
                    end_s=end,
                    annotations=_annotations_for_segment(af.annotations, start, end),
                    is_positive=True,
                )
            )
    return segments


def build_eval_segments(
    audio_files: Sequence[AudioFile],
    *,
    segment_s: float,
    overlap_s: float,
    show_progress: bool = True,
) -> List[Segment]:
    """Tile audio files into overlapping fixed-length segments (Section 5.1)."""
    if overlap_s >= segment_s:
        raise ValueError("overlap must be smaller than segment length")
    stride = segment_s - overlap_s
    out: List[Segment] = []
    iterator = progress(
        audio_files,
        desc="Building eval tiles",
        total=len(audio_files),
        disable=not show_progress,
        unit="file",
    )
    for af in iterator:
        t = 0.0
        while t < af.duration_s:
            end = min(af.duration_s, t + segment_s)
            if end - t < 0.1:  # tiny tail
                break
            out.append(
                Segment(
                    audio_path=af.path,
                    start_s=t,
                    end_s=end,
                    annotations=_annotations_for_segment(af.annotations, t, end),
                    is_positive=False,
                )
            )
            if end >= af.duration_s:
                break
            t += stride
    return out


# ------------------------------------------------------------- Frame labels


def num_spec_frames(num_samples: int, *, n_fft: int, hop_length: int) -> int:
    """Number of STFT frames produced from ``num_samples`` audio samples.

    Matches torchaudio's ``Spectrogram`` with ``center=False``::

        frames = (num_samples - n_fft) // hop_length + 1
    """
    if num_samples < n_fft:
        return 0
    return (num_samples - n_fft) // hop_length + 1


def build_frame_labels(
    segment: Segment,
    *,
    num_frames: int,
    hop_s: float,
    frame_offset_s: float,
    class_to_idx: Mapping[str, int],
    num_classes: int,
) -> Tensor:
    """Construct the binary frame-level target vector for one segment.

    A frame ``n`` is labelled positive for class ``c`` when there exists
    a same-class annotation that *fully* covers the frame interval
    ``[n*hop, (n+1)*hop)`` (Section 5.1: "When a human annotation
    boundary intersects completely with the classification target
    vector at a time instant, the label is true").
    """
    labels = torch.zeros((num_frames, num_classes), dtype=torch.float32)
    if num_frames == 0:
        return labels
    for ann in segment.annotations:
        target_label = map_label_to_class(ann.label, num_classes)
        if target_label is None:
            continue
        cls_idx = class_to_idx[target_label]
        # Frame n covers [t0, t1) inside the segment.
        # Annotation interval relative to segment start:
        a0 = ann.onset_s - segment.start_s
        a1 = ann.offset_s - segment.start_s
        # First frame whose start is >= a0:
        start_frame = max(0, int(math.ceil((a0 - frame_offset_s) / hop_s)))
        # Last frame whose end (n+1)*hop is <= a1:
        end_frame = int(math.floor((a1 - frame_offset_s) / hop_s)) - 1
        end_frame = min(num_frames - 1, end_frame)
        if end_frame >= start_frame:
            labels[start_frame : end_frame + 1, cls_idx] = 1.0
    return labels


# ------------------------------------------------------------- The Dataset


class ATBFLDataset(Dataset):
    """Audio-segment dataset for Whale-VAD training and validation.

    The dataset is *segment*-oriented: each item corresponds to one
    contiguous slice of an underlying audio file, plus its discrete
    frame-level classification target.  Construct positive segments
    once (they are fixed for the training run) and pass a list of
    negative segments — rebuild the negative list each epoch via
    :class:`StochasticNegativeSampler` to obtain the stochastic
    undersampling described in Section 5.5.
    """

    def __init__(
        self,
        segments: Sequence[Segment],
        *,
        num_classes: int,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        class_names: Optional[Sequence[str]] = None,
        audio_loader: Optional[Callable[[Path, int, int], Tensor]] = None,
    ) -> None:
        self._segments: List[Segment] = list(segments)
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.class_names: Tuple[str, ...] = tuple(
            class_names if class_names is not None else resolve_class_names(num_classes)
        )
        self.class_to_idx: Dict[str, int] = class_to_index(self.class_names)
        self._loader = audio_loader or _default_audio_loader

    # ------------------------------------------------------------- segments

    @property
    def segments(self) -> Sequence[Segment]:
        return self._segments

    def set_segments(self, segments: Sequence[Segment]) -> None:
        """Replace the segment list — used to refresh negatives each epoch."""
        self._segments = list(segments)

    def __len__(self) -> int:
        return len(self._segments)

    # --------------------------------------------------------------- access

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        seg = self._segments[index]
        start_sample = int(round(seg.start_s * self.sample_rate))
        end_sample = int(round(seg.end_s * self.sample_rate))
        num_samples = max(self.n_fft, end_sample - start_sample)

        audio = self._loader(seg.audio_path, start_sample, num_samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)  # mono

        n_frames = num_spec_frames(
            audio.size(-1), n_fft=self.n_fft, hop_length=self.hop_length
        )
        hop_s = self.hop_length / float(self.sample_rate)
        # Frame n starts at sample n*hop, i.e. at time n*hop_s from the start
        # of the requested audio interval.  No additional offset because the
        # STFT does not pad.
        labels = build_frame_labels(
            seg,
            num_frames=n_frames,
            hop_s=hop_s,
            frame_offset_s=0.0,
            class_to_idx=self.class_to_idx,
            num_classes=self.num_classes,
        )

        bboxes, bbox_mask = self._encode_bounding_boxes(seg)

        return {
            "audio": audio,                          # (1, T)
            "audio_length": torch.tensor(audio.size(-1), dtype=torch.long),
            "frame_labels": labels,                  # (n_frames, C)
            "frame_length": torch.tensor(n_frames, dtype=torch.long),
            "bboxes": bboxes,                        # (max_boxes, 4)
            "bbox_mask": bbox_mask,                  # (max_boxes,) bool
            "is_positive": torch.tensor(seg.is_positive, dtype=torch.bool),
        }

    # --------------------------------------------------- bounding-box targets

    def _encode_bounding_boxes(
        self, seg: Segment, *, max_boxes: int = 64
    ) -> Tuple[Tensor, Tensor]:
        bboxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
        mask = torch.zeros((max_boxes,), dtype=torch.bool)
        if not seg.annotations:
            return bboxes, mask
        nyquist = self.sample_rate / 2.0
        i = 0
        for ann in seg.annotations:
            if i >= max_boxes:
                break
            cls = map_label_to_class(ann.label, self.num_classes)
            if cls is None:
                continue
            x0 = (ann.onset_s - seg.start_s) / max(seg.duration_s, 1e-6)
            x1 = (ann.offset_s - seg.start_s) / max(seg.duration_s, 1e-6)
            x0 = max(0.0, min(1.0, x0))
            x1 = max(0.0, min(1.0, x1))
            if ann.low_freq_hz is None or ann.high_freq_hz is None:
                y0, y1 = 0.0, 1.0
            else:
                y0 = max(0.0, min(1.0, ann.low_freq_hz / nyquist))
                y1 = max(0.0, min(1.0, ann.high_freq_hz / nyquist))
            bboxes[i] = torch.tensor([x0, y0, x1, y1])
            mask[i] = True
            i += 1
        return bboxes, mask


# ------------------------------------------------------------- audio loading


def _have_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401

        return True
    except Exception:
        return False


def _audio_info(path: Path) -> Tuple[int, int]:
    """Return ``(sample_rate, num_frames)`` for the audio file at ``path``.

    Prefers ``soundfile`` when available because its WAV reader does
    not require FFmpeg / TorchCodec.  Falls back to the stdlib ``wave``
    module for plain PCM WAVs and finally to ``torchaudio.load``.
    """
    p = str(path)
    if _have_soundfile():
        import soundfile as sf

        info = sf.info(p)
        return int(info.samplerate), int(info.frames)
    # stdlib wave (PCM WAV only)
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(p, "rb") as fh:
                return int(fh.getframerate()), int(fh.getnframes())
        except wave.Error:
            pass
    # last resort: full decode through torchaudio
    import torchaudio  # local import keeps this optional

    wav, sr = torchaudio.load(p)
    return int(sr), int(wav.size(-1))


def _default_audio_loader(path: Path, frame_offset: int, num_frames: int) -> Tensor:
    """Load a slice of audio from ``path`` starting at ``frame_offset`` samples.

    Returns a (channels, samples) float32 tensor in [-1, 1].  Uses
    soundfile when available so that the pipeline runs without
    FFmpeg / TorchCodec.
    """
    p = str(path)
    if _have_soundfile():
        import soundfile as sf

        with sf.SoundFile(p, "r") as fh:
            fh.seek(int(frame_offset))
            data = fh.read(frames=int(num_frames), dtype="float32", always_2d=True)
        # soundfile returns (samples, channels)
        tensor = torch.from_numpy(data).transpose(0, 1).contiguous()
        return tensor
    import torchaudio

    audio, _ = torchaudio.load(
        p,
        frame_offset=int(frame_offset),
        num_frames=int(num_frames),
    )
    return audio


# ------------------------------------------------------------- batch collate


def collate_segments(batch: Sequence[Mapping[str, Tensor]]) -> Dict[str, Tensor]:
    """Pad variable-length segments into a fixed-shape mini-batch.

    Padding is applied to the raw audio (so that the STFT inside the
    feature extractor produces a consistent number of frames) and to
    the frame-level labels.  ``frame_length`` is preserved so that
    downstream code can mask the padded region when computing loss
    (Section 5.1: "This padding was removed from each segment during
    loss calculation and model weight backpropagation").
    """
    batch = list(batch)
    # Sort by length descending — required for pack_padded_sequence(enforce_sorted=False)
    # is convenient but the model already passes ``enforce_sorted=False``;
    # keep insertion order to make debugging easier.
    audio_lengths = [b["audio_length"].item() for b in batch]
    frame_lengths = [b["frame_length"].item() for b in batch]
    max_audio = max(audio_lengths)
    max_frames = max(frame_lengths)
    num_classes = batch[0]["frame_labels"].size(-1)

    audio = torch.zeros((len(batch), 1, max_audio), dtype=batch[0]["audio"].dtype)
    labels = torch.zeros(
        (len(batch), max_frames, num_classes), dtype=batch[0]["frame_labels"].dtype
    )
    is_positive = torch.zeros((len(batch),), dtype=torch.bool)
    bboxes = torch.stack([b["bboxes"] for b in batch], dim=0)
    bbox_mask = torch.stack([b["bbox_mask"] for b in batch], dim=0)

    for i, b in enumerate(batch):
        T = b["audio"].size(-1)
        audio[i, :, :T] = b["audio"]
        F = b["frame_labels"].size(0)
        labels[i, :F, :] = b["frame_labels"]
        is_positive[i] = b["is_positive"]

    return {
        "audio": audio,
        "audio_length": torch.tensor(audio_lengths, dtype=torch.long),
        "frame_labels": labels,
        "frame_length": torch.tensor(frame_lengths, dtype=torch.long),
        "bboxes": bboxes,
        "bbox_mask": bbox_mask,
        "is_positive": is_positive,
    }
