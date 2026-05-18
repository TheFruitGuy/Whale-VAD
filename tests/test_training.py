"""Unit tests for the Whale-VAD training pipeline.

These tests synthesise a tiny dataset on disk and exercise the
preprocessing, sampling, loss, postprocessing and trainer modules to
ensure they all interoperate correctly with the paper-faithful
configuration.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torchaudio

from whalevad.training import (
    ATBFLDataset,
    Annotation,
    Segment,
    StochasticNegativeSampler,
    TrainingConfig,
    WeightedBCELoss,
    FocalLoss,
    MultiObjectiveLoss,
    collate_segments,
    collapse_seven_to_three,
    find_optimal_thresholds,
    median_filter_1d,
    compute_bce_pos_weight,
)
from whalevad.training.dataset import (
    build_frame_labels,
    build_positive_segments,
    load_audio_files,
    num_spec_frames,
)
from whalevad.training.postprocessing import probabilities_to_calls
from whalevad.training.trainer import Trainer


# -------------------------------------------------------------- fixtures


def _write_audio(path: Path, *, duration_s: float, sample_rate: int = 250) -> None:
    """Write a short sinusoid as a placeholder audio file (16-bit PCM WAV)."""
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sample_rate)
    t = torch.linspace(0, duration_s, n)
    wav = 0.1 * torch.sin(2 * math.pi * 20.0 * t)
    samples = (wav * 32767.0).clamp(-32768, 32767).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(samples)


def _write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["onset", "offset", "label", "low_frequency", "high_frequency"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a minimal train/val tree with synthetic audio and annotations."""
    train = tmp_path / "train"
    val = tmp_path / "val"

    # Train file with two annotations
    _write_audio(train / "audio" / "siteA" / "rec1.wav", duration_s=30.0)
    _write_csv(
        train / "annotations" / "siteA" / "rec1.csv",
        [
            {"onset": "5.0", "offset": "7.5", "label": "BmA",
             "low_frequency": "20", "high_frequency": "40"},
            {"onset": "15.0", "offset": "18.0", "label": "Bp20",
             "low_frequency": "15", "high_frequency": "30"},
        ],
    )
    _write_audio(train / "audio" / "siteA" / "rec2.wav", duration_s=20.0)
    _write_csv(
        train / "annotations" / "siteA" / "rec2.csv",
        [
            {"onset": "3.0", "offset": "4.0", "label": "BmD",
             "low_frequency": "10", "high_frequency": "30"},
        ],
    )

    # Validation file with one annotation
    _write_audio(val / "audio" / "siteB" / "valrec.wav", duration_s=40.0)
    _write_csv(
        val / "annotations" / "siteB" / "valrec.csv",
        [
            {"onset": "10.0", "offset": "13.0", "label": "BmA",
             "low_frequency": "20", "high_frequency": "40"},
        ],
    )
    return train, val


# ------------------------------------------------------------- dataset


def test_load_audio_files(tiny_dataset: Tuple[Path, Path]) -> None:
    train, _ = tiny_dataset
    files = load_audio_files(train)
    assert len(files) == 2
    assert all(af.sample_rate == 250 for af in files)
    total_anns = sum(len(af.annotations) for af in files)
    assert total_anns == 3


def test_load_audio_files_per_site_index_csv(tmp_path: Path) -> None:
    """BioDCASE 2026 layout: one CSV per site under annotations/."""
    root = tmp_path / "ds"
    _write_audio(root / "audio" / "siteX" / "rec1.wav", duration_s=10.0)
    _write_audio(root / "audio" / "siteX" / "rec2.wav", duration_s=10.0)
    # Per-site CSV with a filename column.
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    with open(root / "annotations" / "siteX.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "onset", "offset", "label"]
        )
        writer.writeheader()
        writer.writerow({"filename": "rec1.wav", "onset": "1.0", "offset": "2.0",
                         "label": "BmA"})
        writer.writerow({"filename": "rec2.wav", "onset": "3.0", "offset": "4.0",
                         "label": "BmD"})
        writer.writerow({"filename": "rec2.wav", "onset": "5.5", "offset": "6.5",
                         "label": "Bp20"})
    files = load_audio_files(root)
    assert len(files) == 2
    by_name = {af.path.name: af for af in files}
    assert len(by_name["rec1.wav"].annotations) == 1
    assert len(by_name["rec2.wav"].annotations) == 2


def test_load_audio_files_biodcase_2026_schema(tmp_path: Path) -> None:
    """BioDCASE 2026: ISO datetime onsets + lowercase labels + 'annotation' column."""
    root = tmp_path / "ds"
    # Filename encodes the recording start time per BioDCASE convention.
    audio_name = "2015-02-04T03-00-00_000.wav"
    _write_audio(root / "audio" / "siteY" / audio_name, duration_s=4000.0)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    with open(root / "annotations" / "siteY.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset", "filename", "annotation", "annotator",
                "low_frequency", "high_frequency",
                "start_datetime", "end_datetime",
            ],
        )
        writer.writeheader()
        # 27 min 32.053 s into the recording
        writer.writerow({
            "dataset": "siteY", "filename": audio_name, "annotation": "bma",
            "annotator": "test", "low_frequency": "21.9", "high_frequency": "28.4",
            "start_datetime": "2015-02-04T03:27:32.053000",
            "end_datetime":   "2015-02-04T03:27:43.709000",
        })
        # 5 s call near the start, label is lowercase 7-class
        writer.writerow({
            "dataset": "siteY", "filename": audio_name, "annotation": "bp20",
            "annotator": "test", "low_frequency": "18.0", "high_frequency": "22.0",
            "start_datetime": "2015-02-04T03:00:10.000000",
            "end_datetime":   "2015-02-04T03:00:15.000000",
        })
    files = load_audio_files(root)
    assert len(files) == 1
    anns = files[0].annotations
    assert len(anns) == 2
    # First annotation should be at 27*60 + 32.053 seconds into the file.
    first = next(a for a in anns if a.label == "bma")
    assert pytest.approx(first.onset_s, abs=1e-3) == 27 * 60 + 32.053
    assert pytest.approx(first.offset_s - first.onset_s, abs=1e-3) == 11.656
    second = next(a for a in anns if a.label == "bp20")
    assert pytest.approx(second.onset_s, abs=1e-3) == 10.0
    assert pytest.approx(second.offset_s, abs=1e-3) == 15.0


def test_num_spec_frames_matches_torchaudio() -> None:
    audio = torch.randn(1, 1, 1250)  # 5 seconds @ 250 Hz
    spec = torchaudio.transforms.Spectrogram(
        n_fft=256, win_length=256, hop_length=5, center=False, power=None
    )(audio)
    # shape: (1, 1, freq, time)
    assert num_spec_frames(audio.size(-1), n_fft=256, hop_length=5) == spec.size(-1)


def test_build_frame_labels_three_class() -> None:
    ann = Annotation(onset_s=1.0, offset_s=2.0, label="BmA")
    seg = Segment(
        audio_path=Path("/dev/null"),
        start_s=0.0,
        end_s=3.0,
        annotations=(ann,),
        is_positive=True,
    )
    labels = build_frame_labels(
        seg,
        num_frames=int(3.0 / 0.02),
        hop_s=0.02,
        frame_offset_s=0.0,
        class_to_idx={"bmabz": 0, "d": 1, "bp": 2},
        num_classes=3,
    )
    # bmabz column should be 1 on frames fully covered by [1.0, 2.0]
    bm = labels[:, 0]
    # Frame n covers [n*0.02, (n+1)*0.02).  Fully covered iff n*0.02 >= 1.0
    # and (n+1)*0.02 <= 2.0, i.e. n in [50, 99].
    assert bm[:50].sum() == 0
    assert bm[50:100].all()
    assert bm[100:].sum() == 0
    # Other classes should remain zero
    assert labels[:, 1:].sum() == 0


def test_build_positive_segments(tiny_dataset: Tuple[Path, Path]) -> None:
    import random as _random

    train, _ = tiny_dataset
    files = load_audio_files(train)
    rng = _random.Random(0)
    pos = build_positive_segments(
        files, num_classes=3, collar_min_s=1.0, collar_max_s=2.0, rng=rng
    )
    assert len(pos) == 3
    for seg in pos:
        assert seg.is_positive
        assert seg.duration_s > 0
        assert seg.duration_s <= 30.0


def test_dataset_getitem_and_collate(tiny_dataset: Tuple[Path, Path]) -> None:
    import random as _random

    train, _ = tiny_dataset
    files = load_audio_files(train)
    rng = _random.Random(1)
    pos = build_positive_segments(
        files, num_classes=3, collar_min_s=0.5, collar_max_s=1.0, rng=rng
    )
    ds = ATBFLDataset(
        segments=pos,
        num_classes=3,
        sample_rate=250,
        n_fft=256,
        hop_length=5,
    )
    items = [ds[i] for i in range(len(ds))]
    batch = collate_segments(items)
    assert batch["audio"].dim() == 3
    assert batch["audio"].size(0) == len(items)
    assert batch["frame_labels"].size(-1) == 3
    # max frame length matches the spec frame count of the longest audio
    expected_max = (batch["audio"].size(-1) - 256) // 5 + 1
    assert batch["frame_labels"].size(1) == expected_max
    assert torch.any(batch["frame_labels"] > 0)


# ------------------------------------------------------------ sampling


def test_negative_sampler_avoids_annotations(tiny_dataset: Tuple[Path, Path]) -> None:
    import random as _random

    train, _ = tiny_dataset
    files = load_audio_files(train)
    sampler = StochasticNegativeSampler(
        files, min_dur_s=1.0, max_dur_s=3.0, rng=_random.Random(0)
    )
    assert sampler.has_negatives
    segs = sampler.sample(20)
    assert len(segs) == 20
    af_by_path = {af.path: af for af in files}
    for s in segs:
        af = af_by_path[s.audio_path]
        for ann in af.annotations:
            assert not (s.start_s < ann.offset_s and ann.onset_s < s.end_s), (
                f"Negative segment {s.start_s}-{s.end_s} overlaps "
                f"annotation {ann.onset_s}-{ann.offset_s}"
            )


# ----------------------------------------------------------------- losses


def test_weighted_bce_masks_padding() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 10, 3)
    targets = torch.randint(0, 2, (2, 10, 3)).float()
    frame_lengths = torch.tensor([5, 10])
    loss = WeightedBCELoss(pos_weight=torch.tensor([1.0, 2.0, 1.0]))
    out = loss(logits, targets, frame_lengths)
    assert out.dim() == 0
    assert out.item() > 0
    # Padding regions in sample 0 should not affect loss
    logits2 = logits.clone()
    logits2[0, 5:] = 100.0  # corrupt padding region
    out2 = loss(logits2, targets, frame_lengths)
    assert torch.allclose(out, out2, atol=1e-5)


def test_focal_loss_matches_bce_at_easy_examples() -> None:
    logits = torch.tensor([[[5.0]]])
    targets = torch.tensor([[[1.0]]])
    focal = FocalLoss(alpha=0.25, gamma=2.0)(
        logits, targets, frame_lengths=torch.tensor([1])
    )
    # focal loss should be tiny for very-confident correct predictions
    assert focal.item() < 0.001


def test_compute_bce_pos_weight() -> None:
    pos_counts = torch.tensor([10.0, 5.0, 20.0])
    w = compute_bce_pos_weight(pos_counts, neg_segment_count=100)
    assert w.tolist() == [10.0, 20.0, 5.0]


def test_multi_objective_loss() -> None:
    torch.manual_seed(0)
    cls_loss = FocalLoss()
    moo = MultiObjectiveLoss(cls_loss, regression_weight=1.0)
    logits = torch.randn(2, 8, 3)
    targets = torch.zeros_like(logits)
    targets[0, 2:5, 1] = 1.0
    frame_lengths = torch.tensor([8, 8])
    bbox_pred = torch.zeros(2, 4, 4)
    bbox_conf = torch.zeros(2, 4)
    bbox_target = torch.zeros(2, 4, 4)
    bbox_mask = torch.tensor([[True, False, False, False], [False] * 4])
    out = moo(
        logits=logits,
        targets=targets,
        frame_lengths=frame_lengths,
        bbox_pred=bbox_pred,
        bbox_conf=bbox_conf,
        bbox_target=bbox_target,
        bbox_mask=bbox_mask,
    )
    assert "loss" in out and "cls_loss" in out and "bbox_reg_loss" in out


# ----------------------------------------------------------- postprocessing


def test_median_filter_smooths_spikes() -> None:
    x = torch.zeros(20, 1)
    x[5] = 1.0  # isolated spike
    y = median_filter_1d(x.unsqueeze(0), kernel_size=5).squeeze(0)
    assert y[5].item() == 0.0  # spike removed
    # plateau preserved
    x[10:16] = 1.0
    y = median_filter_1d(x.unsqueeze(0), kernel_size=5).squeeze(0)
    assert y[12].item() == 1.0


def test_threshold_search_recovers_easy_threshold() -> None:
    torch.manual_seed(0)
    probs = torch.cat(
        [torch.rand(200, 2) * 0.4, torch.rand(200, 2) * 0.5 + 0.5], dim=0
    )
    targets = torch.cat([torch.zeros(200, 2), torch.ones(200, 2)], dim=0)
    th, metrics = find_optimal_thresholds(probs, targets, grid=51)
    # The two distributions are separable somewhere in [0.4, 0.5]; allow
    # slack for floating-point grid representation.
    assert all(0.35 <= t <= 0.55 for t in th.tolist()), th.tolist()
    for m in metrics:
        assert m.f1 > 0.9


def test_collapse_seven_to_three_preserves_max() -> None:
    probs = torch.zeros(1, 5, 7)
    probs[0, 0, 0] = 0.9  # BmA -> bmabz
    probs[0, 0, 1] = 0.1  # BmB -> bmabz
    probs[0, 0, 3] = 0.5  # BmD -> d
    probs[0, 0, 5] = 0.7  # Bp20 -> bp
    out = collapse_seven_to_three(probs)
    assert out.shape == (1, 5, 3)
    assert pytest.approx(out[0, 0, 0].item(), 1e-5) == 0.9
    assert pytest.approx(out[0, 0, 1].item(), 1e-5) == 0.5
    assert pytest.approx(out[0, 0, 2].item(), 1e-5) == 0.7


def test_probabilities_to_calls_pipeline() -> None:
    probs = torch.zeros(500, 3)
    # 1-second positive region for class 0
    probs[100:150, 0] = 0.9
    # short spurious spike for class 1 (should be removed by min-duration)
    probs[200:205, 1] = 0.9
    thresholds = torch.tensor([0.5, 0.5, 0.5])
    events = probabilities_to_calls(
        probs,
        thresholds,
        hop_s=0.02,
        class_names=["bmabz", "d", "bp"],
        median_kernel_ms=200,
        merge_gap_s=0.5,
        min_duration_s=0.5,
        max_duration_s=30.0,
    )
    assert len(events) == 1
    assert events[0].label == "bmabz"
    assert pytest.approx(events[0].onset_s, abs=0.05) == 2.0
    assert pytest.approx(events[0].offset_s, abs=0.05) == 3.0


# -------------------------------------------------------------- trainer


def test_trainer_runs_one_epoch(tiny_dataset: Tuple[Path, Path]) -> None:
    train, val = tiny_dataset
    cfg = TrainingConfig(
        train_root=str(train),
        val_root=str(val),
        num_classes=3,
        loss_type="focal",
        epochs=1,
        batch_size=2,
        num_workers=0,
        collar_min_s=0.5,
        collar_max_s=1.0,
        neg_min_dur_s=1.0,
        neg_max_dur_s=3.0,
        eval_segment_s=10.0,
        eval_overlap_s=2.0,
        pos_to_neg_ratio=1.0,
        device="cpu",
        output_dir=str(train.parent / "runs"),
        log_interval=1,
        pin_memory=False,
        threshold_search_grid=11,
    )
    trainer = Trainer(cfg)
    result = trainer.fit()
    assert "history" in result
    assert len(result["history"]) == 1
    val_metrics = result["history"][0]["val"]
    assert math.isfinite(val_metrics["bce_loss"])
    # The selection checkpoint should exist
    assert (Path(cfg.output_dir) / "latest.pt").exists()
    assert (Path(cfg.output_dir) / "best.pt").exists()


def test_trainer_with_bounding_boxes(tiny_dataset: Tuple[Path, Path]) -> None:
    train, val = tiny_dataset
    cfg = TrainingConfig(
        train_root=str(train),
        val_root=str(val),
        num_classes=3,
        loss_type="focal",
        epochs=1,
        batch_size=2,
        num_workers=0,
        collar_min_s=0.5,
        collar_max_s=1.0,
        neg_min_dur_s=1.0,
        neg_max_dur_s=3.0,
        eval_segment_s=10.0,
        eval_overlap_s=2.0,
        pos_to_neg_ratio=1.0,
        include_bounding_boxes=True,
        device="cpu",
        output_dir=str(train.parent / "runs_bbox"),
        log_interval=1,
        pin_memory=False,
        threshold_search_grid=11,
    )
    trainer = Trainer(cfg)
    result = trainer.fit()
    assert len(result["history"]) == 1
