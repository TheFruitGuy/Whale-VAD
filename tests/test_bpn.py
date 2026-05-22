"""Tests for the BPN-paper reproduction subpackage (whalevad.bpn)."""

from __future__ import annotations

import csv
import math
import wave
from pathlib import Path
from typing import List, Tuple

import pytest
import torch

from whalevad.bpn import (
    BPNTrainingConfig,
    DilatedDepthwiseLayer,
    PostProcessV2Config,
    WhaleVADBPN,
    hangover_filter,
    hysteresis_threshold,
    make_upgraded_depthwise_block,
    probabilities_to_calls_v2,
)
from whalevad.bpn.trainer import BPNTrainer


# --------------------------------------------------------- audio fixture


def _write_audio(path: Path, *, duration_s: float, sample_rate: int = 250) -> None:
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
    train = tmp_path / "train"
    val = tmp_path / "val"
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
    _write_audio(val / "audio" / "siteB" / "valrec.wav", duration_s=40.0)
    _write_csv(
        val / "annotations" / "siteB" / "valrec.csv",
        [
            {"onset": "10.0", "offset": "13.0", "label": "BmA",
             "low_frequency": "20", "high_frequency": "40"},
        ],
    )
    return train, val


# ------------------------------------------------------------ architecture


def test_dilated_depthwise_layer_preserves_shape() -> None:
    layer = DilatedDepthwiseLayer(channels=128, dilation_t=4, dropout=0.0)
    x = torch.randn(2, 128, 3, 50)
    y = layer(x)
    assert y.shape == x.shape


def test_upgraded_depthwise_block_residual_sums() -> None:
    block = make_upgraded_depthwise_block(channels=128, dilations=(2, 4, 8))
    x = torch.randn(2, 128, 3, 100)
    y = block(x)
    assert y.shape == x.shape


def test_whalevad_bpn_forward_shape() -> None:
    """Upgraded backbone should produce the same per-frame output shape."""
    classifier = WhaleVADBPN(num_classes=3, feat_channels=3)
    classifier.eval()
    spec = torch.randn(2, 3, 129, 200)  # (B, channels=3, freq, time)
    # WhaleVADClassifier expects (..., time, embedding_size) so transpose:
    spec_for_clf = spec.transpose(-1, -2)  # -> (B, 3, time, freq)
    logits, probs, _ = classifier(spec_for_clf)
    assert logits.shape == (2, 200, 3)
    assert probs.shape == logits.shape
    assert torch.all((probs >= 0) & (probs <= 1))


# ----------------------------------------------------------- postprocess


def test_hysteresis_threshold_holds_state() -> None:
    probs = torch.tensor([0.1, 0.6, 0.7, 0.4, 0.35, 0.2, 0.55, 0.3])
    binary = hysteresis_threshold(probs, on_t=0.5, off_t=0.3)
    # Activates at idx 1, exits when below 0.3 -> idx 5.
    # Re-activates at idx 6, stays until idx 7 drops below 0.3? 0.3 not <0.3
    expected = torch.tensor(
        [False, True, True, True, True, False, True, True]
    )
    assert torch.equal(binary, expected)


def test_hysteresis_collapses_to_plain_threshold_when_equal() -> None:
    probs = torch.tensor([0.1, 0.6, 0.6, 0.4])
    binary = hysteresis_threshold(probs, on_t=0.5, off_t=0.5)
    assert binary.tolist() == [False, True, True, False]


def test_hangover_filter_majority_vote() -> None:
    # A length-3 majority filter keeps a single spike intact only if
    # it has 2-of-3 support.
    binary = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 1, 0]).bool()
    out = hangover_filter(binary, kernel=3)
    # Centre of each window: votes/3 > 1.5 ?
    #   idx 0: replicate-pad gives [0,0,1] -> 1 vote -> False
    #   idx 1: [0,1,0] -> 1 -> False
    #   idx 2: [1,0,1] -> 2 -> True
    #   idx 3: [0,1,1] -> 2 -> True
    #   idx 4: [1,1,0] -> 2 -> True
    #   idx 5: [1,0,1] -> 2 -> True
    #   idx 6: [0,1,1] -> 2 -> True
    #   idx 7: [1,1,1] -> 3 -> True
    #   idx 8: [1,1,0] -> 2 -> True
    #   idx 9: replicate-pad -> [1,0,0] -> 1 -> False
    expected = [False, False, True, True, True, True, True, True, True, False]
    assert out.tolist() == expected


def test_probabilities_to_calls_v2_class_specific() -> None:
    T, C = 500, 3
    probs = torch.zeros(T, C)
    # Class 0: solid 1-second activation in the middle
    probs[100:150, 0] = 0.8
    # Class 1: chatter just under the on-threshold (no hysteresis enter)
    probs[200:230, 1] = 0.45
    # Class 2: long enough activation to survive duration filter
    probs[300:380, 2] = 0.9

    config = PostProcessV2Config(
        on_thresholds=(0.5, 0.5, 0.5),
        off_thresholds=(0.3, 0.3, 0.3),
        median_kernels=(None, None, None),
        hangover_kernels=(None, None, None),
        min_event_durations_s=(0.5, 0.5, 0.5),
        max_event_durations_s=(30.0, 30.0, 30.0),
        min_inter_event_s=(0.5, 0.5, 0.5),
    )
    events = probabilities_to_calls_v2(
        probs, hop_s=0.02, class_names=["bmabz", "d", "bp"], config=config
    )
    labels = sorted({e.label for e in events})
    # Class 0 and 2 fire; class 1 is below threshold the whole time.
    assert labels == ["bmabz", "bp"]


# ----------------------------------------------------------------- train


def test_bpn_trainer_smoke(tiny_dataset: Tuple[Path, Path]) -> None:
    train, val = tiny_dataset
    cfg = BPNTrainingConfig(
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
        output_dir=str(train.parent / "runs_bpn"),
        log_interval=1,
        pin_memory=False,
        threshold_search_grid=11,
        use_upgraded_depthwise=True,
        enable_bpn_module=False,
        neg_resample_every_epochs=1,
    )
    trainer = BPNTrainer(cfg)
    # Confirm the upgraded depthwise was actually wired in.
    from whalevad.bpn.model import WhaleVADBPN as _UpgradedClassifier

    assert isinstance(trainer.classifier, _UpgradedClassifier)
    result = trainer.fit()
    assert len(result["history"]) == 1
    assert (Path(cfg.output_dir) / "best.pt").exists()
