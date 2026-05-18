"""Training configuration for Whale-VAD.

Defaults follow the paper (Geldenhuys et al., 2025, DCASE 2025).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional


@dataclass
class TrainingConfig:
    # ----------------------------------------------------------------- Data
    train_root: str = "data/biodcase_development_set/train"
    val_root: str = "data/biodcase_development_set/validation"
    audio_subdir: str = "audio"
    annotation_subdir: str = "annotations"
    audio_ext: str = ".wav"
    annotation_ext: str = ".csv"

    sample_rate: int = 250

    # ------------------------------------------------------- Segmentation
    # Random collar drawn independently for both segment edges (Section 5.1).
    collar_min_s: float = 1.0
    collar_max_s: float = 5.0

    # Negative segment duration range (sampled per epoch by negative sampler,
    # Section 5.5).  Drawn uniformly from this range.
    neg_min_dur_s: float = 2.0
    neg_max_dur_s: float = 30.0

    # Eval-time tiling (Section 5.1).
    eval_segment_s: float = 30.0
    eval_overlap_s: float = 2.0

    # ---------------------------------------------------- Feature pipeline
    # STFT settings (Section 5.2).  Hop of 5 samples at 250 Hz = 20 ms,
    # n_fft of 256 samples ~ 1.024 s — both described in the paper.
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 5
    window_fn: str = "hann_window"
    complex_repr: Literal[
        "real+imag", "mag+phase", "trig", "real+imag+mag+phase", "none"
    ] = "trig"  # phase information (r, cos θ, sin θ) — Section 5.4.
    norm_features: Optional[Literal["demean"]] = "demean"
    power: Optional[float] = None  # complex spectrogram

    # ----------------------------------------------------------- Classes
    # The paper trains on 3-class collapsed labels for the best F1 (Table 3).
    num_classes: Literal[3, 7] = 3
    # When training on 7 classes we still evaluate after collapsing to 3.
    collapse_eval_to_three: bool = True

    # ----------------------------------------------------------- Model
    include_bottleneck_layers: bool = True
    include_aggregation_layers: bool = True
    include_bounding_boxes: bool = False  # Multi-objective regression (5.7)
    num_anchors: int = 64
    feat_channels: int = 3  # trig representation has 3 channels

    # ----------------------------------------------------------- Loss
    loss_type: Literal["bce", "focal"] = "focal"
    # Focal loss hyper-parameters from the original paper (Section 5.6).
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Joint multi-objective regression weight (Section 5.7).
    regression_loss_weight: float = 1.0

    # ----------------------------------------------------------- Optimiser
    learning_rate: float = 1e-5
    weight_decay: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999

    # ----------------------------------------------------------- Training
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    grad_clip: Optional[float] = 1.0
    # Approximate positive:negative balance per epoch (Section 5.5).
    pos_to_neg_ratio: float = 1.0

    # ----------------------------------------------------- Augmentation
    # The paper finds these to be counterproductive; disabled by default.
    use_specaugment: bool = False
    specaugment_freq_mask: int = 16
    specaugment_time_mask: int = 32
    specaugment_num_freq_masks: int = 2
    specaugment_num_time_masks: int = 2
    use_noise_perturbation: bool = False
    noise_snr_db: float = 10.0  # Section 5.4.

    # ----------------------------------------------------- Postprocessing
    # Section 5.8.
    median_filter_kernel_ms: float = 500.0
    min_call_duration_s: float = 0.5
    max_call_duration_s: float = 30.0
    merge_gap_s: float = 0.5

    # ----------------------------------------------------- Checkpointing
    output_dir: str = "runs/whale_vad"
    save_every_epoch: bool = False
    # Model selection is based on lowest validation BCE loss (Section 5.8).
    select_metric: Literal["bce_loss", "f1"] = "bce_loss"

    # ----------------------------------------------------- Misc
    seed: int = 42
    device: str = "cuda"
    log_interval: int = 50  # iterations
    threshold_search_grid: int = 101  # thresholds tried for per-class θ_c
    pin_memory: bool = True

    # Audio-header indexing parallelism + on-disk cache (massively speeds
    # up the dataset scan on networked filesystems).  Set
    # ``audio_index_cache_path`` to "none" to disable the cache entirely;
    # leave empty to use ``<root>/.whalevad_audio_index.json``.
    audio_index_workers: int = 16
    audio_index_cache_path: str = ""

    # Optional explicit class list ordering (otherwise inferred from
    # config.num_classes).  Useful when ground-truth CSVs use a non-default
    # naming convention.
    class_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.collar_min_s < 0 or self.collar_max_s < self.collar_min_s:
            raise ValueError("Invalid collar range")
        if self.neg_min_dur_s <= 0 or self.neg_max_dur_s < self.neg_min_dur_s:
            raise ValueError("Invalid negative duration range")
        if self.complex_repr == "none":
            self.complex_repr = None  # type: ignore[assignment]
