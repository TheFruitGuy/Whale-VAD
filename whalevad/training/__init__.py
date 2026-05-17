"""Training pipeline for Whale-VAD.

Implements the training procedure described in:

    Geldenhuys, C. M., Tonitz, G., & Niesler, T. R. (2025).
    Whale-VAD: Whale Vocalisation Activity Detection. DCASE 2025.

Key components:
    * Segment extraction with random collar (Section 5.1)
    * Spectrogram with phase information (Section 5.2/5.4)
    * Stochastic negative mini-batch undersampling (Section 5.5)
    * Weighted BCE and focal loss (Section 5.6)
    * Optional multi-objective bounding-box regression (Section 5.7)
    * Postprocessing: median filter, per-class thresholds, call merging (5.8)
"""

from .config import TrainingConfig
from .dataset import (
    ATBFLDataset,
    Annotation,
    AudioFile,
    Segment,
    collate_segments,
    CLASS_MAP_7,
    CLASS_MAP_3,
    SEVEN_TO_THREE,
)
from .sampling import StochasticNegativeSampler
from .losses import (
    WeightedBCELoss,
    FocalLoss,
    MultiObjectiveLoss,
    compute_bce_pos_weight,
)
from .metrics import (
    compute_frame_metrics,
    find_optimal_thresholds,
)
from .postprocessing import (
    median_filter_1d,
    probabilities_to_calls,
    collapse_seven_to_three,
)
from .trainer import Trainer


__all__ = [
    "TrainingConfig",
    "ATBFLDataset",
    "Annotation",
    "AudioFile",
    "Segment",
    "collate_segments",
    "CLASS_MAP_7",
    "CLASS_MAP_3",
    "SEVEN_TO_THREE",
    "StochasticNegativeSampler",
    "WeightedBCELoss",
    "FocalLoss",
    "MultiObjectiveLoss",
    "compute_bce_pos_weight",
    "compute_frame_metrics",
    "find_optimal_thresholds",
    "median_filter_1d",
    "probabilities_to_calls",
    "collapse_seven_to_three",
    "Trainer",
]
