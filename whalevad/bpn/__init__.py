"""WhaleVAD-BPN reproduction (Geldenhuys et al., arXiv:2510.21280v2).

Parallel pipeline to ``whalevad.training`` that reproduces the
follow-up BPN paper rather than the original DCASE 2025 paper.  The
two key differences:

* :class:`WhaleVADBPN` swaps the depthwise aggregation block for a
  dilated-residual version with spatial dropout (Section V.A).
* :class:`BPNTrainingConfig` carries the actually-reported recipe
  (lr=1e-3, weight_decay=0.01, batch=48, ~32 epochs; Section V.B.5)
  rather than the DCASE paper's 1e-5/0.001/unspecified values.

The optional :class:`BoundaryProposalNetwork` adds the full gating
mechanism from Section V.B.
"""

from .config import BPNTrainingConfig
from .model import (
    BoundaryProposalNetwork,
    DilatedDepthwiseLayer,
    WhaleVADBPN,
    WhaleVADBPNWrapper,
    make_upgraded_depthwise_block,
)
from .postprocessing import (
    PostProcessV2Config,
    hangover_filter,
    hysteresis_threshold,
    probabilities_to_calls_v2,
)


__all__ = [
    "BPNTrainingConfig",
    "BoundaryProposalNetwork",
    "DilatedDepthwiseLayer",
    "WhaleVADBPN",
    "WhaleVADBPNWrapper",
    "make_upgraded_depthwise_block",
    "PostProcessV2Config",
    "hangover_filter",
    "hysteresis_threshold",
    "probabilities_to_calls_v2",
]
