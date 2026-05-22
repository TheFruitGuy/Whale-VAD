"""BPN-paper training driver.

Subclasses :class:`whalevad.training.Trainer` and overrides only the
model-construction step so the trainer wires up the upgraded depthwise
block (and, optionally, the BPN gating head) from
:mod:`whalevad.bpn.model` rather than the DCASE baseline.

Everything else — the dataset, segment construction, loss, validation,
checkpointing, scheduler, etc. — is inherited unchanged.  All
BPN-specific hyperparameters live on :class:`BPNTrainingConfig`, which
itself inherits from :class:`TrainingConfig`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..model import WhaleVADClassifier, WhaleVADModel
from ..training.trainer import Trainer, build_spectrogram_extractor
from .config import BPNTrainingConfig
from .model import (
    BoundaryProposalNetwork,
    WhaleVADBPN,
    WhaleVADBPNWrapper,
)


class BPNTrainer(Trainer):
    """:class:`Trainer` configured for the WhaleVAD-BPN reproduction."""

    cfg: BPNTrainingConfig  # type: ignore[assignment]

    def __init__(self, cfg: BPNTrainingConfig) -> None:
        if not isinstance(cfg, BPNTrainingConfig):
            raise TypeError(
                "BPNTrainer requires a BPNTrainingConfig instance "
                f"(got {type(cfg).__name__})"
            )
        super().__init__(cfg)
        self._post_init_overrides()

    # ------------------------------------------------------------- model

    def _build_classifier_module(self) -> WhaleVADClassifier:
        """Construct the upgraded backbone according to the BPN config.

        When ``cfg.enable_bpn_module`` is True we also enable
        ``include_intermediate_features`` so the residual block exposes
        the maps the BPN needs as inputs.
        """
        cfg = self.cfg
        common = dict(
            num_classes=cfg.num_classes,
            feat_channels=cfg.feat_channels,
            include_bottleneck_layers=cfg.include_bottleneck_layers,
            include_aggregation_layers=cfg.include_aggregation_layers,
            include_bounding_boxes=cfg.include_bounding_boxes,
            num_anchors=cfg.num_anchors,
            include_intermediate_features=cfg.enable_bpn_module,
        )
        if cfg.use_upgraded_depthwise:
            return WhaleVADBPN(
                depthwise_dilations=tuple(cfg.depthwise_dilations),
                depthwise_dropout=float(cfg.depthwise_dropout),
                **common,
            )
        return WhaleVADClassifier(**common)

    def _post_init_overrides(self) -> None:
        """Rewire ``self.classifier`` / ``self.model`` after super().__init__.

        The parent :class:`Trainer` instantiates a plain
        ``WhaleVADClassifier`` directly; here we replace it with the
        upgraded backbone (and optionally wrap with the BPN head).
        """
        cfg = self.cfg
        classifier = self._build_classifier_module().to(self.device)
        if cfg.enable_bpn_module:
            # We can't know the per-head channel widths without a dummy
            # forward, so use the standard value emitted by the
            # upgraded ResidualBlock (128 channels at each level).
            head_channels = [128, 128, 128]
            bpn = BoundaryProposalNetwork(
                intermediate_channels=head_channels,
                proj_channels=128,
                bpn_channels=64,
                num_rois=cfg.bpn_num_rois,
                lstm_hidden=cfg.bpn_lstm_hidden,
                num_classes=cfg.num_classes,
                dropout=cfg.bpn_dropout,
            ).to(self.device)
            wrapper = WhaleVADBPNWrapper(classifier, bpn=bpn).to(self.device)
            self.classifier = wrapper  # type: ignore[assignment]
        else:
            self.classifier = classifier  # type: ignore[assignment]

        # Reuse the existing spectrogram extractor — only the
        # classifier swap matters here.
        self.transform = build_spectrogram_extractor(cfg).to(self.device)
        self.model = WhaleVADModel(self.classifier, self.transform).to(self.device)

        # Rebuild the optimiser so it points at the new parameters.
        from torch.optim import AdamW

        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        # And rebuild the LR scheduler around the new optimiser.
        self.lr_scheduler = self._build_lr_scheduler()
