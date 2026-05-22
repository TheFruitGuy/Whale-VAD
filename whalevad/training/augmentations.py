"""Input regularisation experiments from Section 5.4.

The paper investigates two augmentations (SpecAugment and Gaussian
noise perturbation at 10 dB SNR) and finds both to be
counterproductive.  They are still implemented here for completeness
and disabled by default in :class:`TrainingConfig`.
"""

from __future__ import annotations


import torch
from torch import Tensor
from torch.nn import Module


class NoisePerturbation(Module):
    """Inject Gaussian noise to achieve a target audio SNR.

    The noise is generated to match the per-segment signal power,
    scaled so that ``10 * log10(P_signal / P_noise) == target_snr_db``.
    """

    def __init__(self, *, target_snr_db: float = 10.0) -> None:
        super().__init__()
        self.target_snr_db = float(target_snr_db)

    def forward(self, audio: Tensor) -> Tensor:
        if not self.training:
            return audio
        power = audio.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        snr_linear = 10 ** (self.target_snr_db / 10.0)
        noise_power = power / snr_linear
        noise = torch.randn_like(audio) * noise_power.sqrt()
        return audio + noise


class SpecAugment(Module):
    """SpecAugment (Park et al., 2019).

    Operates on the magnitude/feature spectrogram with shape
    ``(batch, [channel], time, feature)``.  The mask value is the
    spectrogram batch mean to keep the input statistics roughly
    unchanged.
    """

    def __init__(
        self,
        *,
        freq_mask: int = 16,
        time_mask: int = 32,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        super().__init__()
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.num_freq_masks = int(num_freq_masks)
        self.num_time_masks = int(num_time_masks)

    def forward(self, spec: Tensor) -> Tensor:
        if not self.training:
            return spec
        x = spec.clone()
        time_dim, feat_dim = x.size(-2), x.size(-1)
        for _ in range(self.num_freq_masks):
            if self.freq_mask <= 0:
                break
            f = int(torch.randint(0, max(1, self.freq_mask), (1,)).item())
            f0 = int(torch.randint(0, max(1, feat_dim - f), (1,)).item())
            x[..., :, f0 : f0 + f] = x.mean()
        for _ in range(self.num_time_masks):
            if self.time_mask <= 0:
                break
            t = int(torch.randint(0, max(1, self.time_mask), (1,)).item())
            t0 = int(torch.randint(0, max(1, time_dim - t), (1,)).item())
            x[..., t0 : t0 + t, :] = x.mean()
        return x
