"""Input corruptions used to measure robustness of the retinal front-end."""

from __future__ import annotations

from typing import Callable

import torch
from torchvision.transforms import functional as TF


def gaussian_noise(x: torch.Tensor, severity: float = 0.4) -> torch.Tensor:
    """Add zero-mean Gaussian noise scaled by severity."""
    return x + torch.randn_like(x) * severity


def gaussian_blur(x: torch.Tensor, severity: float = 0.4) -> torch.Tensor:
    """Blur with a Gaussian kernel whose sigma is severity."""
    kernel_size = int(2 * round(severity) + 1)
    return TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[severity, severity])


def brightness_shift(x: torch.Tensor, severity: float = 0.4) -> torch.Tensor:
    """Shift overall brightness by an additive constant."""
    return x + severity


def contrast_shift(x: torch.Tensor, severity: float = 0.4) -> torch.Tensor:
    """Scale contrast around each image's mean; severity < 1 reduces contrast."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    return (x - mean) * severity + mean


CorruptionFn = Callable[[torch.Tensor], torch.Tensor]

# Default corruption suite used for robustness scoring ("clean" excluded).
CORRUPTIONS: dict[str, CorruptionFn] = {
    "gaussian_noise": gaussian_noise,
    "guassian_blur": gaussian_blur,
    "brightness": brightness_shift,
    "contrast": contrast_shift,
}