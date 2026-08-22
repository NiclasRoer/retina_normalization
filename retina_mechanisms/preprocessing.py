"""Retina-inspired preprocessing modules."""

import torch.nn as nn


class RetinaPreprocessingBlock(nn.Module):
    """Preprocessing block for applying retina-inspired input normalization."""

    def __init__(self, channels: int, temportal_alpha: float = 0.2) -> None:
        """Initialize the preprocessing block.

        Args:
            channels: Number of input channels.
            temportal_alpha: Temporal adaptation factor.
        """
        super().__init__()