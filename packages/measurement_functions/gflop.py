"""Utilities for estimating model FLOPs."""

import torch
from fvcore.nn import FlopCountAnalysis
from torch import nn


def measure_gflop(model: nn.Module) -> None:
    """Measure and print the GFLOPs for a model.

    Args:
        model: The PyTorch model to profile.

    Returns:
        None: The function prints the FLOP count and does not return a value.

    Raises:
        None: This function does not raise custom exceptions.
    """
    model.eval()

    inputs = torch.randn(1, 3, 224, 224)

    flop_counter = FlopCountAnalysis(model, inputs)
    gflops = flop_counter.total() / 1e9
    print(f"Total FLOPs: {gflops:.3f} GFLOPs")
