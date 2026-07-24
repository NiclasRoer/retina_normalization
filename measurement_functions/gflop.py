import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis

# calflops as alternative


def measure_gflop(model: nn.Module):

    model.eval()

    # Create dummy input tensor (batch size 1, 3 channels, 224x224)
    inputs = torch.randn(1, 3, 224, 224)

    # Calculate GFLOPs
    # calflops as alternative
    flop_counter = FlopCountAnalysis(model, inputs)
    gflops = flop_counter.total() / 1e9
    print(f"Total FLOPs: {gflops:.3f} GFLOPs")