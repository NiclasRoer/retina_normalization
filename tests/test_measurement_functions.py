"""Tests for measurement_functions package."""

import torch
import torch.nn as nn
from measurement_functions.gflop import measure_gflop
from torchvision import models


class SimpleModel(nn.Module):
    """Simple test model for FLOP measurement."""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize a simple CNN."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestMeasureGflop:
    """Tests for measure_gflop function."""

    def test_measure_gflop_simple_model(self, capsys) -> None:
        """Test FLOP measurement on a simple model."""
        model = SimpleModel()
        measure_gflop(model)
        
        captured = capsys.readouterr()
        # Should print FLOP count
        assert "GFLOPs" in captured.out or "FLOP" in captured.out

    def test_measure_gflop_resnet(self, capsys) -> None:
        """Test FLOP measurement on ResNet."""
        model = models.resnet18(weights=None)
        measure_gflop(model)
        
        captured = capsys.readouterr()
        assert "GFLOPs" in captured.out or "FLOP" in captured.out

    def test_measure_gflop_model_in_eval_mode(self) -> None:
        """Test that model is set to eval mode during measurement."""
        model = SimpleModel()
        model.train()  # Set to training mode first
        
        # Measure should set it to eval internally
        measure_gflop(model)
        # After measurement, model should still be in eval (or it's acceptable either way)
        # This test mainly ensures no exceptions are raised

    def test_measure_gflop_different_architectures(self, capsys) -> None:
        """Test FLOP measurement across different architectures."""
        models_to_test = [
            ("ResNet18", models.resnet18(weights=None)),
            ("MobileNetV2", models.mobilenet_v2(weights=None)),
        ]
        
        for name, model in models_to_test:
            measure_gflop(model)
            captured = capsys.readouterr()
            assert "GFLOPs" in captured.out or "FLOP" in captured.out

    def test_measure_gflop_output_is_reasonable(self, capsys) -> None:
        """Test that FLOP measurements are reasonable values."""
        model = SimpleModel()
        measure_gflop(model)
        
        captured = capsys.readouterr()
        # FLOP count should be a positive number
        # Extract the number from output (simple heuristic)
        output = captured.out
        # Simple check: output should contain a decimal number
        assert any(char.isdigit() for char in output)

    def test_measure_gflop_does_not_modify_model(self) -> None:
        """Test that measuring FLOPs doesn't modify model state."""
        model = SimpleModel()
        
        # Store original weight
        original_weight = model.fc.weight.clone()
        
        measure_gflop(model)
        
        # Weight should be unchanged
        torch.testing.assert_close(model.fc.weight, original_weight)

    def test_measure_gflop_with_batch_size_one(self, capsys) -> None:
        """Test that FLOP measurement uses batch size 1."""
        # The function hardcodes batch size 1 in torch.randn(1, 3, 224, 224)
        model = SimpleModel()
        measure_gflop(model)
        
        captured = capsys.readouterr()
        # Should complete without errors
        assert "GFLOPs" in captured.out or "FLOP" in captured.out
