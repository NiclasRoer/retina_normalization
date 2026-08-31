"""Tests for model_functions package."""

import urllib.error

import pytest
import torch
import torch.nn as nn
from model_functions.model_loader import (
    CustomModel,
    discover_model_names,
    load_models,
    provide_model,
)
from model_functions.train_models import adapt_model_to_data
from torchvision import models


class TestCustomModel:
    """Tests for CustomModel base class."""

    def test_custom_model_forward_not_implemented(self) -> None:
        """Test that forward raises NotImplementedError."""
        model = CustomModel("test_model")
        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(NotImplementedError, match="forward must be implemented"):
            model(x)


class TestAdaptModelToData:
    """Tests for adapt_model_to_data function."""

    def test_adapt_model_input_channels_rgb_to_grayscale(self) -> None:
        """Test adapting model from 3 channels to 1."""
        model = models.resnet18(weights=None)
        adapted = adapt_model_to_data(model, input_channels=1, num_classes=10)

        # Model should have adapted conv1 to accept 1 channel
        assert adapted.conv1.in_channels == 1

        # Should be able to forward single-channel input
        x = torch.randn(1, 1, 224, 224)
        output = adapted(x)
        assert output.shape == (1, 10)

    def test_adapt_model_output_classes(self) -> None:
        """Test adapting model to different number of output classes."""
        model = models.resnet18(weights=None)
        adapted = adapt_model_to_data(model, input_channels=3, num_classes=100)

        # Model should have adapted fc layer for 100 classes
        assert adapted.fc.out_features == 100

        # Should output correct shape
        x = torch.randn(1, 3, 224, 224)
        output = adapted(x)
        assert output.shape == (1, 100)

    def test_adapt_model_preserves_backbone_weights(self) -> None:
        """Test that adapting preserves existing backbone weights when possible."""
        model = models.resnet18(weights=None)
        # Set a specific weight value to check
        original_conv_weight = model.conv1.weight.clone()

        adapted = adapt_model_to_data(model, input_channels=3, num_classes=10)

        # If input channels match, conv1 weights should be preserved
        torch.testing.assert_close(
            adapted.conv1.weight[:, :, :, :],
            original_conv_weight,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_adapt_model_mobilenet(self) -> None:
        """Test adapting MobileNet architecture."""
        model = models.mobilenet_v2(weights=None)
        adapted = adapt_model_to_data(model, input_channels=3, num_classes=10)

        # Verify it can process input and output correct shape
        x = torch.randn(2, 3, 224, 224)
        output = adapted(x)
        assert output.shape == (2, 10)


class TestProvideModel:
    """Tests for provide_model function."""

    def test_provide_model_resnet18(self) -> None:
        """Test loading ResNet18."""
        try:
            model = provide_model("resnet18", pretrained=False)
            assert isinstance(model, nn.Module)

            # Should be able to forward
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            assert output.shape == (1, 1000)  # Default ImageNet classes
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            pytest.skip(f"Network error while loading model: {e}")

    def test_provide_model_mobilenet_v3_small(self) -> None:
        """Test loading MobileNetV3Small."""
        model = provide_model("mobilenet_v3_small", pretrained=False)
        assert isinstance(model, nn.Module)

        # Should be able to forward
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 1000  # ImageNet classes

    def test_provide_model_with_pretrained_weights(self) -> None:
        """Test that pretrained models load with weights."""
        model = provide_model("resnet18", pretrained=True)
        # Pretrained model should have non-zero weights
        conv_weight = model.conv1.weight
        assert conv_weight.abs().sum() > 0


class TestDiscoverModelNames:
    """Tests for discover_model_names function."""

    def test_discover_model_names_resnet(self) -> None:
        """Test discovering ResNet model names."""
        resnet_models = discover_model_names("resnet")
        assert len(resnet_models) > 0
        assert any("resnet" in name.lower() for name in resnet_models)

    def test_discover_model_names_mobilenet(self) -> None:
        """Test discovering MobileNet model names."""
        mobilenet_models = discover_model_names("mobilenetv2")
        assert len(mobilenet_models) > 0
        assert "mobilenet_v2" in mobilenet_models

    def test_discover_model_names_invalid_family(self) -> None:
        """Test discovering models from invalid family returns empty list."""
        invalid_models = discover_model_names("nonexistent_family_xyz")
        # Should return empty or handle gracefully
        assert isinstance(invalid_models, list)


class TestLoadModels:
    """Tests for load_models function."""

    def test_load_models_by_name(self) -> None:
        """Test loading specific models by name."""
        loaded = load_models(models=["resnet18", "resnet50"], families=None)
        assert isinstance(loaded, dict)
        assert "resnet18" in loaded or len(loaded) > 0
        assert all(isinstance(m, nn.Module) for m in loaded.values())

    def test_load_models_by_family(self) -> None:
        """Test loading models by family."""
        loaded = load_models(models=None, families=["mobilenet"])
        assert isinstance(loaded, dict)
        assert len(loaded) > 0
        assert all(isinstance(m, nn.Module) for m in loaded.values())

    def test_load_models_single_model(self) -> None:
        """Test loading a single model."""
        loaded = load_models(models=["resnet18"])
        assert isinstance(loaded, dict)
        assert "resnet18" in loaded
        assert isinstance(loaded["resnet18"], nn.Module)

    def test_load_models_empty_returns_empty(self) -> None:
        """Test that no models/families returns default or empty."""
        # This depends on implementation; verify graceful behavior
        try:
            loaded = load_models(models=None, families=None)
            assert isinstance(loaded, dict)
        except ValueError:
            # Also acceptable if it raises an error
            pass


class TestModelForwardPass:
    """Integration tests for model forward passes."""

    def test_resnet18_forward_pass(self) -> None:
        """Test ResNet18 forward pass with various batch sizes."""
        model = provide_model("resnet18", pretrained=False)
        model.eval()

        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 1000)

    def test_adapted_model_forward_pass(self) -> None:
        """Test forward pass on adapted model."""
        model = provide_model("resnet18", pretrained=False)
        adapted = adapt_model_to_data(model, input_channels=3, num_classes=10)
        adapted.eval()

        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            output = adapted(x)
        assert output.shape == (4, 10)

    def test_model_gradient_flow(self) -> None:
        """Test that gradients can flow through adapted models."""
        model = provide_model("resnet18", pretrained=False)
        adapted = adapt_model_to_data(model, input_channels=3, num_classes=10)

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = adapted(x)
        loss = output.sum()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert x.grad.shape == x.shape
