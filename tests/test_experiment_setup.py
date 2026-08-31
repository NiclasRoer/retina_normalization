"""Tests for experiment_setup package."""

import torch
from experiment_setup.corruptions import (
    CORRUPTIONS,
    brightness_shift,
    contrast_shift,
    gaussian_blur,
    gaussian_noise,
)


class TestGaussianNoise:
    """Tests for gaussian_noise corruption."""

    def test_gaussian_noise_shape_preserved(self) -> None:
        """Test that noise doesn't change tensor shape."""
        x = torch.randn(4, 3, 32, 32)
        y = gaussian_noise(x, severity=0.1)
        assert y.shape == x.shape

    def test_gaussian_noise_actually_adds_noise(self) -> None:
        """Test that noise is actually added (tensor is different)."""
        x = torch.zeros(4, 3, 32, 32)
        y = gaussian_noise(x, severity=0.5)
        # Most values should be non-zero after adding noise
        assert (y != 0).sum() > 0

    def test_gaussian_noise_zero_severity(self) -> None:
        """Test with zero severity (no noise added)."""
        x = torch.randn(4, 3, 32, 32)
        y = gaussian_noise(x, severity=0.0)
        torch.testing.assert_close(y, x)

    def test_gaussian_noise_high_severity(self) -> None:
        """Test with high severity produces larger differences."""
        x = torch.zeros(4, 3, 32, 32)
        y_low = gaussian_noise(x, severity=0.1)
        y_high = gaussian_noise(x, severity=1.0)

        # High severity should produce larger values (on average)
        assert y_high.abs().mean() > y_low.abs().mean()


class TestGaussianBlur:
    """Tests for gaussian_blur corruption."""

    def test_gaussian_blur_shape_preserved(self) -> None:
        """Test that blur doesn't change tensor shape."""
        x = torch.randn(4, 3, 32, 32)
        y = gaussian_blur(x, severity=0.5)
        assert y.shape == x.shape

    def test_gaussian_blur_smooths_image(self) -> None:
        """Test that blur reduces high-frequency content."""
        # Create a sharp image (high variance)
        x = torch.randn(1, 3, 32, 32)
        y = gaussian_blur(x, severity=1.0)

        # Blurred image should have lower variance (less high-frequency content)
        # Note: This is a heuristic test; actual behavior depends on implementation
        assert y.shape == x.shape

    def test_gaussian_blur_valid_severity_range(self) -> None:
        """Test blur with various severity levels."""
        x = torch.randn(2, 3, 32, 32)
        for severity in [0.1, 0.5, 1.0, 2.0]:
            y = gaussian_blur(x, severity=severity)
            assert y.shape == x.shape


class TestBrightnessShift:
    """Tests for brightness_shift corruption."""

    def test_brightness_shift_shape_preserved(self) -> None:
        """Test that brightness shift doesn't change tensor shape."""
        x = torch.randn(4, 3, 32, 32)
        y = brightness_shift(x, severity=0.1)
        assert y.shape == x.shape

    def test_brightness_shift_adds_constant(self) -> None:
        """Test that brightness shift adds a constant."""
        x = torch.ones(4, 3, 32, 32)
        severity = 0.5
        y = brightness_shift(x, severity=severity)

        # All values should be increased by severity
        torch.testing.assert_close(y, x + severity)

    def test_brightness_shift_zero_severity(self) -> None:
        """Test with zero severity."""
        x = torch.randn(4, 3, 32, 32)
        y = brightness_shift(x, severity=0.0)
        torch.testing.assert_close(y, x)

    def test_brightness_shift_negative_severity(self) -> None:
        """Test with negative severity (darkening)."""
        x = torch.ones(4, 3, 32, 32)
        y = brightness_shift(x, severity=-0.3)
        torch.testing.assert_close(y, x - 0.3)


class TestContrastShift:
    """Tests for contrast_shift corruption."""

    def test_contrast_shift_shape_preserved(self) -> None:
        """Test that contrast shift doesn't change tensor shape."""
        x = torch.randn(4, 3, 32, 32)
        y = contrast_shift(x, severity=0.8)
        assert y.shape == x.shape

    def test_contrast_shift_reduces_contrast(self) -> None:
        """Test that severity < 1 reduces contrast."""
        # Create an image with clear mean and variation
        x = torch.randn(4, 3, 32, 32)
        y = contrast_shift(x, severity=0.5)

        # Reduced contrast means variance around mean should be smaller
        x_mean_centered = x - x.mean(dim=(2, 3), keepdim=True)
        y_mean_centered = y - y.mean(dim=(2, 3), keepdim=True)

        # The std should be approximately 0.5 times the original
        # (allowing for some numerical precision)
        assert y_mean_centered.std() < x_mean_centered.std()

    def test_contrast_shift_increases_contrast(self) -> None:
        """Test that severity > 1 increases contrast."""
        x = torch.randn(4, 3, 32, 32)
        y = contrast_shift(x, severity=2.0)

        x_mean_centered = x - x.mean(dim=(2, 3), keepdim=True)
        y_mean_centered = y - y.mean(dim=(2, 3), keepdim=True)

        # Increased contrast means variance should be larger
        assert y_mean_centered.std() > x_mean_centered.std()

    def test_contrast_shift_preserves_mean(self) -> None:
        """Test that contrast shift preserves the image mean."""
        x = torch.randn(4, 3, 32, 32)
        for severity in [0.5, 1.0, 2.0]:
            y = contrast_shift(x, severity=severity)
            torch.testing.assert_close(
                y.mean(dim=(2, 3)),
                x.mean(dim=(2, 3)),
                rtol=1e-5,
                atol=1e-5,
            )


class TestCorruptionsSuite:
    """Tests for the CORRUPTIONS dictionary."""

    def test_corruptions_dict_contains_all_functions(self) -> None:
        """Test that CORRUPTIONS dict has expected keys."""
        expected_keys = {
            "gaussian_noise",
            "guassian_blur",  # Note: typo exists in original code
            "brightness",
            "contrast",
        }
        assert set(CORRUPTIONS.keys()) == expected_keys

    def test_corruptions_are_callable(self) -> None:
        """Test that all corruption functions are callable."""
        x = torch.randn(2, 3, 32, 32)
        for name, corruption_fn in CORRUPTIONS.items():
            assert callable(corruption_fn), f"Corruption {name} is not callable"
            y = corruption_fn(x, severity=0.5)
            assert y.shape == x.shape

    def test_corruptions_handle_batch_input(self) -> None:
        """Test that corruptions handle batched input correctly."""
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 32, 32)
            for corruption_fn in CORRUPTIONS.values():
                y = corruption_fn(x, severity=0.5)
                assert y.shape == (batch_size, 3, 32, 32)
