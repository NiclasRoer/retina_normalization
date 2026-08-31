"""Pytest configuration and shared fixtures for retina_normalization tests."""

import sys
from pathlib import Path

import pytest

# Add packages directory to path so imports work correctly
project_root = Path(__file__).parent.parent
packages_dir = project_root / "packages"
if str(packages_dir) not in sys.path:
    sys.path.insert(0, str(packages_dir))


@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for corruption tests."""
    import torch

    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_batch():
    """Provide a sample batch of images and labels."""
    import torch

    images = torch.randn(8, 3, 224, 224)
    labels = torch.randint(0, 10, (8,))
    return images, labels
