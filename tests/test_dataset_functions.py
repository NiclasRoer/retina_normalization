"""Tests for dataset_functions package."""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dataset_functions.data_analyst import infer_input_spec, infer_num_classes
from dataset_functions.datasets import CustomDataset, ExampleDataset, build_dataloaders


class TestDataAnalyst:
    """Tests for data analysis functions."""

    def test_infer_input_spec_4d_tensor(self) -> None:
        """Test inferring input spec from standard 4D batch."""
        batch = torch.randn(8, 3, 32, 32)  # B, C, H, W
        labels = torch.randint(0, 10, (8,))
        dataset = list(zip([batch[i] for i in range(8)], labels))
        
        loader = DataLoader(dataset, batch_size=8)
        channels, height, width = infer_input_spec(loader)
        
        assert channels == 3
        assert height == 32
        assert width == 32

    def test_infer_input_spec_3d_tensor(self) -> None:
        """Test inferring input spec from 3D tensor (adds channel dimension)."""
        # 3D tensor: H, W, C -> gets unsqueezed to C, H, W
        batch = torch.randn(8, 32, 32)
        labels = torch.randint(0, 10, (8,))
        dataset = [(batch[i].unsqueeze(0), labels[i]) for i in range(8)]
        
        loader = DataLoader(dataset, batch_size=8)
        channels, height, width = infer_input_spec(loader)
        
        assert channels == 1
        assert height == 32
        assert width == 32

    def test_infer_input_spec_empty_loader(self) -> None:
        """Test that inferring from empty loader raises ValueError."""
        empty_loader = DataLoader([], batch_size=8)
        with pytest.raises(ValueError, match="DataLoader is empty"):
            infer_input_spec(empty_loader)

    def test_infer_num_classes(self) -> None:
        """Test inferring number of classes from labels."""
        batch = torch.randn(8, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # 0-7 = 8 classes
        dataset = [(batch[i], labels[i]) for i in range(8)]
        
        loader = DataLoader(dataset, batch_size=8)
        num_classes = infer_num_classes(loader)
        
        assert num_classes == 8

    def test_infer_num_classes_empty_loader(self) -> None:
        """Test that inferring classes from empty loader raises ValueError."""
        empty_loader = DataLoader([], batch_size=8)
        with pytest.raises(ValueError, match="DataLoader is empty"):
            infer_num_classes(empty_loader)


class TestCustomDataset:
    """Tests for CustomDataset base class."""

    def test_custom_dataset_len_not_implemented(self) -> None:
        """Test that __len__ raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CustomDataset(Path(tmpdir), train=True)
            with pytest.raises(NotImplementedError, match="__len__ must be implemented"):
                len(dataset)

    def test_custom_dataset_getitem_not_implemented(self) -> None:
        """Test that __getitem__ raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = CustomDataset(Path(tmpdir), train=True)
            with pytest.raises(NotImplementedError, match="__getitem__ must be implemented"):
                dataset[0]


class TestExampleDataset:
    """Tests for ExampleDataset implementation."""

    def test_example_dataset_file_not_found(self) -> None:
        """Test that ExampleDataset raises when files are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="Example dataset split was not found"):
                ExampleDataset(Path(tmpdir), train=True)

    def test_example_dataset_with_valid_files(self) -> None:
        """Test ExampleDataset with properly formatted NPZ files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            dataset_dir = tmpdir_path / "datasets" / "example_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy NPZ files with correct format [samples, height, width, 3]
            train_data = {
                "images": np.random.randint(0, 256, (10, 32, 32, 3), dtype=np.uint8),
                "labels": np.random.randint(0, 10, 10, dtype=np.int64),
            }
            test_data = {
                "images": np.random.randint(0, 256, (5, 32, 32, 3), dtype=np.uint8),
                "labels": np.random.randint(0, 10, 5, dtype=np.int64),
            }
            
            np.savez(dataset_dir / "train.npz", **train_data)
            np.savez(dataset_dir / "test.npz", **test_data)
            
            # Test train split
            train_dataset = ExampleDataset(tmpdir_path, train=True)
            assert len(train_dataset) == 10
            image, label = train_dataset[0]
            assert image.shape == (3, 32, 32)
            assert isinstance(label, torch.Tensor)
            
            # Test test split
            test_dataset = ExampleDataset(tmpdir_path, train=False)
            assert len(test_dataset) == 5


class TestBuildDataloaders:
    """Tests for build_dataloaders function."""

    def test_build_dataloaders_cifar10(self) -> None:
        """Test building CIFAR10 dataloaders."""
        train_loader, test_loader = build_dataloaders(
            batch_size=32,
            dataset_name="CIFAR10",
            max_train_samples=100,
            max_test_samples=50,
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Verify we can iterate
        images, labels = next(iter(train_loader))
        assert images.shape[0] <= 32  # batch_size or less
        assert images.shape[1] == 3   # CIFAR10 has 3 channels
        assert images.shape[2] == 32  # 32x32 images
        assert images.shape[3] == 32

    def test_build_dataloaders_batch_size(self) -> None:
        """Test that batch size is respected."""
        batch_size = 16
        train_loader, _ = build_dataloaders(
            batch_size=batch_size,
            dataset_name="CIFAR10",
            max_train_samples=100,
        )
        
        images, labels = next(iter(train_loader))
        assert images.shape[0] == batch_size
        assert labels.shape[0] == batch_size

    def test_build_dataloaders_invalid_dataset(self) -> None:
        """Test that invalid dataset name raises appropriate error."""
        with pytest.raises((AttributeError, ValueError)):
            build_dataloaders(
                batch_size=32,
                dataset_name="NonExistentDataset",
            )
