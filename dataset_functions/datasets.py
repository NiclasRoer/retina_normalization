"""Dataset wrappers and data loader construction helpers."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


class LocalMNIST(Dataset):
    """Simple local MNIST dataset reader that uses the downloaded gzip files."""

    def __init__(self, data_root: Path, train: bool, transform: Any | None = None) -> None:
        """Initialize the dataset and load the local MNIST files.

        Args:
            data_root: Directory containing the MNIST files.
            train: Whether to load the training split.
            transform: Optional transform applied to each image.
        """
        self.transform = transform
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        images_path, labels_path = self._resolve_paths(train)
        self.images = self._read_images(images_path)
        self.labels = self._read_labels(labels_path)

    def _resolve_paths(self, train: bool) -> tuple[Path, Path]:
        if train:
            image_name, label_name = "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"
        else:
            image_name, label_name = "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"

        direct_images = self.data_root / image_name
        direct_labels = self.data_root / label_name
        if direct_images.exists() and direct_labels.exists():
            return direct_images, direct_labels

        raw_root = self.data_root / "MNIST" / "raw"
        raw_images = raw_root / image_name
        raw_labels = raw_root / label_name
        if raw_images.exists() and raw_labels.exists():
            return raw_images, raw_labels

        from torchvision.datasets import MNIST

        MNIST(root=str(self.data_root), train=train, download=True)

        if raw_images.exists() and raw_labels.exists():
            return raw_images, raw_labels
        if direct_images.exists() and direct_labels.exists():
            return direct_images, direct_labels

        raise FileNotFoundError(
            f"MNIST data for train={train} was not found at {self.data_root} and could not be downloaded."
        )

    def _read_images(self, path: Path) -> torch.Tensor:
        import gzip

        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        magic = int.from_bytes(raw[:4], "big")
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number: {magic}")
        count = int.from_bytes(raw[4:8], "big")
        rows = int.from_bytes(raw[8:12], "big")
        cols = int.from_bytes(raw[12:16], "big")
        pixels = raw[16:]
        arr = torch.frombuffer(pixels, dtype=torch.uint8)
        arr = arr.reshape(count, rows, cols)
        return arr.clone().to(torch.float32) / 255.0

    def _read_labels(self, path: Path) -> torch.Tensor:
        import gzip

        with gzip.open(path, "rb") as handle:
            raw = handle.read()
        magic = int.from_bytes(raw[:4], "big")
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number: {magic}")
        count = int.from_bytes(raw[4:8], "big")
        labels = raw[8:8 + count]
        return torch.tensor(list(labels), dtype=torch.int64)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an image and label at the requested index."""
        image = self.images[idx]
        label = self.labels[idx]
        image = image.unsqueeze(0).repeat(3, 1, 1)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LocalDataset(Dataset):
    """Thin wrapper around a torchvision dataset stored under a local root."""

    def __init__(
            self,
            data_root: Path,
            dataset_name: str,
            train: bool,
            transform: Any | None = None,
    ) -> None:
        """Initialize a torchvision dataset split and download it when necessary.

        Args:
            data_root: Directory used to store the dataset files.
            dataset_name: Name of the dataset class in ``torchvision.datasets``.
            train: Whether to load the training split.
            transform: Optional transform applied to each image.
        """
        self.transform = transform
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)

        try:
            dataset_class = getattr(datasets, dataset_name)
        except AttributeError as error:
            raise ValueError(f"Unknown torchvision dataset: {dataset_name}") from error

        self.dataset = dataset_class(
            root=str(self.data_root),
            train=train,
            download=True,
            transform=None,
        )

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a channel-first image tensor and label at the index."""
        image, label = self.dataset[idx]
        image = np.asarray(image)
        image_tensor = torch.from_numpy(image)
        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
        else:
            image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor.to(torch.float32) / 255.0
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        return image_tensor, torch.tensor(label, dtype=torch.int64)


def build_dataloaders(
        batch_size: int = 64,
        num_workers: int = 0,
        max_train_samples: None | int = 1024,
        max_test_samples: None | int = 256,
        dataset_name: str = "CIFAR10",
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build training and test data loaders for a torchvision dataset.

    Args:
        batch_size: Number of samples in each batch.
        num_workers: Number of worker processes used by each loader.
        max_train_samples: Optional cap on training samples.
        max_test_samples: Optional cap on test samples.
        dataset_name: Name of the dataset class in torchvision.datasets.

    Returns:
        A training data loader and a test data loader.
    """
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    data_root = Path(__file__).resolve().parents[1] / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_dataset = LocalDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        train=True,
        transform=transform,
    )
    test_dataset = LocalDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        train=False,
        transform=transform,
    )

    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_indices = torch.randperm(len(train_dataset))[:max_train_samples]
        train_dataset = Subset(train_dataset, train_indices.tolist())
    if max_test_samples is not None and len(test_dataset) > max_test_samples:
        test_indices = torch.randperm(len(test_dataset))[:max_test_samples]
        test_dataset = Subset(test_dataset, test_indices.tolist())

    print(f'Models training on dataset: {dataset_name}')
    print(f'  Training device: {torch.device("cuda" if torch.cuda.is_available() else "cpu")}')
    print(f'  Training/Test Samples: {len(train_dataset)}/{len(test_dataset)}')
    print(f'  Batch Size: {batch_size}')
    print(f'  Transformations: {transform}\n')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader