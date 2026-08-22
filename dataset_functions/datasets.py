from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

class LocalMNIST(Dataset):
    """Simple local MNIST dataset reader that uses the downloaded gzip files."""

    def __init__(self, data_root: Path, train: bool, transform: Any | None = None) -> None:
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
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        label = self.labels[idx]
        image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LocalCIFAR10(Dataset):
    """Thin wrapper around torchvision's CIFAR-10 dataset for RGB pretrained models."""

    def __init__(self, data_root: Path, train: bool, transform: Any | None = None) -> None:
        self.transform = transform
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.dataset = datasets.CIFAR10(
            root=str(self.data_root),
            train=train,
            download=True,
            transform=None,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = self.dataset[idx]
        image = np.asarray(image)
        image = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32) / 255.0
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.int64)


def build_dataloaders(
        batch_size: int = 64,
        num_workers: int = 0,
        max_train_samples: int = 1024,
        max_test_samples: int = 256,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    data_root = Path(__file__).resolve().parents[1] / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    train_dataset = LocalCIFAR10(data_root=data_root, train=True, transform=transform)
    test_dataset = LocalCIFAR10(data_root=data_root, train=False, transform=transform)

    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_indices = torch.randperm(len(train_dataset))[:max_train_samples]
        train_dataset = Subset(train_dataset, train_indices.tolist())
    if max_test_samples is not None and len(test_dataset) > max_test_samples:
        test_indices = torch.randperm(len(test_dataset))[:max_test_samples]
        test_dataset = Subset(test_dataset, test_indices.tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader