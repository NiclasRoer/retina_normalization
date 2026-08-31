"""Helpers for inferring dataset input and label specifications."""

from typing import Any

from torch.utils.data import DataLoader


def infer_input_spec(loader: DataLoader[Any]) -> tuple[int, int, int]:
    """Infer channel count and spatial dimensions from the first batch in a loader."""
    try:
        images, _ = next(iter(loader))
    except StopIteration as exc:
        raise ValueError("DataLoader is empty; cannot infer input shape.") from exc

    if images.ndim == 2:
        images = images.unsqueeze(1).unsqueeze(-1)
    elif images.ndim == 3:
        images = images.unsqueeze(1)
    elif images.ndim != 4:
        raise ValueError(
            f"Expected a 2D/3D/4D image tensor, got shape {tuple(images.shape)}."
        )

    channels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]
    return channels, height, width


def infer_num_classes(loader: DataLoader[Any]) -> int:
    """Infer the number of classes from dataset metadata or the first batch."""
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    if classes:
        return len(classes)

    try:
        _, labels = next(iter(loader))
    except StopIteration as exc:
        raise ValueError(
            "DataLoader is empty; cannot infer number of classes."
        ) from exc

    if labels.numel() == 0:
        raise ValueError("The first batch contains no labels.")
    return int(labels.max().item()) + 1
