"""Small experiment comparing different models and their variants.

The comparison uses a pretrained and a retinal-inspired preprocessing block on CIFAR10.

The script is intentionally lightweight and educational. It trains for a few 
epochs on MNIST so the comparison can be inspected quickly.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def adapt_model_to_data(model: nn.Module, input_channels: int, num_classes: int) -> nn.Module:
    """Resize pretrained backbones to match the actual dataset input/output specification."""
    if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
        conv = model.conv1
        if conv.in_channels != input_channels:
            new_conv = nn.Conv2d(
                input_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=conv.bias is not None,
            )
            with torch.no_grad():
                if input_channels < conv.in_channels:
                    new_conv.weight[:, :input_channels] = conv.weight[:, :input_channels]
                else:
                    new_conv.weight[:, :conv.in_channels] = conv.weight
            model.conv1 = new_conv

    if hasattr(model, "features") and isinstance(model.features, nn.Sequential):
        first_layer = model.features[0]
        if isinstance(first_layer, nn.Sequential) and hasattr(first_layer[0], "in_channels"):
            conv = first_layer[0]
            if conv.in_channels != input_channels:
                new_conv = nn.Conv2d(
                    input_channels,
                    conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    bias=conv.bias is not None,
                )
                with torch.no_grad():
                    if input_channels < conv.in_channels:
                        new_conv.weight[:, :input_channels] = conv.weight[:, :input_channels]
                    else:
                        new_conv.weight[:, :conv.in_channels] = conv.weight
                first_layer[0] = new_conv

    if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
        if model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            last = classifier[-1]
            if hasattr(last, "in_features") and getattr(last, "out_features", None) != num_classes:
                classifier[-1] = nn.Linear(last.in_features, num_classes)
        elif hasattr(classifier, "in_features") and getattr(classifier, "out_features", None) != num_classes:
            model.classifier = nn.Linear(classifier.in_features, num_classes)

    return model


def train_one_epoch(model: nn.Module, loader: DataLoader[Any], optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    """Train a model for one epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loader_loop = tqdm(loader, desc='ACC: 0.0', bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', colour='blue', leave=False)
    for images, labels in loader_loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loader_loop.set_description(f"ACC: {(correct / total):.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> tuple[float, float]:
    """Evaluate a model and return mean loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_confusion_matrix(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a model and return a confusion matrix with marginal totals."""
    model.eval()
    confusion_matrix: torch.Tensor | None = None

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        num_classes = logits.shape[1]

        if confusion_matrix is None:
            confusion_matrix = torch.zeros(
                (num_classes, num_classes), dtype=torch.int64, device=device
            )

        indices = labels * num_classes + predictions
        confusion_matrix += torch.bincount(
            indices, minlength=num_classes * num_classes
        ).reshape(num_classes, num_classes)

    if confusion_matrix is None:
        raise ValueError("Cannot evaluate a model with an empty data loader.")

    matrix = confusion_matrix.cpu().tolist()
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix))]
    result: dict[str, Any] = {
        "matrix": matrix,
        "row_totals": row_totals,
        "column_totals": column_totals,
        "total": sum(row_totals),
    }
    if label_names is not None and len(label_names) == len(matrix):
        result["labels"] = label_names
    return result


def get_label_names(loader: DataLoader[Any]) -> list[str] | None:
    """Return dataset label names when the wrapped dataset exposes them."""
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    labels = getattr(dataset, "classes", None)
    return [str(label) for label in labels] if labels is not None else None


# @torch.no_grad()
# def evaluate_corrupted(
#     model: nn.Module,
#     loader: DataLoader[Any],
#     device: torch.device,
#     corruption: CorruptionFn
#     ) -> float:
#     """Return accuracy when 'corruption' is applied to each input batch."""
#     model.eval()
#     correct = 0
#     total = 0
#     for images, labels in loader:
#         images = corruption(images.to(device))
#         labels = labels.to(device)
#         preds = model(images).argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)
#     return correct / total


# @torch.no_grad()
# def evaluate_robustness(
#     model: nn.Module,
#     loader: DataLoader[Any],
#     device: torch.device,
#     corruptions: dict[str, CorruptionFn] | None = None,
#     ) -> float:
#     """Evaluate accuracy per corruption plus a mean over the suite."""
#     corruptions = CORRUPTIONS if corruptions is None else corruptions
#     per_corruption = {
#         name: evaluate_corrupted(model, loader, device, fn) for name, fn in corruptions.items()
#     }
#     per_corruption["mean"] = sum(per_corruption.values()) / len(per_corruption)
#     return per_corruption



def fit_and_evaluate(
    models: dict[str, nn.Module],
        train_loader: DataLoader[Any],
        test_loader: DataLoader[Any],
        device: torch.device,
        epochs: int=1,
        lr: float=1e-4,
) -> dict[str, Any]:
    """Train model for epochs and return per-epoch train/test metrics."""
    results: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"    Epoch {epoch + 1}: train_acc={train_acc:.4f} test_acc={test_acc:.4f} train_loss={train_loss:.4f} test_loss={test_loss:.4f} ")
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
        confusion_matrix = evaluate_confusion_matrix(
            model,
            test_loader,
            device,
            label_names=get_label_names(test_loader),
        )
        print()
        results[name] = {
            "history": history,
            "confusion_matrix": confusion_matrix,
        }

    summary = {
        'device': str(device),
        'epochs': epochs,
        'results': results,
    }
    return summary