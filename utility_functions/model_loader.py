import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models

import logging

from utility_functions.constants import TORCHVISION_MODEL_CATALOG

def discover_model_names(family: str):

    family_key = family.lower()
    names = TORCHVISION_MODEL_CATALOG.get(family_key, [])
    return sorted(set(names))

def provide_model(model_name: str, weights=None):
    # Best available weights (currently alias for IMAGENET1K_V2)
    # Note that these weights may change across versions
    model = model_builder(model_name=model_name, weights=weights)

    return model

def model_builder(model_name, weights):
    model_build = getattr(torchvision.models, model_name, None)
    model = model_build(weights=weights)
    return model

def list_all_available_models():

    all_names = [name for name in dir(torchvision.models) if not name.startswith("_") and not 'Weights' in name and not 'weights' in name]
    logging.info(f"{len(all_names)} models available in torchvision.models")
    logging.info(f"All available models in torchvision.models: {all_names}")


def _build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.35),
        nn.Linear(128, num_classes),
    )


def reshape_fc_layer(model, num_classes=10, freeze_backbone=True):

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = _build_classifier_head(in_features, num_classes)

    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = _build_classifier_head(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = _build_classifier_head(in_features, num_classes)

    else:
        raise ValueError("Model does not expose a standard final layer like 'fc' or 'classifier'.")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model, 'fc'):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            else:
                for param in model.classifier.parameters():
                    param.requires_grad = True

    return model

def get_model_shape(model, layer_breakdown=False):
    logging.info(f"Model Summary:")
    logging.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if layer_breakdown:
        for name, layer in model.named_modules():
            if name == "":
                continue
            if hasattr(layer, 'weight'):
                logging.info(f"  Layer: {name}, weights shape: {layer.weight.shape}, requires_grad: {layer.weight.requires_grad}")


def retrain_model(model: nn.Module, data_root=".", epochs=5, batch_size=32, lr: float = 3e-4, device: str | None = None,) -> nn.Module:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([

        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):

        # train step
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # validation step
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        logging.info(f"Epoch {epoch+1}/{epochs} | loss={running_loss / len(train_loader):.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return model