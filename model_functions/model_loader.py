"""Utilities for discovering, constructing, and inspecting torchvision models."""

import torch
import torch.nn as nn
import torchvision.models

from model_functions.constants import (
    TORCHVISION_FAMILY_CATALOG,
    TORCHVISION_MODEL_CATALOG,
)


def experimental_models() -> None:
    """Create and inspect a sample MobileOne model.

    Args:
        None: This function does not accept parameters.

    Returns:
        None: The function prints model summary information.

    Raises:
        None: This function does not raise custom exceptions.
    """
    pass
    # model = timm.create_model("mobileone_s0")
    # get_model_shape(model)


def discover_model_names(family: str) -> list[str]:
    """Discover model names for a given torchvision family.

    Args:
        family: The architecture family to look up, such as "mobilenetv2".

    Returns:
        list[str]: A sorted list of matching model names.

    Raises:
        None: This function does not raise custom exceptions.
    """
    family_key = family.lower()
    names = TORCHVISION_FAMILY_CATALOG.get(family_key, [])
    return sorted(set(names))


def provide_model(model_name: str, weights=None, pretrained: bool = True):
    """Build and return a model for the requested name.

    Args:
        model_name: The torchvision model name to load.
        weights: Optional pretrained weights to use.
        pretrained: Whether to use pretrained weights when available.

    Returns:
        nn.Module: The constructed PyTorch model.

    Raises:
        None: This function does not raise custom exceptions.
    """
    print(f"Modelname: {model_name}")
    if not weights and pretrained:
        weights = get_default_weights(model_name)
        get_model_metadata(weights)
    model = model_builder(model_name=model_name, weights=weights)

    return model


def get_default_weights(model_name: str):
    """Resolve the default pretrained weights for a model name.

    Args:
        model_name: The torchvision model name to resolve.

    Returns:
        object: The resolved pretrained weights object.

    Raises:
        None: This function does not raise custom exceptions.
    """
    weights_name: str | None = None
    for name, weights, _ in TORCHVISION_MODEL_CATALOG:
        if name == model_name:
            weights_name = weights
            print(f"Found pretrained weights called: {weights_name} ...")

    weights = getattr(torchvision.models, weights_name, None)
    return weights.DEFAULT


def model_builder(model_name: str, weights=None, pretrained: bool = True):
    """Construct a model with optional pretrained weights.

    Args:
        model_name: The torchvision model name to construct.
        weights: Optional pretrained weights to apply.
        pretrained: Whether to use pretrained weights when available.

    Returns:
        nn.Module: The constructed PyTorch model.

    Raises:
        None: This function does not raise custom exceptions.
    """
    model_build = getattr(torchvision.models, model_name, None)
    if weights is None and pretrained:
        print("Weights not provided, falling back on DEFAULT pretrained weights...")
        weights = "DEFAULT"
    elif weights and pretrained:
        print("Model build with found weights...")
    else:
        print("Model build without pretrained weights...")
    model = model_build(weights=weights)
    print("Model sucessfully build.\n")
    # get_model_shape(model, layer_breakdown=False)
    return model


def list_all_available_models() -> None:
    """Print the names of all torchvision models that are discoverable.

    Args:
        None: This function does not accept parameters.

    Returns:
        None: The function prints the discovered model names.

    Raises:
        None: This function does not raise custom exceptions.
    """
    all_names = [
        name
        for name in dir(torchvision.models)
        if not name.startswith("_") and "Weights" not in name and "weights" not in name
    ]
    print(f"{len(all_names)} models available in torchvision.models")
    print(f"All available models in torchvision.models: {all_names}")


def _build_classifier_head(in_features: int, num_classes: int) -> nn.Module:
    """Build a small classifier head for transfer learning.

    Args:
        in_features: The number of input features for the linear layer.
        num_classes: The number of output classes.

    Returns:
        nn.Module: A sequential classifier head.

    Raises:
        None: This function does not raise custom exceptions.
    """
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.35),
        nn.Linear(128, num_classes),
    )


def reshape_fc_layer(model, num_classes: int = 10, freeze_backbone: bool = True):
    """Replace the final layer of a model and optionally freeze the backbone.

    Args:
        model: The model whose final layer should be replaced.
        num_classes: The number of output classes for the replacement head.
        freeze_backbone: Whether to freeze the backbone parameters.

    Returns:
        object: The modified model.

    Raises:
        ValueError: If the model does not expose a standard final layer.
    """
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = _build_classifier_head(in_features, num_classes)

    elif hasattr(model, "classifier"):
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

        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                for param in model.classifier[-1].parameters():
                    param.requires_grad = True
            else:
                for param in model.classifier.parameters():
                    param.requires_grad = True

    return model


def get_model_shape(model, layer_breakdown: bool = False) -> None:
    """Print a summary of the model's parameter counts.

    Args:
        model: The model to summarize.
        layer_breakdown: Whether to print per-layer weight information.

    Returns:
        None: The function prints the model summary.

    Raises:
        None: This function does not raise custom exceptions.
    """
    print("Model Summary:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if layer_breakdown:
        for name, layer in model.named_modules():
            if name == "":
                continue
            if hasattr(layer, "weight"):
                print(
                    f"  Layer: {name}, weights shape: {layer.weight.shape}, "
                    f"requires_grad: {layer.weight.requires_grad}"
                )


def get_model_metadata(model_weights) -> None:
    """Print metadata information for a model weight object.

    Args:
        model_weights: The pretrained weight metadata container.

    Returns:
        None: The function prints the metadata summary.

    Raises:
        None: This function does not raise custom exceptions.
    """
    meta = model_weights.meta
    metrics = meta.get("_metrics", float("nan"))
    imagenet_metrics = metrics.get("ImageNet-1K", {})
    acc1, acc5 = imagenet_metrics["acc@1"], imagenet_metrics["acc@5"]
    flops = meta.get("_ops", float("nan"))
    print(f"Metadata:\n    GFLOPS: {flops}\n    ImageNet-1K Acc@1: {acc1},Acc@5: {acc5}")
    

def load_models(models: list[str] | None = None, families: list[str] | None = None) -> dict[str, nn.Module]:
    """Load requested torchvision models on the available compute device.

    Args:
        models: Optional model names to load.
        families: Optional catalog family names whose models should be loaded.

    Returns:
        A dictionary mapping successfully loaded model names to model instances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_models = list(models or [])

    for family in families or []:
        requested_models.extend(discover_model_names(family))

    requested_models = list(dict.fromkeys(requested_models))

    if not requested_models:
        requested_models = ["mobilenet_v3_small"]
        print("No models provided; using mobilenet_v3_small as the baseline.")
    else:
        print(f"Models to benchmark: {requested_models}")

    loaded_models = {}
    for model_name in requested_models:
        if hasattr(torchvision.models, model_name):
            loaded_models[model_name] = provide_model(model_name).to(device)
        else:
            print(f"Unknown torchvision model: '{model_name}'")

    return loaded_models