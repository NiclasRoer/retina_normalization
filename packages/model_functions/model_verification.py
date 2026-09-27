"""Verify model architectures against the project's CIFAR-10 training workflow."""

from __future__ import annotations

import gc
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

VERIFIED_MODELS_PATH = Path(__file__).resolve().parents[2] / "verified_models.json"
USE_CASE = {
    "dataset": "CIFAR10",
    "input_shape": [2, 3, 32, 32],
    "num_classes": 10,
    "checks": ["model construction", "adaptation", "training forward", "backward"],
}


def _create_torchvision_model(model_name: str) -> nn.Module:
    """Construct torchvision architectures without implicit pretrained weights."""
    import torchvision.models

    constructor = torchvision.models.get_model_builder(model_name)
    parameters = inspect.signature(constructor).parameters
    kwargs: dict[str, Any] = {}
    if "weights" in parameters:
        kwargs["weights"] = None
    if "weights_backbone" in parameters:
        kwargs["weights_backbone"] = None
    return torchvision.models.get_model(model_name, **kwargs)


def _check_model(factory: Callable[[], nn.Module]) -> None:
    """Raise if a model cannot be adapted and trained for the CIFAR-10 workflow."""
    from model_functions.train_models import adapt_model_to_data

    model = factory()
    if not isinstance(model, nn.Module):
        raise TypeError("constructor did not return a torch.nn.Module")

    model = adapt_model_to_data(model, input_channels=3, num_classes=10)
    model.train()
    images = torch.zeros(USE_CASE["input_shape"])
    labels = torch.zeros(USE_CASE["input_shape"][0], dtype=torch.long)
    output = model(images)
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"training forward returned {type(output).__name__}, not Tensor"
        )
    if output.shape != (len(labels), USE_CASE["num_classes"]):
        raise ValueError(
            f"expected output shape {(len(labels), USE_CASE['num_classes'])}, "
            f"got {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all():
        raise ValueError("training forward returned non-finite logits")
    nn.functional.cross_entropy(output, labels).backward()
    if not any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    ):
        raise ValueError("training backward produced no finite trainable gradients")


def verify_models(
    timm_model_names: list[str] | None = None,
    output_path: Path | str = VERIFIED_MODELS_PATH,
) -> dict[str, Any]:
    """Verify built-in torchvision/custom models and optional selected timm models.

    Model checks disable pretrained weights, so verification does not require
    network access. Optional timm models are supplied explicitly to avoid
    attempting to construct the entire timm catalog.
    """
    import timm
    import torchvision.models

    from model_functions.model_loader import ExampleModel

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    verified: dict[str, list[str]] = {
        "torchvision": [],
        "timm": [],
        "custom": [],
    }
    rejected: dict[str, dict[str, str]] = {
        "torchvision": {},
        "timm": {},
        "custom": {},
    }

    candidates: dict[str, list[tuple[str, Callable[[], nn.Module]]]] = {
        "torchvision": [
            (
                model_name,
                lambda model_name=model_name: _create_torchvision_model(model_name),
            )
            for model_name in torchvision.models.list_models()
        ],
        "timm": [
            (
                model_name,
                lambda model_name=model_name: timm.create_model(
                    model_name, pretrained=False
                ),
            )
            for model_name in dict.fromkeys(timm_model_names or [])
        ],
        "custom": [("ExampleModel", ExampleModel)],
    }

    try:
        for source, source_candidates in candidates.items():
            for index, (model_name, factory) in enumerate(source_candidates, start=1):
                print(
                    f"Verifying {source} model {index}/{len(source_candidates)}: "
                    f"{model_name}"
                )
                try:
                    _check_model(factory)
                    verified[source].append(model_name)
                except Exception as error:
                    rejected[source][model_name] = f"{type(error).__name__}: {error}"[
                        :500
                    ]
                    print(f"  Rejected: {rejected[source][model_name]}")
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        torch.set_num_threads(previous_threads)

    result: dict[str, Any] = {
        "schema_version": 1,
        "use_case": USE_CASE,
        "torchvision": sorted(verified["torchvision"]),
        "timm": sorted(verified["timm"]),
        "custom": sorted(verified["custom"]),
        "rejected": rejected,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    temporary_path.replace(path)
    print(
        "Verified models saved to "
        f"{path}: torchvision={len(result['torchvision'])}, "
        f"timm={len(result['timm'])}, custom={len(result['custom'])}"
    )
    return result


def load_verified_models(path: Path | str = VERIFIED_MODELS_PATH) -> dict[str, Any]:
    """Read and validate the generated verified-model manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("schema_version") != 1 or any(
        not isinstance(result.get(source), list)
        for source in ("torchvision", "timm", "custom")
    ):
        raise ValueError(f"Invalid verified model manifest: {path}")
    return result


def ensure_verified_models() -> dict[str, Any]:
    """Generate the manifest once if it does not exist, then load it."""
    if not VERIFIED_MODELS_PATH.exists():
        print("Verified-model manifest not found; running model verification once.")
        verify_models(output_path=VERIFIED_MODELS_PATH)
    return load_verified_models(VERIFIED_MODELS_PATH)
