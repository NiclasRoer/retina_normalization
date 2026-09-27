"""Tests for the verified-model catalog."""

import json

import model_functions.model_verification as model_verification
import torch.nn as nn


class TinyImageClassifier(nn.Module):
    """Small CIFAR-like classifier used to exercise catalog generation."""

    def __init__(self) -> None:
        """Initialize a linear image classifier."""
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(3 * 32 * 32, 10)

    def forward(self, inputs):
        """Return ten logits for a CIFAR-shaped input batch."""
        return self.classifier(self.flatten(inputs))


def test_verify_models_writes_compatible_models(monkeypatch, tmp_path) -> None:
    """Save compatible torchvision/custom models and report rejected models."""
    import torchvision.models

    monkeypatch.setattr(torchvision.models, "list_models", lambda: ["tiny_ok"])
    monkeypatch.setattr(
        model_verification,
        "_create_torchvision_model",
        lambda model_name: TinyImageClassifier(),
    )
    destination = tmp_path / "verified_models.json"

    result = model_verification.verify_models(output_path=destination)

    assert result["torchvision"] == ["tiny_ok"]
    assert result["custom"] == ["ExampleModel"]
    assert result["timm"] == []
    assert result["use_case"]["dataset"] == "CIFAR10"
    assert json.loads(destination.read_text(encoding="utf-8")) == result


def test_ensure_verified_models_runs_verification_if_missing(
    monkeypatch, tmp_path
) -> None:
    """Create the model manifest automatically when no JSON exists."""
    destination = tmp_path / "verified_models.json"
    generated = {
        "schema_version": 1,
        "use_case": model_verification.USE_CASE,
        "torchvision": ["tiny_ok"],
        "timm": [],
        "custom": ["ExampleModel"],
        "rejected": {"torchvision": {}, "timm": {}, "custom": {}},
    }
    calls = []

    def write_manifest(output_path):
        calls.append(output_path)
        output_path.write_text(json.dumps(generated), encoding="utf-8")
        return generated

    monkeypatch.setattr(model_verification, "VERIFIED_MODELS_PATH", destination)
    monkeypatch.setattr(model_verification, "verify_models", write_manifest)

    assert model_verification.ensure_verified_models() == generated
    assert model_verification.ensure_verified_models() == generated
    assert calls == [destination]
