"""Training and report-generation workflow for model experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dataset_functions.data_analyst import infer_input_spec, infer_num_classes
from dataset_functions.datasets import build_dataloaders
from model_functions.train_models import adapt_model_to_data, fit_and_evaluate
from utility_functions.visualize_training import plot_report


def run_experiment(
    models: dict[str, nn.Module],
    output_dir: str | None = None,
    epochs: int = 1,
    dataset_name: str = "CIFAR10",
) -> dict[str, Any]:
    """Train models, evaluate them, and optionally save a JSON report.

    Args:
        models: Models to train, keyed by their display names.
        output_dir: Optional directory for the generated report.
        epochs: Number of training epochs per model.
        dataset_name: Name of the dataset class in torchvision.datasets.

    Returns:
        A dictionary containing device, epoch count, and metric histories.
    """
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # train_loader, test_loader = build_dataloaders(batch_size=16, dataset_name=dataset_name, max_train_samples=None, max_test_samples=None)
    train_loader, test_loader = build_dataloaders(batch_size=16, dataset_name=dataset_name)

    input_channels, _, _ = infer_input_spec(train_loader)
    num_classes = infer_num_classes(train_loader)

    models = {
        name: adapt_model_to_data(
            model, input_channels=input_channels, num_classes=num_classes
        ).to(device)
        for name, model in models.items()
    }

    summary = fit_and_evaluate(models=models, train_loader=train_loader, test_loader=test_loader, device=device, epochs=epochs)

    if output_dir is not None:
        experiment_dir = Path(output_dir) / (
            f"experiment_{datetime.now().strftime('%m%d_%H%M')}"
        )
        experiment_dir.mkdir(parents=True, exist_ok=True)
        report_path = experiment_dir / "report.json"
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        plot_report(summary, experiment_dir)

    return summary


if __name__=='__main__':
    print('Running experiment...')
    report = run_experiment(output_dir='./reports', epochs=20)