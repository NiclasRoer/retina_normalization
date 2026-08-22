from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dataset_functions.data_analyst import infer_input_spec, infer_num_classes
from dataset_functions.datasets import build_dataloaders
from model_functions.train_models import adapt_model_to_data, train_one_epoch, evaluate



def run_experiment(models: dict[str, nn.Module], output_dir: str | None = None, epochs: int = 1) -> dict[str, Any]:
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_dataloaders(batch_size=16)

    input_channels, _, _ = infer_input_spec(train_loader)
    num_classes = infer_num_classes(train_loader)

    models = {
                name: adapt_model_to_data(model.to(device), input_channels=input_channels, num_classes=num_classes) for name, model in models.items()
            }

    results: dict[str, list[dict[str, float]]] = {}
    for name, model in models.items():
        print(f"Training {name}...")
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            print(f"    Epoch {epoch + 1}: train_acc={train_acc:.4f} test_acc={test_acc:.4f}")
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                }
            )
        print()
        results[name] = history

    summary = {
        'device': str(device),
        'epochs': epochs,
        'results': results,
    }

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with (out_path / "mobilenet_mnist_report.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return summary


if __name__=='__main__':
    print('Running experiment...')
    report = run_experiment(output_dir='./reports', epochs=20)