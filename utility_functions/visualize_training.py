"""Create plots from a training report JSON file."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_report(report_path: Path) -> dict[str, Any]:
    """Load a training report from JSON."""
    with report_path.open(encoding="utf-8") as report_file:
        return json.load(report_file)


def plot_report(report: dict[str, Any], output_dir: Path) -> None:
    """Save comparison and per-model training plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_histories = report.get("results", {})

    if not model_histories:
        raise ValueError("The report does not contain any model results.")

    comparison_figure, comparison_axes = plt.subplots(
        1, 2, figsize=(12, 5), constrained_layout=True
    )
    loss_axis, accuracy_axis = comparison_axes

    for model_name, model_result in model_histories.items():
        history = (
            model_result["history"]
            if isinstance(model_result, dict)
            else model_result
        )
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        test_loss = [entry["test_loss"] for entry in history]
        train_accuracy = [entry["train_acc"] for entry in history]
        test_accuracy = [entry["test_acc"] for entry in history]

        loss_axis.plot(epochs, test_loss, marker="o", label=model_name)
        accuracy_axis.plot(epochs, test_accuracy, marker="o", label=model_name)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].plot(epochs, train_loss, marker="o", label="Train")
        axes[0].plot(epochs, test_loss, marker="o", label="Test")
        axes[0].set_title(f"{model_name} loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        axes[1].plot(epochs, train_accuracy, marker="o", label="Train")
        axes[1].plot(epochs, test_accuracy, marker="o", label="Test")
        axes[1].set_title(f"{model_name} accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()

        figure.savefig(output_dir / f"{model_name}_metrics.png", dpi=150)
        plt.close(figure)

        if isinstance(model_result, dict) and "confusion_matrix" in model_result:
            confusion_figure, confusion_axis = plt.subplots(
                figsize=(7, 6), constrained_layout=True
            )
            image = confusion_axis.imshow(
                model_result["confusion_matrix"], cmap="Blues"
            )
            confusion_figure.colorbar(image, ax=confusion_axis)
            confusion_axis.set_title(f"{model_name} confusion matrix")
            confusion_axis.set_xlabel("Predicted label")
            confusion_axis.set_ylabel("True label")
            confusion_figure.savefig(
                output_dir / f"{model_name}_confusion_matrix.png", dpi=150
            )
            plt.close(confusion_figure)

    loss_axis.set_title("Test loss by model")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.legend()
    accuracy_axis.set_title("Test accuracy by model")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.legend()
    comparison_figure.savefig(output_dir / "training_comparison.png", dpi=150)
    plt.close(comparison_figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize training metrics from a JSON report."
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path("reports/mobilenet_mnist_report.json"),
        help="Path to the training report JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/training_visualizations"),
        help="Directory in which to save PNG figures.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the selected report and save its visualizations."""
    args = parse_args()
    plot_report(load_report(args.report), args.output_dir)
    print(f"Saved training visualizations to {args.output_dir}")


if __name__ == "__main__":
    main()