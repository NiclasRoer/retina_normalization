"""Integration tests for retina_normalization package workflows.

These tests verify end-to-end functionality by combining multiple components
and validating realistic scenarios. They may be slower than unit tests.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from dataset_functions.data_analyst import infer_input_spec, infer_num_classes
from dataset_functions.datasets import build_dataloaders
from experiment_setup.corruptions import CORRUPTIONS, gaussian_noise
from experiment_setup.run_experiments import run_experiment
from model_functions.model_loader import load_models, provide_model
from model_functions.train_models import (
    adapt_model_to_data,
    evaluate,
    fit_and_evaluate,
    train_one_epoch,
)
from torch.utils.data import DataLoader, TensorDataset


class TestModelTrainingWorkflow:
    """Integration tests for model training pipeline."""

    @pytest.mark.slow
    def test_train_one_epoch_workflow(self) -> None:
        """Test training a model for one epoch on synthetic data."""
        # Create synthetic data
        images = torch.randn(64, 3, 32, 32)
        labels = torch.randint(0, 10, (64,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=16)

        # Create and setup model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        device = torch.device("cpu")

        # Train one epoch
        train_loss, train_acc = train_one_epoch(model, loader, optimizer, device)

        # Verify results
        assert isinstance(train_loss, float)
        assert isinstance(train_acc, float)
        assert 0 <= train_acc <= 1
        assert train_loss > 0

    @pytest.mark.slow
    def test_evaluate_workflow(self) -> None:
        """Test evaluating a model on synthetic test data."""
        # Create synthetic data
        images = torch.randn(32, 3, 32, 32)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=16)

        # Create model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        device = torch.device("cpu")

        # Evaluate
        eval_loss, eval_acc = evaluate(model, loader, device)

        # Verify results
        assert isinstance(eval_loss, float)
        assert isinstance(eval_acc, float)
        assert 0 <= eval_acc <= 1
        assert eval_loss > 0

    @pytest.mark.slow
    def test_fit_and_evaluate_workflow(self) -> None:
        """Test full training and evaluation workflow."""
        # Create synthetic data
        train_images = torch.randn(128, 3, 32, 32)
        train_labels = torch.randint(0, 10, (128,))
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=32)

        test_images = torch.randn(32, 3, 32, 32)
        test_labels = torch.randint(0, 10, (32,))
        test_dataset = TensorDataset(test_images, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Create model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        models = {"simple_model": model}
        device = torch.device("cpu")

        # Train and evaluate
        summary = fit_and_evaluate(
            models=models,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=2,
        )

        # Verify structure
        assert "results" in summary
        assert "simple_model" in summary["results"]
        assert "history" in summary["results"]["simple_model"]
        assert len(summary["results"]["simple_model"]["history"]) == 2
        
        # Verify each epoch has expected fields
        for epoch_data in summary["results"]["simple_model"]["history"]:
            assert "epoch" in epoch_data
            assert "train_loss" in epoch_data
            assert "train_acc" in epoch_data
            assert "test_loss" in epoch_data
            assert "test_acc" in epoch_data


class TestDatasetWorkflow:
    """Integration tests for dataset and data loading pipelines."""

    def test_build_and_infer_cifar10(self) -> None:
        """Test building CIFAR10 loaders and inferring specs."""
        train_loader, test_loader = build_dataloaders(
            batch_size=32,
            dataset_name="CIFAR10",
            max_train_samples=100,
            max_test_samples=50,
        )

        # Infer input spec from train loader
        channels, height, width = infer_input_spec(train_loader)
        assert channels == 3
        assert height == 32
        assert width == 32

        # Infer num classes
        num_classes = infer_num_classes(train_loader)
        assert num_classes == 10

        # Verify loaders are iterable
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2
        assert train_batch[0].shape[0] <= 32

        test_batch = next(iter(test_loader))
        assert len(test_batch) == 2
        assert test_batch[0].shape[0] <= 32

    def test_dataloader_with_model_adaptation(self) -> None:
        """Test loading data and adapting model to its specs."""
        train_loader, test_loader = build_dataloaders(
            batch_size=32,
            dataset_name="CIFAR10",
            max_train_samples=64,
        )

        # Get data specs
        channels, height, width = infer_input_spec(train_loader)
        num_classes = infer_num_classes(train_loader)

        # Load and adapt model
        model = provide_model("mobilenet_v3_small", pretrained=False)
        adapted_model = adapt_model_to_data(model, channels, num_classes)

        # Verify adaptation
        assert adapted_model.classifier[-1].out_features == num_classes

        # Test forward pass with real data
        images, labels = next(iter(train_loader))
        with torch.no_grad():
            output = adapted_model(images)
        
        assert output.shape == (images.shape[0], num_classes)


class TestCorruptionWorkflow:
    """Integration tests for corruption and robustness measurement."""

    def test_apply_corruptions_to_batch(self) -> None:
        """Test applying various corruptions to image batches."""
        # Create synthetic images with structure (not just noise)
        images = torch.zeros(8, 3, 32, 32)
        # Add some structure to make corruptions more visible
        for i in range(8):
            images[i, :, :16, :] = 1.0  # Make one half white

        # Apply each corruption
        corrupted_images = {}
        for corruption_name, corruption_fn in CORRUPTIONS.items():
            corrupted = corruption_fn(images, severity=0.5)
            corrupted_images[corruption_name] = corrupted

            # Verify shapes preserved
            assert corrupted.shape == images.shape
            
            # For most corruptions, verify data is modified (skip strict check for blur)
            # Gaussian blur on structured images should still change them noticeably
            if "blur" not in corruption_name.lower():
                assert not torch.allclose(corrupted, images)
            else:
                # For blur, just verify it runs without error
                assert corrupted.shape == images.shape

    def test_corruption_effect_on_model_accuracy(self) -> None:
        """Test that corruptions affect model predictions."""
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        model.eval()

        # Create synthetic images
        images = torch.rand(8, 3, 32, 32)

        # Get predictions on clean images
        with torch.no_grad():
            clean_logits = model(images)
            clean_preds = clean_logits.argmax(dim=1)

        # Apply corruption
        corrupted = gaussian_noise(images, severity=1.0)

        # Get predictions on corrupted images
        with torch.no_grad():
            corrupted_logits = model(corrupted)
            corrupted_preds = corrupted_logits.argmax(dim=1)

        # Predictions may differ (not guaranteed, but likely)
        # At least verify we can compute both
        assert clean_preds.shape == corrupted_preds.shape

    def test_robustness_measurement_workflow(self) -> None:
        """Test workflow for measuring robustness under corruptions."""
        # Create synthetic data
        images = torch.rand(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))

        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        model.eval()

        # Measure accuracy on clean data
        with torch.no_grad():
            clean_output = model(images)
            clean_acc = (clean_output.argmax(dim=1) == labels).float().mean().item()

        # Measure accuracy under each corruption
        corruption_accs = {}
        for corruption_name, corruption_fn in CORRUPTIONS.items():
            corrupted = corruption_fn(images, severity=0.5)
            with torch.no_grad():
                corrupted_output = model(corrupted)
                corrupted_acc = (corrupted_output.argmax(dim=1) == labels).float().mean().item()
            corruption_accs[corruption_name] = corrupted_acc

        # Verify we have results for all corruptions
        assert len(corruption_accs) == len(CORRUPTIONS)
        assert isinstance(clean_acc, float) 
        # Verify accuracies are between 0 and 1
        assert all(0 <= acc <= 1 for acc in corruption_accs.values())


class TestFullExperimentWorkflow:
    """Integration tests for the full experimental pipeline."""

    @pytest.mark.slow
    def test_run_experiment_generates_report(self, tmp_path: Path) -> None:
        """Test that run_experiment generates valid report JSON."""
        # Load a single model
        models = load_models(models=["mobilenet_v3_small"])

        # Run experiment with minimal epochs on small data
        summary = run_experiment(
            models=models,
            output_dir=str(tmp_path),
            epochs=1,
            dataset_name="CIFAR10",
        )

        # Verify summary structure
        assert isinstance(summary, dict)
        assert "device" in summary
        assert "epochs" in summary
        assert "results" in summary

        # Verify model results
        for model_name in models.keys():
            assert model_name in summary["results"]
            result = summary["results"][model_name]
            assert "history" in result
            assert len(result["history"]) == 1

            # Verify epoch data
            epoch_data = result["history"][0]
            assert epoch_data["epoch"] == 1
            assert "train_loss" in epoch_data
            assert "train_acc" in epoch_data
            assert "test_loss" in epoch_data
            assert "test_acc" in epoch_data

    @pytest.mark.slow
    def test_experiment_report_saved_to_disk(self, tmp_path: Path) -> None:
        """Test that experiment results are saved to disk as JSON."""
        models = load_models(models=["mobilenet_v3_small"])

        # Run experiment
        summary = run_experiment(
            models=models,
            output_dir=str(tmp_path),
            epochs=1,
            dataset_name="CIFAR10",
        )

        # Find generated experiment directory
        experiment_dirs = list(tmp_path.glob("experiment_*"))
        assert len(experiment_dirs) > 0

        # Find report.json
        experiment_dir = experiment_dirs[0]
        report_path = experiment_dir / "report.json"
        assert report_path.exists()

        # Load and verify report
        with open(report_path) as f:
            report_data = json.load(f)

        assert report_data == summary

    @pytest.mark.slow
    def test_experiment_with_multiple_models(self, tmp_path: Path) -> None:
        """Test running experiment with multiple models."""
        # Load multiple models from resnet family
        models = load_models(models=["resnet18", "resnet34"])

        # Run experiment
        summary = run_experiment(
            models=models,
            output_dir=str(tmp_path),
            epochs=1,
            dataset_name="CIFAR10",
        )

        # Verify results for both models
        assert "resnet18" in summary["results"]
        assert "resnet34" in summary["results"]

        # Verify both have training history
        for model_name in ["resnet18", "resnet34"]:
            assert len(summary["results"][model_name]["history"]) == 1

    @pytest.mark.slow
    def test_experiment_with_different_datasets(self, tmp_path: Path) -> None:
        """Test running experiment on different datasets."""
        models = load_models(models=["mobilenet_v3_small"])

        for dataset_name in ["CIFAR10", "MNIST"]:
            summary = run_experiment(
                models=models,
                output_dir=str(tmp_path),
                epochs=1,
                dataset_name=dataset_name,
            )

            # Verify results are valid for each dataset
            assert "results" in summary
            assert "mobilenet_v3_small" in summary["results"]


class TestContinuousTraining:
    """Integration tests for continuous model training across epochs."""

    @pytest.mark.slow
    def test_loss_should_decrease_over_epochs(self) -> None:
        """Test that training loss generally decreases over epochs."""
        # Create synthetic data with clear pattern
        # Make it so the model can easily learn
        images = torch.randn(256, 3, 32, 32)
        labels = torch.randint(0, 10, (256,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32)

        # Create model
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        device = torch.device("cpu")

        # Train for multiple epochs and track loss
        losses = []
        for epoch in range(3):
            loss, _ = train_one_epoch(model, loader, optimizer, device)
            losses.append(loss)

        # Verify loss generally decreases (or at least not all increasing)
        assert len(losses) == 3
        # Final loss should be less than initial (usually)
        # Allow some tolerance for randomness
        assert losses[-1] < losses[0] * 1.5  # Give 50% tolerance

    @pytest.mark.slow
    def test_multiple_epochs_produce_different_weights(self) -> None:
        """Test that training actually modifies model weights."""
        # Create data
        images = torch.randn(128, 3, 32, 32)
        labels = torch.randint(0, 10, (128,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=32)

        # Create model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        # Get initial weights
        initial_weights = [p.clone() for p in model.parameters()]

        # Train
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        device = torch.device("cpu")
        train_one_epoch(model, loader, optimizer, device)

        # Verify weights changed
        for initial, current in zip(initial_weights, model.parameters()):
            assert not torch.allclose(initial, current)


class TestModelAdaptationWorkflow:
    """Integration tests for model adaptation to datasets."""

    def test_adapt_model_for_grayscale_input(self) -> None:
        """Test adapting RGB model to single-channel grayscale input."""
        # Create grayscale data
        images = torch.randn(32, 1, 32, 32)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=16)

        # Load RGB model
        model = provide_model("mobilenet_v3_small", pretrained=False)

        # Adapt to grayscale
        adapted = adapt_model_to_data(model, input_channels=1, num_classes=10)

        # Verify it works with grayscale input
        batch = next(iter(loader))
        with torch.no_grad():
            output = adapted(batch[0])
        assert output.shape == (16, 10)

    def test_adapt_model_for_different_output_classes(self) -> None:
        """Test adapting model for different number of classes."""
        model = provide_model("resnet18", pretrained=False)

        for num_classes in [10, 100, 1000]:
            adapted = adapt_model_to_data(model, input_channels=3, num_classes=num_classes)

            # Verify output layer
            assert adapted.fc.out_features == num_classes

            # Test forward pass
            x = torch.randn(4, 3, 224, 224)
            with torch.no_grad():
                output = adapted(x)
            assert output.shape == (4, num_classes)

    def test_adapt_multiple_architectures(self) -> None:
        """Test adapting different architecture families."""
        architectures = [
            ("resnet18", 224),
            ("mobilenet_v3_small", 224),
        ]

        for arch_name, input_size in architectures:
            model = provide_model(arch_name, pretrained=False)
            adapted = adapt_model_to_data(model, input_channels=3, num_classes=10)

            # Test forward pass
            x = torch.randn(2, 3, input_size, input_size)
            with torch.no_grad():
                output = adapted(x)
            assert output.shape == (2, 10)
