"""Tests for utility_functions package."""

import sys
from unittest.mock import patch

from utility_functions.parser import parse_args


class TestParseArgs:
    """Tests for argument parser."""

    def test_parse_args_default_values(self) -> None:
        """Test default argument values."""
        with patch.object(sys, "argv", ["test_script.py"]):
            args = parse_args()
            assert args.dataset == "CIFAR10"
            assert args.models is None
            assert args.families is None
            assert args.custom_models is None

    def test_parse_args_single_model(self) -> None:
        """Test parsing single model argument."""
        with patch.object(sys, "argv", ["test_script.py", "--model", "resnet18"]):
            args = parse_args()
            assert args.models == ["resnet18"]

    def test_parse_args_multiple_models(self) -> None:
        """Test parsing multiple model arguments."""
        with patch.object(
            sys,
            "argv",
            ["test_script.py", "--model", "resnet18", "--model", "resnet50"],
        ):
            args = parse_args()
            assert args.models == ["resnet18", "resnet50"]

    def test_parse_args_single_family(self) -> None:
        """Test parsing single family argument."""
        with patch.object(sys, "argv", ["test_script.py", "--family", "resnet"]):
            args = parse_args()
            assert args.families == ["resnet"]

    def test_parse_args_multiple_families(self) -> None:
        """Test parsing multiple family arguments."""
        with patch.object(
            sys,
            "argv",
            ["test_script.py", "--family", "resnet", "--family", "mobilenet"],
        ):
            args = parse_args()
            assert args.families == ["resnet", "mobilenet"]

    def test_parse_args_custom_dataset(self) -> None:
        """Test parsing custom dataset argument."""
        with patch.object(sys, "argv", ["test_script.py", "--dataset", "ImageNet"]):
            args = parse_args()
            assert args.dataset == "ImageNet"

    def test_parse_args_single_custom_model(self) -> None:
        """Test parsing single custom model argument."""
        with patch.object(
            sys,
            "argv",
            ["test_script.py", "--custom-model", "my_custom_model"],
        ):
            args = parse_args()
            assert args.custom_models == ["my_custom_model"]

    def test_parse_args_multiple_custom_models(self) -> None:
        """Test parsing multiple custom model arguments."""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py",
                "--custom-model",
                "model1",
                "--custom-model",
                "model2",
            ],
        ):
            args = parse_args()
            assert args.custom_models == ["model1", "model2"]

    def test_parse_args_combined(self) -> None:
        """Test parsing combined arguments."""
        with patch.object(
            sys,
            "argv",
            [
                "test_script.py",
                "--model",
                "resnet18",
                "--family",
                "mobilenet",
                "--dataset",
                "CIFAR100",
                "--custom-model",
                "custom1",
            ],
        ):
            args = parse_args()
            assert args.models == ["resnet18"]
            assert args.families == ["mobilenet"]
            assert args.dataset == "CIFAR100"
            assert args.custom_models == ["custom1"]

    def test_parse_args_namespace_type(self) -> None:
        """Test that parse_args returns argparse.Namespace."""
        import argparse

        with patch.object(sys, "argv", ["test_script.py"]):
            args = parse_args()
            assert isinstance(args, argparse.Namespace)

    def test_parse_args_model_can_be_none(self) -> None:
        """Test that models field can be None when not specified."""
        with patch.object(sys, "argv", ["test_script.py"]):
            args = parse_args()
            assert args.models is None

    def test_parse_args_dataset_short_form(self) -> None:
        """Test dataset argument in shorthand."""
        with patch.object(
            sys,
            "argv",
            ["test_script.py", "--dataset", "MNIST"],
        ):
            args = parse_args()
            assert args.dataset == "MNIST"
