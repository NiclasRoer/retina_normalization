"""Parse command-line arguments for the benchmark entry point."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark script.

    Args:
        None: This function does not accept direct parameters.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    Raises:
        None: This function does not raise custom exceptions.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark pretrained image classification models on CIFAR-10."
    )

    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Specify the model(s) to benchmark. Can be specified multiple times for multiple models.",
    )
    parser.add_argument(
        "--family",
        action="append",
        dest="families",
        help="Architecture family to inspect/evaluate. Can be specified multiple times for multiple families.",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR10",
        help="Name of the dataset class in torchvision.datasets.",
    )
    parser.add_argument(
        "--custom-model",
        action="append",
        dest="custom_models",
        help="Custom model name to benchmark. Can be specified multiple times.",
    )

    return parser.parse_args()