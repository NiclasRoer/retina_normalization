"""Run the benchmark entry point for the Retina normalization project."""

import numpy as np
import torch

from utility_functions.model_loader import experimental_models, provide_model
from utility_functions.parser import parse_args


def main() -> None:
    """Run the benchmark workflow for the selected models.

    Args:
        None: This function reads its configuration from the command-line parser.

    Returns:
        None: The function prints benchmark results and does not return a value.

    Raises:
        None: This function does not raise custom exceptions.
    """
    args = parse_args()

    models = args.models if args.models else []

    if args.models:
        print(f"Models to benchmark: {args.models}")
    if args.families:
        print(f"Architecture families to inspect/evaluate: {args.families}")

    for model_name in models:
        model = provide_model(model_name)

        experimental_models()

        model.eval()

        img = np.random.rand(1, 3, 224, 224).astype(np.float32)

        with torch.no_grad():
            output = model(torch.from_numpy(img))
            print(output.shape, output.argmax(dim=1))

        print()


if __name__ == "__main__":
    main()