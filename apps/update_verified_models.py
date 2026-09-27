"""Generate the verified model catalog used by the benchmark loader."""

import argparse

from model_functions.model_verification import verify_models


def main() -> None:
    """Verify supported model architectures and save the manifest JSON."""
    parser = argparse.ArgumentParser(
        description=(
            "Check whether models can be adapted and trained with the project's "
            "CIFAR-10 workflow, then write verified_models.json."
        )
    )
    parser.add_argument(
        "--timm-model",
        action="append",
        dest="timm_models",
        help=(
            "Also verify this timm model name. Repeat the option to check "
            "multiple selected timm models."
        ),
    )
    args = parser.parse_args()
    verify_models(timm_model_names=args.timm_models)


if __name__ == "__main__":
    main()
