"""Run the benchmark entry point for the Retina normalization project."""

from experiment_setup.run_experiments import run_experiment
from model_functions.model_loader import load_models
from utility_functions.parser import parse_args


def main() -> None:
    """Run the benchmark workflow for the selected models."""
    args = parse_args()

    models = args.models if args.models else None
    loaded_models = load_models(models=models)

    run_experiment(models=loaded_models, output_dir='./reports', epochs=20)


if __name__ == "__main__":
    main()