import argparse

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(description="Benchmark pretrained image classification models on CIFAR-10.")
    
    parser.add_argument('--model', action='append', dest='models', help='Specify the model(s) to benchmark. Can be specified multiple times for multiple models.')
    parser.add_argument('--family', action='append', dest='families', help='Architecture family to inspect/evaluate. Can be specified multiple times for multiple families.')

    return parser.parse_args()