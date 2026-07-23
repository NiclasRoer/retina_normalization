import torch
import numpy as np
import logging

from utility_functions.parser import parse_args
from utility_functions.model_loader import discover_model_names, provide_model, list_all_available_models, reshape_fc_layer, retrain_model, get_model_shape

def main():
    args = parse_args()

    models = args.models if args.models else []

    if args.models:
        logging.info(f"Models to benchmark: {args.models}")
    if args.families:
        logging.info(f"Architecture families to inspect/evaluate: {args.families}")

    # families = ["mobilenetv2", "mobilenetv3", "mobileone", "resnet", "regnet", "resnest", "regnest", "efficientnet", "efficientnetv2", "repvgg", "nfnet", "convnext", "convnextv2", "relknet", "hgnetv2"]
    # for family in families:
    #     print(f"Models discovered for specified family {family}:")
    #     print(f"    {discover_model_names(family)}\n")

    for model_name in models:
        model = provide_model(model_name)

        get_model_shape(model, layer_breakdown=True)
        reshape_fc_layer(model, freeze_backbone=True)
        get_model_shape(model, layer_breakdown=True)
        # logging.info(f"Training {model_name}...")
        # retrain_model(model)

        model.eval()

        img = np.random.rand(1, 3, 224, 224).astype(np.float32)

        with torch.no_grad():
            output = model(torch.from_numpy(img))
            logging.info(output.shape, output.argmax(dim=1))
        # https://docs.pytorch.org/vision/stable/models.html

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    # list_all_available_models()