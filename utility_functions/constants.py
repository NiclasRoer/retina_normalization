"""Model catalog constants used by the benchmark loader."""

TORCHVISION_MODEL_CATALOG = {
    "mobilenetv2": ["mobilenet_v2"],
    "mobilenetv3": ["mobilenet_v3_large", "mobilenet_v3_small"],
    "resnet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "resnext": ["resnext50_32x4d", "resnext101_32x8d"],
    "regnet": ["regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf"],
    "resnest": ["resnest50", "resnest101", "resnest200", "resnest269"],
    "regnest": ["regnest200", "regnest269"],
    "efficientnet": [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
    ],
    "efficientnetv2": ["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"],
    "repvgg": ["repvgg_a0", "repvgg_a1", "repvgg_a2", "repvgg_b0", "repvgg_b1", "repvgg_b2", "repvgg_b3"],
    "nfnet": ["nfnet_f0", "nfnet_f1", "nfnet_f2", "nfnet_f3", "nfnet_f4", "nfnet_f5", "nfnet_f6", "nfnet_f7"],
    "convnext": ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
    "convnextv2": ["convnextv2_tiny", "convnextv2_small", "convnextv2_base", "convnextv2_large"],
    "relknet": ["relknet_200", "relknet_300", "relknet_400", "relknet_500"],
    "hgnetv2": ["hgnetv2_tiny", "hgnetv2_small", "hgnetv2_base", "hgnetv2_large"],
}

['ConvNeXt', 'EfficientNet', 'MobileNetV2', 'MobileNetV3', 'RegNet', 'ResNet', 'ShuffleNetV2', 'SqueezeNet', 'SwinTransformer', 'VGG', 'VisionTransformer', 'alexnet', 'convnext', 'convnext_base', 'convnext_large', 'convnext_small', 'convnext_tiny', 'densenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'detection', 'efficientnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'efficientnet_v2_l', 'efficientnet_v2_m', 'efficientnet_v2_s', 'get_model', 'get_model_builder', 'get_weight', 'googlenet', 'inception', 'inception_v3', 'list_models', 'maxvit', 'maxvit_t', 'mnasnet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3', 'mobilenet', 'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small', 'mobilenetv2', 'mobilenetv3', 'optical_flow', 'quantization', 'regnet', 'regnet_x_16gf', 'regnet_x_1_6gf', 'regnet_x_32gf', 'regnet_x_3_2gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_8gf', 'regnet_y_128gf', 'regnet_y_16gf', 'regnet_y_1_6gf', 'regnet_y_32gf', 'regnet_y_3_2gf', 'regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_8gf', 'resnet', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x8d', 'resnext101_64x4d', 'resnext50_32x4d', 'segmentation', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'shufflenetv2', 'squeezenet', 'squeezenet1_0', 'squeezenet1_1', 'swin_b', 'swin_s', 'swin_t', 'swin_transformer', 'swin_v2_b', 'swin_v2_s', 'swin_v2_t', 'vgg', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'video', 'vision_transformer', 'vit_b_16', 'vit_b_32', 'vit_h_14', 'vit_l_16', 'vit_l_32', 'wide_resnet101_2', 'wide_resnet50_2']

# (model_name, weights, transforms)
CATALOG = {
    ("mobilenet_v2", "MobileNet_V2_Weights", "MobileNet_V2_Weights.IMAGENET1K_V1.transforms"),
    ("mobilenet_v3_small", "MobileNet_V3_Small_Weights", "MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms"),
    ("mobilenet_v3_large", "MobileNet_V3_Large_Weights", "MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms"),
    ("convnext_tiny", "ConvNeXt_Tiny_Weights", "ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms"),
    ("convnext_small", "ConvNeXt_Small_Weights", "ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms"),
    ("convnext_base", "ConvNeXt_Base_Weights", "ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms"),
    ("convnext_large", "ConvNeXt_Large_Weights", "ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms"),
    ("regnet_y_400mf", "RegNet_Y_400MF_Weights", "RegNet_Y_400MF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_y_800mf", "RegNet_Y_800MF_Weights", "RegNet_Y_800MF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_y_1_6gf", "RegNet_Y_1_6GF_Weights", "RegNet_Y_1_6GF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_y_3_2gf", "RegNet_Y_3_2GF_Weights", "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_x_400mf", "RegNet_X_400MF_Weights", "RegNet_X_400MF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_x_800mf", "RegNet_X_800MF_Weights", "RegNet_X_800MF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_x_1_6gf", "RegNet_X_1_6GF_Weights", "RegNet_X_1_6GF_Weights.IMAGENET1K_V2.transforms"),
    ("regnet_x_3_2gf", "RegNet_X_3_2GF_Weights", "RegNet_X_3_2GF_Weights.IMAGENET1K_V2.transforms"),
    # There are more regnets...
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
    ("", "", ""),
}