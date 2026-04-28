import torch
import torch.nn as nn
from torchvision import models
import config_brain as config

def get_model(model_name, freeze_fraction=0.0):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, config.NUM_CLASSES)
        )
        backbone_layers = [name for name, _ in model.named_parameters()
                           if not name.startswith("fc")]

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, config.NUM_CLASSES)
        )
        backbone_layers = [name for name, _ in model.named_parameters()
                           if not name.startswith("classifier")]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    _apply_freeze(model, backbone_layers, freeze_fraction)
    return model.to(config.DEVICE)

def _apply_freeze(model, backbone_layers, freeze_fraction):
    if freeze_fraction == 0.0:
        return
    n_to_freeze = int(len(backbone_layers) * freeze_fraction)
    frozen_names = set(backbone_layers[:n_to_freeze])
    for name, param in model.named_parameters():
        if name in frozen_names:
            param.requires_grad = False

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable