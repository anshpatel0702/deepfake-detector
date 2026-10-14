"""EfficientNet-B0 binary classifier (REAL=0, FAKE=1) for the Deepfake and AI Content Detector."""
import torch.nn as nn
from torchvision import models


def build_model(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """EfficientNet-B0 with the classifier replaced by Dropout + Linear(1280, 1).

    The model outputs ONE logit per image. Apply sigmoid to get P(fake).
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, 1))
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Stage 1: freeze all convolutional layers, train only the new head."""
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Stage 2: unfreeze everything for fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
