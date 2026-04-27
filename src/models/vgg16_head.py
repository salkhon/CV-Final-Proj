"""VGG16 with ImageNet pre-trained features and 11-class head for SVHN-style digits."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from src.data.svhn_11 import NUM_CLASSES


def build_vgg16_classifier(
    num_classes: int = NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """Replace final FC with num_classes. Expects 224x224 after interpolation."""
    try:
        from torchvision.models import VGG16_Weights  # type: ignore[import]

        w = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.vgg16(weights=w)
    except (ImportError, AttributeError):
        m = models.vgg16(pretrained=pretrained)  # torchvision < 0.13
    in_f = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_f, num_classes)
    return m


def vgg_param_groups(model: nn.Module, lr_head: float, lr_backbone: float):
    """Differential learning rates: lower LR on conv feature extractor."""
    backbone = list(model.features.parameters())
    head = list(model.classifier.parameters())
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]
