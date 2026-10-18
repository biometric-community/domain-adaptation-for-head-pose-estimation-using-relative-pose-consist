"""ResNet-18 → 3 Euler angles (paper §IV-B)."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_pose_resnet18(pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, 3)
    return net
