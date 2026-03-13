from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_classes: int = 3
    input_channels: int = 3


class CustomCNNv1(nn.Module):
    """
    Lightweight CNN baseline for Phase 3.

    Architecture:
        Block 1: Conv(3->32) -> ReLU -> MaxPool
        Block 2: Conv(32->64) -> ReLU -> MaxPool
        Block 3: Conv(64->128) -> ReLU -> MaxPool
        Head   : AdaptiveAvgPool2d(1) -> Flatten -> Linear(128->256) -> ReLU
                 -> Dropout(0.5) -> Linear(256->num_classes)
    """

    def __init__(self, num_classes: int = 3, input_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

        self.apply(initialize_weights_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class CustomCNNv2(nn.Module):
    """
    Deeper VGG-style CNN baseline for Phase 3.

    Architecture:
        Block 1:
            Conv(3->32) -> BN -> ReLU
            Conv(32->32) -> BN -> ReLU -> MaxPool
        Block 2:
            Conv(32->64) -> BN -> ReLU
            Conv(64->64) -> BN -> ReLU -> MaxPool
        Block 3:
            Conv(64->128) -> BN -> ReLU
            Conv(128->128) -> BN -> ReLU -> MaxPool
        Head:
            AdaptiveAvgPool2d(1) -> Flatten -> Linear(128->512) -> ReLU
            -> Dropout(0.5) -> Linear(512->num_classes)
    """

    def __init__(self, num_classes: int = 3, input_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

        self.apply(initialize_weights_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def initialize_weights_kaiming(module: nn.Module) -> None:
    """
    Kaiming/He initialization for Conv/Linear layers suitable for ReLU networks.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(model_name: str, num_classes: int = 3, input_channels: int = 3, dropout_p: float = 0.5) -> nn.Module:
    """
    Factory for Phase 3 scratch CNN models.
    """
    name = model_name.strip().lower()

    if name == "customcnn_v1":
        return CustomCNNv1(num_classes=num_classes, input_channels=input_channels, dropout_p=dropout_p)
    if name == "customcnn_v2":
        return CustomCNNv2(num_classes=num_classes, input_channels=input_channels, dropout_p=dropout_p)

    raise ValueError(f"Unsupported model_name: {model_name}")


def list_available_models() -> Dict[str, str]:
    return {
        "customcnn_v1": "Lightweight scratch CNN baseline",
        "customcnn_v2": "Deeper VGG-style scratch CNN with BatchNorm",
    }