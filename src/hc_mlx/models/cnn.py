"""Tiny CNN for house-type classification. About 390K params at defaults."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class HouseCNN(nn.Module):
    """Four conv blocks with batch norm, then global average pool and a head."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=16, stride=16),  # 16x16 -> 1x1
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(256, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        # x is (batch, height, width, channels); MLX convs use NHWC.
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = mx.reshape(x, (x.shape[0], -1))
        return self.head(self.dropout(x))


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.parameters()))
