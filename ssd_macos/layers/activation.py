"""Activation functions for SSD-macOS."""
import mlx.core as mx
import mlx.nn as nn


def silu(x: mx.array) -> mx.array:
    """SiLU (Swish) activation: x * sigmoid(x)."""
    return nn.silu(x)
