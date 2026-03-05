"""Embedding and LM head layers for SSD-macOS."""
import mlx.core as mx
import mlx.nn as nn


class Embedding(nn.Embedding):
    """Token embedding wrapping mlx.nn.Embedding."""
    pass


class LMHead(nn.Module):
    """Linear projection to vocabulary size (language model head)."""

    def __init__(self, hidden_size: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)
