"""Rotary positional embeddings for SSD-macOS."""
import mlx.nn as nn


class RotaryEmbedding(nn.RoPE):
    """Rotary position embedding wrapping mlx.nn.RoPE.

    Supports custom rope_theta and rope_scaling via constructor args.
    """

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000.0,
        scale: float = 1.0,
    ):
        super().__init__(
            dims=dims,
            traditional=traditional,
            base=base,
            scale=scale,
        )
        self.max_position_embeddings = max_position_embeddings
