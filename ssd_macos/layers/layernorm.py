"""RMSNorm layer for SSD-macOS."""
import mlx.nn as nn


class RMSNorm(nn.RMSNorm):
    """RMSNorm wrapping mlx.nn.RMSNorm."""
    pass
