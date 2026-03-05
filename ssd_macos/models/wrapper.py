"""Model wrapper adapting mlx-lm models for the SSD speculative decoding engine.

mlx-lm models have a __call__(inputs, cache=None) interface. This wrapper
provides a uniform API for our scheduler and speculative decoding engine.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache


class ModelWrapper:
    """Wraps an mlx-lm model for use with the SSD engine.

    Provides:
    - forward(): run a forward pass and get logits
    - prefill(): process prompt tokens and populate KV cache
    - decode_step(): single-token decode step
    - create_cache(): create a fresh KV cache
    - reset_cache(): clear the KV cache
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def forward(
        self,
        input_ids: mx.array,
        cache: list | None = None,
    ) -> mx.array:
        """Run forward pass and return logits.

        Args:
            input_ids: token ids [batch, seq_len]
            cache: mlx-lm KV cache (list of cache objects per layer)

        Returns:
            Logits array [batch, seq_len, vocab_size]
        """
        return self.model(input_ids, cache=cache)

    def prefill(
        self,
        input_ids: mx.array,
        cache: list | None = None,
    ) -> mx.array:
        """Process prompt tokens and return logits for the last token.

        Args:
            input_ids: prompt token ids [1, seq_len]
            cache: KV cache to populate

        Returns:
            Logits for the last position [1, vocab_size]
        """
        logits = self.model(input_ids, cache=cache)
        return logits[:, -1:, :]

    def decode_step(
        self,
        token: mx.array,
        cache: list,
    ) -> mx.array:
        """Single decode step.

        Args:
            token: single token [1, 1]
            cache: populated KV cache

        Returns:
            Logits [1, 1, vocab_size]
        """
        return self.model(token, cache=cache)

    def create_cache(self) -> list:
        """Create a fresh KV cache for this model."""
        return make_prompt_cache(self.model)

    def reset_cache(self, cache: list):
        """Reset all entries in the KV cache."""
        for layer_cache in cache:
            layer_cache.reset()

    @property
    def num_layers(self) -> int:
        """Number of transformer layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return len(self.model.layers)
        raise AttributeError("Cannot determine number of layers from model structure")
