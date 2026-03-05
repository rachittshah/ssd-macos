"""MLX model execution wrapper using mlx-lm."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

from ssd_macos.layers.sampler import sample


class ModelRunner:
    """Wraps mlx-lm model loading and forward passes."""

    def __init__(self, model_path: str):
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

    def make_cache(self, max_kv_size: int | None = None) -> list[KVCache]:
        return make_prompt_cache(self.model, max_kv_size=max_kv_size)

    def prefill(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        """Run prefill on full prompt. Returns logits for last position.

        Args:
            input_ids: shape [seq_len] (1D token ids)
            cache: list of per-layer KVCache objects

        Returns:
            logits: shape [vocab_size]
        """
        # mlx-lm models expect [batch, seq_len]
        logits = self.model(input_ids[None], cache=cache)
        mx.eval(logits)
        # Return logits for last token: [vocab_size]
        return logits[0, -1, :]

    def decode(
        self,
        input_ids: mx.array,
        cache: list[KVCache] | None = None,
    ) -> mx.array:
        """Run single-token decode step. Returns logits.

        Args:
            input_ids: shape [1] (single token)
            cache: list of per-layer KVCache objects

        Returns:
            logits: shape [vocab_size]
        """
        logits = self.model(input_ids[None], cache=cache)
        mx.eval(logits)
        return logits[0, -1, :]
