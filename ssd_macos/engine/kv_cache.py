"""KV cache management for MLX.

Uses mlx-lm's cache pattern: concatenation-based with pre-allocated buffers.
On Apple Silicon unified memory, we don't need paged attention.
"""
from mlx_lm.models.cache import KVCache, make_prompt_cache


class CacheManager:
    """Manages KV caches for a model using mlx-lm's native cache."""

    def __init__(self, model):
        self.model = model
        self.caches: list[KVCache] | None = None

    def make_cache(self, max_kv_size: int | None = None) -> list[KVCache]:
        self.caches = make_prompt_cache(self.model, max_kv_size=max_kv_size)
        return self.caches

    def reset(self):
        if self.caches is not None:
            for cache in self.caches:
                cache.trim(cache.size())
        self.caches = None

    def get_or_create(self, max_kv_size: int | None = None) -> list[KVCache]:
        if self.caches is None:
            return self.make_cache(max_kv_size)
        return self.caches
