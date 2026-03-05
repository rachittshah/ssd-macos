"""Attention layer with KV cache for SSD-macOS.

Replaces the original FlashInfer/Triton-based attention with MLX's
scaled_dot_product_attention and plain array indexing for KV cache.
"""
import mlx.core as mx
import mlx.nn as nn
import math


class KVCache:
    """Paged KV cache stored as plain MLX arrays.

    The cache is pre-allocated as a flat buffer. Slot mapping (positions)
    is handled via array indexing instead of Triton kernels.
    """

    def __init__(
        self,
        num_slots: int,
        num_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float16,
    ):
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = mx.zeros((num_slots, num_heads, head_dim), dtype=dtype)
        self.values = mx.zeros((num_slots, num_heads, head_dim), dtype=dtype)

    def update(self, keys: mx.array, values: mx.array, positions: mx.array):
        """Scatter update KV cache at given positions.

        Args:
            keys: [num_tokens, num_heads, head_dim]
            values: [num_tokens, num_heads, head_dim]
            positions: [num_tokens] integer indices into the cache
        """
        self.keys[positions] = keys
        self.values[positions] = values

    def fetch(self, positions: mx.array) -> tuple[mx.array, mx.array]:
        """Gather KV entries at given positions.

        Args:
            positions: [num_tokens] integer indices

        Returns:
            (keys, values) each of shape [num_tokens, num_heads, head_dim]
        """
        return self.keys[positions], self.values[positions]

    def reset(self):
        """Zero out the cache."""
        self.keys = mx.zeros_like(self.keys)
        self.values = mx.zeros_like(self.values)


class Attention(nn.Module):
    """Multi-head attention with KV cache support.

    Supports three modes:
    - prefill: variable-length input, builds KV cache
    - decode: single query token per sequence against cached KV
    - verify: multi-query speculative decode verification
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: KVCache | None = None,
        positions: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: input tensor [batch, seq_len, hidden_size]
            mask: attention mask [batch, 1, seq_len, kv_len] or None
            cache: KV cache to read from / write to
            positions: slot positions for KV cache [batch * seq_len]

        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Update KV cache if provided
        if cache is not None and positions is not None:
            # Flatten for cache update: [B*L, num_kv_heads, head_dim]
            k_flat = k.transpose(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
            v_flat = v.transpose(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
            cache.update(k_flat, v_flat, positions)

        # GQA: repeat KV heads to match query heads
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        # Use MLX's optimized SDPA
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # output: [B, num_heads, L, head_dim] -> [B, L, hidden_size]
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def decode_with_cache(
        self,
        x: mx.array,
        cache: KVCache,
        q_positions: mx.array,
        kv_positions: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Decode mode: query against cached KV.

        Args:
            x: input tensor [batch, num_queries, hidden_size]
            cache: populated KV cache
            q_positions: positions for new queries (for cache write) [batch * num_queries]
            kv_positions: all positions to attend to [total_kv_len]
            mask: attention mask

        Returns:
            Output tensor [batch, num_queries, hidden_size]
        """
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_new = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_new = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Write new KV to cache
        k_flat = k_new.transpose(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        v_flat = v_new.transpose(0, 2, 1, 3).reshape(-1, self.num_kv_heads, self.head_dim)
        cache.update(k_flat, v_flat, q_positions)

        # Fetch all KV entries we need to attend to
        k_cached, v_cached = cache.fetch(kv_positions)
        # Reshape for attention: [B, kv_len, num_kv_heads, head_dim]
        kv_len = kv_positions.shape[0] // B if B > 0 else kv_positions.shape[0]
        k = k_cached.reshape(B, kv_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v_cached.reshape(B, kv_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # GQA
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)
