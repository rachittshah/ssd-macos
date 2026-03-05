"""Token sampling for SSD-macOS."""
import mlx.core as mx


def sample_greedy(logits: mx.array) -> mx.array:
    """Greedy decoding: return argmax token ids.

    Args:
        logits: shape [batch, vocab_size] or [vocab_size]

    Returns:
        Token ids with shape [batch] or scalar.
    """
    return mx.argmax(logits, axis=-1)


def sample_top_p(logits: mx.array, temperature: float = 1.0, top_p: float = 0.9) -> mx.array:
    """Top-p (nucleus) sampling.

    Args:
        logits: shape [batch, vocab_size]
        temperature: sampling temperature
        top_p: cumulative probability threshold

    Returns:
        Sampled token ids with shape [batch].
    """
    if temperature <= 0:
        return sample_greedy(logits)

    probs = mx.softmax(logits / temperature, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)
    # Reverse to descending order
    sorted_indices = sorted_indices[..., ::-1]

    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond top_p threshold
    mask = cumulative_probs - sorted_probs > top_p
    sorted_probs = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)

    # Renormalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample from categorical distribution
    sampled_sorted_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))

    # Map back to original vocab indices
    if logits.ndim == 1:
        return sorted_indices[sampled_sorted_idx]
    return mx.take_along_axis(
        sorted_indices, mx.expand_dims(sampled_sorted_idx, axis=-1), axis=-1
    ).squeeze(-1)


def sample_top_k(logits: mx.array, temperature: float = 1.0, top_k: int = 50) -> mx.array:
    """Top-k sampling.

    Args:
        logits: shape [batch, vocab_size]
        temperature: sampling temperature
        top_k: number of top tokens to consider

    Returns:
        Sampled token ids with shape [batch].
    """
    if temperature <= 0:
        return sample_greedy(logits)

    vocab_size = logits.shape[-1]
    top_k = min(top_k, vocab_size)

    sorted_indices = mx.argsort(logits, axis=-1)[..., ::-1]
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

    # Zero out everything beyond top_k
    top_k_logits = mx.where(
        mx.arange(vocab_size) < top_k,
        sorted_logits,
        mx.full_like(sorted_logits, -1e9),
    )

    probs = mx.softmax(top_k_logits / temperature, axis=-1)
    sampled_sorted_idx = mx.random.categorical(mx.log(probs + 1e-10))

    if logits.ndim == 1:
        return sorted_indices[sampled_sorted_idx]
    return mx.take_along_axis(
        sorted_indices, mx.expand_dims(sampled_sorted_idx, axis=-1), axis=-1
    ).squeeze(-1)


def sample(
    logits: mx.array,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> mx.array:
    """Unified sampling interface.

    Args:
        logits: shape [batch, vocab_size]
        temperature: 0.0 for greedy
        top_p: nucleus sampling threshold (1.0 = disabled)
        top_k: top-k threshold (-1 = disabled)

    Returns:
        Sampled token ids.
    """
    if temperature <= 0:
        return sample_greedy(logits)
    if top_k > 0:
        return sample_top_k(logits, temperature, top_k)
    if top_p < 1.0:
        return sample_top_p(logits, temperature, top_p)
    # Pure temperature sampling
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(mx.log(probs + 1e-10))
