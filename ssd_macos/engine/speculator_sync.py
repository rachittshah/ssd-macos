"""Synchronous speculative decoding for SSD-macOS.

Standard speculative decoding where draft and target run sequentially
on the same device. The draft model generates K candidate tokens
autoregressively, then the target model verifies all K tokens in one
forward pass.
"""
import mlx.core as mx

from ssd_macos.models.wrapper import ModelWrapper


class SpeculatorSync:
    """Synchronous speculative decoding.

    Algorithm:
    1. Draft model generates K candidate tokens autoregressively
    2. Target model verifies all K tokens in one forward pass
    3. Accept longest matching prefix + sample one new token from target
    """

    def __init__(self, draft_runner: ModelWrapper, lookahead: int = 3):
        self.draft_runner = draft_runner
        self.lookahead = lookahead

    def speculate(
        self,
        input_ids: mx.array,
        draft_cache: list | None = None,
    ) -> tuple[mx.array, list]:
        """Generate K draft tokens autoregressively.

        Args:
            input_ids: last accepted token [1, 1]
            draft_cache: KV cache for the draft model

        Returns:
            (draft_token_ids [1, K], draft_cache)
        """
        draft_tokens = []
        token = input_ids

        for _ in range(self.lookahead):
            logits = self.draft_runner.forward(token, cache=draft_cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            draft_tokens.append(next_token)
            token = next_token

        return mx.concatenate(draft_tokens, axis=-1), draft_cache
