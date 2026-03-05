"""Token verification for speculative decoding in SSD-macOS.

Compares draft tokens against target model predictions.
Supports both greedy and stochastic (rejection sampling) verification.
"""
import mlx.core as mx

from ssd_macos.layers.sampler import sample
from ssd_macos.models.wrapper import ModelWrapper


class Verifier:
    """Verifies draft tokens against target model.

    For each position i in the draft sequence:
    - Compare target's token at position i with draft's token at position i
    - Accept if they match (greedy) or pass sampling check (stochastic)
    - Return accepted prefix length + recovery token from target
    """

    def __init__(self, target_runner: ModelWrapper, lookahead: int = 3):
        self.target_runner = target_runner
        self.lookahead = lookahead

        # Stats
        self.total_draft = 0
        self.total_accepted = 0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_draft if self.total_draft > 0 else 0.0

    def reset_stats(self):
        self.total_draft = 0
        self.total_accepted = 0

    def verify(
        self,
        draft_tokens: mx.array,
        target_logits: mx.array,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> tuple[mx.array, int]:
        """Verify draft tokens against target logits.

        Args:
            draft_tokens: [1, K] draft token ids
            target_logits: [1, K+1, vocab] target logits for input + draft positions.
                Position i contains logits for verifying draft token i.
                Position K contains logits for the bonus token.
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling threshold

        Returns:
            (accepted_tokens [1, N], num_accepted) where accepted_tokens includes
            the recovery/bonus token from target. num_accepted is total output
            tokens (matched prefix + 1 recovery/bonus).
        """
        K = draft_tokens.shape[-1]
        accepted = []
        num_matched = 0

        self.total_draft += K

        for i in range(K):
            if temperature <= 0.0:
                target_token = mx.argmax(target_logits[:, i, :], axis=-1)
            else:
                # Stochastic verification via rejection sampling
                target_token = sample(
                    target_logits[:, i, :],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )

            target_val = target_token.item()
            draft_val = draft_tokens[0, i].item()

            if target_val == draft_val:
                accepted.append(target_val)
                num_matched += 1
            else:
                # Mismatch: accept target's token as recovery
                accepted.append(target_val)
                break
        else:
            # All K draft tokens matched: sample bonus token from position K
            if temperature <= 0.0:
                bonus = mx.argmax(target_logits[:, K, :], axis=-1)
            else:
                bonus = sample(
                    target_logits[:, K, :],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            accepted.append(bonus.item())

        self.total_accepted += num_matched

        accepted_tokens = mx.array([accepted])
        return accepted_tokens, len(accepted)

    def verify_batch(
        self,
        input_ids: mx.array,
        draft_tokens: mx.array,
        target_cache: list | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> tuple[mx.array, int]:
        """Convenience method: run target forward pass then verify.

        Concatenates input_ids with draft_tokens, runs target model,
        then verifies.

        Args:
            input_ids: last accepted token(s) [1, 1]
            draft_tokens: [1, K] draft token ids
            target_cache: KV cache for target model
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling threshold

        Returns:
            (accepted_tokens [1, N], num_accepted)
        """
        # Build verification input
        verify_input = mx.concatenate([input_ids, draft_tokens], axis=-1)

        # Run target model
        target_logits = self.target_runner.forward(verify_input, cache=target_cache)
        mx.eval(target_logits)

        # Verify starting from position 0 (which corresponds to the draft tokens)
        # target_logits[:, 0, :] verifies draft_tokens[:, 0]
        # We need logits for positions [0..K] where K = draft length
        return self.verify(
            draft_tokens,
            target_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
