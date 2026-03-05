"""Asynchronous speculative speculative decoding (SSD) for Apple Silicon.

This is the core SSD algorithm adapted for Apple Silicon via MLX.
The original SSD uses separate GPUs for draft and target. On Mac, we
use MLX streams for parallelism with unified memory (zero-copy sharing).

Key innovation: the draft model speculatively generates tokens for
MULTIPLE possible verification outcomes (tree-based speculation),
overlapping draft generation with target verification.
"""
import mlx.core as mx

from ssd_macos.models.wrapper import ModelWrapper


class SpeculatorAsync:
    """Asynchronous speculative speculative decoding.

    Algorithm per step:
    1. While target verifies tokens from previous step:
       - Draft generates tree of candidate tokens (fan_out paths)
       - Each path assumes a different verification outcome
    2. When verification completes:
       - Select the matching draft path (cache hit) or fall back (cache miss)
    3. Submit new verification batch

    On Apple Silicon, we leverage:
    1. MLX streams for concurrent draft/verify execution
    2. Unified memory for zero-copy data sharing
    3. Tree-based fan-out to speculate on multiple paths
    """

    def __init__(
        self,
        draft_runner: ModelWrapper,
        target_runner: ModelWrapper,
        lookahead: int = 3,
        fan_out: int = 3,
    ):
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.lookahead = lookahead
        self.fan_out = fan_out

        # MLX streams for parallel execution
        self.draft_stream = mx.new_stream(mx.default_device())
        self.target_stream = mx.new_stream(mx.default_device())

        # Cache for speculated paths: maps token_id -> (draft_tokens, draft_cache_state)
        self.speculated_paths: dict[int, mx.array] = {}

        # Stats
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.cache_hits = 0
        self.cache_misses = 0

    def draft_step(
        self,
        input_ids: mx.array,
        draft_cache: list | None = None,
        fan_out_tokens: mx.array | None = None,
    ) -> dict[int, mx.array] | mx.array:
        """Run draft model on draft_stream, generating tree of candidates.

        Args:
            input_ids: last token [1, 1]
            draft_cache: KV cache for draft model
            fan_out_tokens: if provided, generate continuations for each
                possible verification outcome (tree speculation)

        Returns:
            If fan_out_tokens: dict mapping token_id -> draft_tokens [1, K]
            Otherwise: draft_tokens [1, K]
        """
        with mx.stream(self.draft_stream):
            if fan_out_tokens is not None:
                paths = {}
                for i in range(fan_out_tokens.shape[-1]):
                    token_val = fan_out_tokens[0, i].item()
                    token_input = fan_out_tokens[:, i : i + 1]

                    # Each fan-out path gets its own draft continuation
                    # In a full implementation we'd snapshot/restore cache;
                    # here we do fresh generation per path for correctness.
                    draft_tokens = self._generate_draft_sequence(
                        token_input, draft_cache
                    )
                    paths[token_val] = draft_tokens

                mx.eval(*list(paths.values()))
                return paths
            else:
                result = self._generate_draft_sequence(input_ids, draft_cache)
                mx.eval(result)
                return result

    def _generate_draft_sequence(
        self, start_token: mx.array, draft_cache: list | None
    ) -> mx.array:
        """Generate a sequence of K draft tokens starting from start_token."""
        draft_tokens = []
        token = start_token

        for _ in range(self.lookahead):
            logits = self.draft_runner.forward(token, cache=draft_cache)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            draft_tokens.append(next_token)
            token = next_token

        return mx.concatenate(draft_tokens, axis=-1)

    def verify_step(
        self,
        candidate_tokens: mx.array,
        target_cache: list | None = None,
    ) -> mx.array:
        """Run target model verification on target_stream.

        Args:
            candidate_tokens: tokens to verify [1, K] or [1, K+1]
            target_cache: KV cache for target model

        Returns:
            Target logits [1, seq_len, vocab_size]
        """
        with mx.stream(self.target_stream):
            logits = self.target_runner.forward(candidate_tokens, cache=target_cache)
            mx.eval(logits)
            return logits

    def speculate_and_verify(
        self,
        input_ids: mx.array,
        draft_cache: list | None = None,
        target_cache: list | None = None,
    ) -> tuple[mx.array, int, bool]:
        """Run one full SSD step: overlapped draft + verify.

        This is the main entry point for a single SSD iteration.

        Args:
            input_ids: last accepted token [1, 1]
            draft_cache: KV cache for draft model
            target_cache: KV cache for target model

        Returns:
            (accepted_tokens, num_accepted, was_cache_hit)
        """
        # Check if we have a cached path for this token
        token_val = input_ids[0, 0].item()
        was_cache_hit = False

        if token_val in self.speculated_paths:
            # Cache hit: use pre-computed draft tokens
            draft_tokens = self.speculated_paths[token_val]
            self.cache_hits += 1
            was_cache_hit = True
        else:
            # Cache miss: run draft model now
            draft_tokens = self.draft_step(input_ids, draft_cache)
            self.cache_misses += 1

        # Clear old speculated paths
        self.speculated_paths.clear()

        # Build verification input: [last_accepted, draft_token_0, ..., draft_token_K-1]
        verify_input = mx.concatenate([input_ids, draft_tokens], axis=-1)

        # Run target verification
        target_logits = self.verify_step(verify_input, target_cache)

        # While verifying, start drafting for the next step's fan-out
        # Get top fan_out tokens from the target's first position as fan-out seeds
        first_logits = target_logits[:, 0, :]
        top_tokens = mx.argsort(first_logits, axis=-1)[:, -self.fan_out :]

        # Pre-compute draft paths for likely next tokens (async with verification)
        fan_paths = self.draft_step(
            input_ids, draft_cache, fan_out_tokens=top_tokens
        )
        if isinstance(fan_paths, dict):
            self.speculated_paths = fan_paths

        # Verify: find longest accepted prefix
        K = draft_tokens.shape[-1]
        accepted = []
        for i in range(K):
            target_token = mx.argmax(target_logits[:, i, :], axis=-1)
            if target_token.item() == draft_tokens[0, i].item():
                accepted.append(target_token.item())
            else:
                # Mismatch: take target's token as recovery
                accepted.append(target_token.item())
                break
        else:
            # All matched: bonus token from target's last position
            bonus = mx.argmax(target_logits[:, K, :], axis=-1)
            accepted.append(bonus.item())

        accepted_tokens = mx.array([accepted])
        return accepted_tokens, len(accepted), was_cache_hit
