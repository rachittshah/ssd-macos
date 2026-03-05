"""Inference step logic for SSD-macOS engine."""
import mlx.core as mx

from ssd_macos.engine.model_runner import ModelRunner
from ssd_macos.engine.sequence import Sequence
from ssd_macos.engine.verifier import Verifier
from ssd_macos.layers.sampler import sample


class AutoRegressiveStep:
    """Standard autoregressive generation step."""

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner

    def prefill(self, seq: Sequence, cache) -> int:
        """Run prefill on a sequence's prompt tokens.

        Returns the first generated token id.
        """
        input_ids = mx.array(seq.prompt_token_ids)
        logits = self.model_runner.prefill(input_ids, cache=cache)
        token_id = sample(
            logits[None],
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item()
        seq.append_token(token_id)
        return token_id

    def decode(self, seq: Sequence, cache) -> int:
        """Run one decode step on a sequence.

        Returns the next generated token id.
        """
        input_ids = mx.array([seq.get_last_token_id()])
        logits = self.model_runner.decode(input_ids, cache=cache)
        token_id = sample(
            logits[None],
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item()
        seq.append_token(token_id)
        return token_id


class SpecDecodeStep:
    """Synchronous speculative decoding step.

    Uses a draft model to generate K candidates, then verifies them
    all at once with the target model.
    """

    def __init__(
        self,
        draft_runner: ModelRunner,
        target_runner: ModelRunner,
        verifier: Verifier,
        lookahead: int = 3,
    ):
        self.draft_runner = draft_runner
        self.target_runner = target_runner
        self.verifier = verifier
        self.lookahead = lookahead

    def prefill(self, seq: Sequence, draft_cache, target_cache) -> int:
        """Run prefill on both draft and target models."""
        input_ids = mx.array(seq.prompt_token_ids)

        # Prefill both models
        target_logits = self.target_runner.prefill(input_ids, cache=target_cache)
        self.draft_runner.prefill(input_ids, cache=draft_cache)

        token_id = sample(
            target_logits[None],
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item()
        seq.append_token(token_id)
        return token_id

    def decode(self, seq: Sequence, draft_cache, target_cache) -> int:
        """Run one speculative decode step.

        Returns the number of tokens accepted.
        """
        last_token = mx.array([[seq.get_last_token_id()]])

        # 1. Draft: generate K candidates
        draft_tokens = []
        token = last_token
        for _ in range(self.lookahead):
            logits = self.draft_runner.model(token, cache=draft_cache)
            mx.eval(logits)
            next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            draft_tokens.append(next_token)
            token = next_token

        draft_ids = mx.concatenate(draft_tokens, axis=-1)

        # 2. Verify: run target on [last_token, draft_tokens]
        verify_input = mx.concatenate([last_token, draft_ids], axis=-1)
        target_logits = self.target_runner.model(verify_input, cache=target_cache)
        mx.eval(target_logits)

        # 3. Accept valid prefix + recovery token
        accepted, num_accepted = self.verifier.verify(
            draft_ids,
            target_logits,
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        )

        # 4. Append accepted tokens to sequence
        for i in range(num_accepted):
            seq.append_token(accepted[0, i].item())
            if seq.is_finished:
                break

        # 5. Roll back caches for rejected positions.
        # Draft cache advanced by K positions (one per draft token).
        # Target cache advanced by K+1 positions (last_token + K draft tokens).
        # We accepted (num_accepted - 1) draft tokens + 1 recovery = num_accepted total.
        # So target needs to keep 1 (last_token) + num_accepted - 1 (matched draft) + 1 (recovery pos) = num_accepted + 1
        # But actually target already has the recovery token position cached which is correct.
        # We need to trim: target advanced K+1, should keep num_accepted, so trim (K+1 - num_accepted)
        target_extra = (self.lookahead + 1) - num_accepted
        if target_extra > 0:
            for layer_cache in target_cache:
                layer_cache.trim(target_extra)

        # Draft cache advanced K positions. Matched = num_accepted - 1. Trim the rest.
        draft_extra = self.lookahead - (num_accepted - 1)
        if draft_extra > 0:
            for layer_cache in draft_cache:
                layer_cache.trim(draft_extra)

        return num_accepted


class SSDStep:
    """Asynchronous SSD inference step.

    Uses overlapped draft/verify with tree-based speculation
    for maximum throughput on Apple Silicon via MLX streams.
    """

    def __init__(
        self,
        draft_runner: ModelRunner,
        target_runner: ModelRunner,
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

        # Pre-computed draft paths: token_id -> draft_tokens
        self.speculated_paths: dict[int, mx.array] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def prefill(self, seq: Sequence, draft_cache, target_cache) -> int:
        """Run prefill on both draft and target models."""
        input_ids = mx.array(seq.prompt_token_ids)

        target_logits = self.target_runner.prefill(input_ids, cache=target_cache)
        self.draft_runner.prefill(input_ids, cache=draft_cache)

        token_id = sample(
            target_logits[None],
            temperature=seq.sampling_params.temperature,
            top_p=seq.sampling_params.top_p,
            top_k=seq.sampling_params.top_k,
        ).item()
        seq.append_token(token_id)
        return token_id

    def decode(self, seq: Sequence, draft_cache, target_cache) -> int:
        """Run one async SSD decode step.

        Overlaps draft generation with target verification using MLX streams.
        Uses tree-based fan-out to pre-compute draft paths for likely next tokens.
        """
        last_token = mx.array([[seq.get_last_token_id()]])
        token_val = seq.get_last_token_id()

        # Check for cached draft path from previous step's fan-out
        if token_val in self.speculated_paths:
            draft_ids = self.speculated_paths[token_val]
            self.cache_hits += 1
            # On cache hit, the draft cache wasn't advanced for this path
            # (fan-out paths are speculative), so we need to advance it now
            # by feeding the draft tokens through
            with mx.stream(self.draft_stream):
                # Feed last_token + matched draft path through draft model
                feed = mx.concatenate([last_token, draft_ids], axis=-1)
                self.draft_runner.model(feed, cache=draft_cache)
                mx.eval(mx.zeros(1))
        else:
            # Cache miss: generate draft tokens now
            with mx.stream(self.draft_stream):
                draft_tokens = []
                token = last_token
                for _ in range(self.lookahead):
                    logits = self.draft_runner.model(token, cache=draft_cache)
                    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                    draft_tokens.append(next_token)
                    token = next_token
                draft_ids = mx.concatenate(draft_tokens, axis=-1)
                mx.eval(draft_ids)
            self.cache_misses += 1

        self.speculated_paths.clear()

        # Verify with target model (on target stream)
        verify_input = mx.concatenate([last_token, draft_ids], axis=-1)
        with mx.stream(self.target_stream):
            target_logits = self.target_runner.model(verify_input, cache=target_cache)
            mx.eval(target_logits)

        # Verify draft tokens
        K = draft_ids.shape[-1]
        accepted = []
        num_matched = 0
        for i in range(K):
            target_token = mx.argmax(target_logits[:, i, :], axis=-1)
            mx.eval(target_token)
            if target_token.item() == draft_ids[0, i].item():
                accepted.append(target_token.item())
                num_matched += 1
            else:
                accepted.append(target_token.item())
                break
        else:
            bonus = mx.argmax(target_logits[:, K, :], axis=-1)
            mx.eval(bonus)
            accepted.append(bonus.item())

        num_accepted = len(accepted)

        # Trim caches for rejected positions
        target_extra = (K + 1) - num_accepted
        if target_extra > 0:
            for layer_cache in target_cache:
                layer_cache.trim(target_extra)

        draft_extra = K - num_matched
        if draft_extra > 0:
            for layer_cache in draft_cache:
                layer_cache.trim(draft_extra)

        # Pre-compute fan-out paths for next step using the LAST accepted
        # token's logits from target. Fan-out uses top-k likely next tokens
        # and generates draft continuations for each.
        # We get the logits at the position of the last accepted token.
        last_accepted_pos = num_accepted - 1
        fan_logits = target_logits[:, last_accepted_pos, :]
        top_k_tokens = mx.argsort(fan_logits, axis=-1)[:, -self.fan_out:]
        mx.eval(top_k_tokens)

        # Save draft cache offset so we can restore after fan-out
        draft_offset = draft_cache[0].offset if draft_cache else 0

        with mx.stream(self.draft_stream):
            for i in range(self.fan_out):
                tok = top_k_tokens[0, i].item()
                tok_input = top_k_tokens[:, i:i+1]
                path_tokens = []
                t = tok_input

                # Generate draft continuation without modifying the main cache
                # We use a temporary approach: generate without cache context
                # for the fan-out tokens (the first draft token has context
                # from the fan seed, subsequent ones chain from draft)
                # This is a simplification — full impl would snapshot the cache
                temp_logits = self.draft_runner.model(t, cache=None)
                nt = mx.argmax(temp_logits[:, -1, :], axis=-1, keepdims=True)
                path_tokens.append(nt)
                for _ in range(self.lookahead - 1):
                    temp_logits = self.draft_runner.model(nt, cache=None)
                    nt = mx.argmax(temp_logits[:, -1, :], axis=-1, keepdims=True)
                    path_tokens.append(nt)

                self.speculated_paths[tok] = mx.concatenate(path_tokens, axis=-1)

            if self.speculated_paths:
                mx.eval(*list(self.speculated_paths.values()))

        # Append accepted tokens
        for tok_id in accepted:
            seq.append_token(tok_id)
            if seq.is_finished:
                break

        return num_accepted
