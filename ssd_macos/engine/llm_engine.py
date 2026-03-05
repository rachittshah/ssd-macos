"""Core LLM engine tying together model, scheduler, and inference steps."""
import mlx.core as mx

from ssd_macos.config import Config
from ssd_macos.sampling_params import SamplingParams
from ssd_macos.engine.model_runner import ModelRunner
from ssd_macos.engine.scheduler import Scheduler
from ssd_macos.engine.sequence import Sequence, SequenceStatus
from ssd_macos.engine.step import AutoRegressiveStep


class LLMEngine:
    def __init__(self, model_path: str, **kwargs):
        self.config = Config(model=model_path, **kwargs)
        self.model_runner = ModelRunner(model_path)
        self.tokenizer = self.model_runner.tokenizer
        self.scheduler = Scheduler(self.config)
        self.step = AutoRegressiveStep(self.model_runner)

        # EOS token from tokenizer
        if self.config.eos == -1:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None:
                self.config.eos = eos

    def add_request(self, prompt: str, sampling_params: SamplingParams) -> Sequence:
        token_ids = self.tokenizer.encode(prompt)
        # Inject EOS as stop token if not already set
        if sampling_params.stop_token_ids is None and self.config.eos >= 0:
            sampling_params.stop_token_ids = [self.config.eos]
        seq = Sequence(prompt_token_ids=token_ids, sampling_params=sampling_params)
        self.scheduler.add(seq)
        return seq

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: SamplingParams | None = None,
    ) -> list[str]:
        """Generate completions for one or more prompts.

        Args:
            prompts: A single prompt string or list of prompts.
            sampling_params: Sampling parameters. Defaults to greedy.

        Returns:
            List of generated text completions.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        sequences = []
        for prompt in prompts:
            # Each sequence gets its own copy of sampling params
            sp = SamplingParams(
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                max_tokens=sampling_params.max_tokens,
                repetition_penalty=sampling_params.repetition_penalty,
                stop_token_ids=(
                    list(sampling_params.stop_token_ids)
                    if sampling_params.stop_token_ids
                    else None
                ),
            )
            sequences.append(self.add_request(prompt, sp))

        # Run inference loop
        step_count = 0
        while not self.scheduler.is_finished():
            seqs_to_run, is_prefill = self.scheduler.schedule()
            if not seqs_to_run:
                break

            for seq in seqs_to_run:
                if is_prefill:
                    cache = self.model_runner.make_cache(
                        max_kv_size=self.config.max_model_len
                    )
                    # Store cache on the sequence for decode phase
                    seq._cache = cache
                    self.step.prefill(seq, cache)
                else:
                    self.step.decode(seq, seq._cache)

            step_count += 1
            if self.config.max_steps is not None and step_count >= self.config.max_steps:
                break

        # Collect results
        outputs = []
        for seq in sequences:
            text = self.tokenizer.decode(seq.completion_token_ids)
            outputs.append(text)

        return outputs
