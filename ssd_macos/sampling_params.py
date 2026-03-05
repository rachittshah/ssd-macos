"""Sampling parameters for generation."""
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
