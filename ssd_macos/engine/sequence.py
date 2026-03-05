"""Sequence state management for SSD-macOS inference engine."""
from enum import Enum

from ssd_macos.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Sequence:
    _next_id = 0

    def __init__(self, prompt_token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = Sequence._next_id
        Sequence._next_id += 1
        self.prompt_token_ids = prompt_token_ids
        self.completion_token_ids: list[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_token_ids)

    @property
    def is_finished(self) -> bool:
        if self.num_completion_tokens >= self.sampling_params.max_tokens:
            return True
        if (
            self.sampling_params.stop_token_ids
            and self.completion_token_ids
            and self.completion_token_ids[-1] in self.sampling_params.stop_token_ids
        ):
            return True
        return False

    def append_token(self, token_id: int):
        self.completion_token_ids.append(token_id)
        if self.is_finished:
            self.status = SequenceStatus.FINISHED

    def get_all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.completion_token_ids

    def get_last_token_id(self) -> int:
        if self.completion_token_ids:
            return self.completion_token_ids[-1]
        return self.prompt_token_ids[-1]
