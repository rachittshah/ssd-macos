"""Configuration for SSD-macOS inference engine."""
import os
from dataclasses import dataclass, field
from transformers import AutoConfig
import mlx.core as mx


@dataclass
class Config:
    model: str = ""
    max_num_seqs: int = 1
    max_model_len: int = 4096
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    # spec config
    speculate: bool = False
    draft: str = ""
    draft_hf_config: AutoConfig | None = None
    speculate_k: int = 1
    draft_async: bool = False

    # async spec
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None

    # debugging
    verbose: bool = False
    debug_mode: bool = False
    max_steps: int | None = None

    @property
    def max_blocks(self):
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    def __post_init__(self):
        if not self.model:
            return

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings)

        if self.speculate:
            assert self.draft, "draft model path required for speculative decoding"
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft)
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)

            if self.draft_async:
                if self.fan_out_list is None:
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list)
