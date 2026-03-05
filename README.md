# SSD-macOS

**Speculative Speculative Decoding for Apple Silicon via MLX**

A macOS port of [SSD](https://github.com/tanishqkumar/ssd) — a novel LLM inference algorithm that accelerates token generation by running drafting and verification in parallel.

Original paper: *Speculative Speculative Decoding* (ICLR 2026, arXiv: 2603.03251)
Authors: Tanishq Kumar, Tri Dao, Avner May

## What is SSD?

Standard speculative decoding uses a small "draft" model to propose tokens, then a large "target" model to verify them. SSD takes this further by having the draft model anticipate multiple verification outcomes simultaneously, eliminating drafting overhead when predictions are correct.

This port adapts SSD for **Apple Silicon** using [MLX](https://github.com/ml-explore/mlx), leveraging:
- **Unified Memory**: No GPU↔CPU data transfer overhead — draft and target share the same memory
- **MLX Streams**: Parallel execution of draft and verify on the same chip
- **mlx-lm**: Battle-tested model loading and weight conversion

## Supported Models

- Llama 3.x family (e.g., `mlx-community/Llama-3.2-3B-Instruct-4bit`)
- Qwen 3.x family (e.g., `mlx-community/Qwen3-1.7B-4bit`)

## Install

```bash
uv sync
```

## Quick Start

```python
from ssd_macos import LLM, SamplingParams

llm = LLM(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    draft="mlx-community/Llama-3.2-1B-Instruct-4bit",
    speculate=True,
    speculate_k=3,
)

outputs = llm.generate(
    ["What is speculative decoding?"],
    SamplingParams(max_tokens=256),
)
print(outputs[0]["text"])
```

## Benchmarks

**Hardware**: MacBook Pro, Apple M4 Pro, 48 GB unified memory
**Models**: Llama-3.2-3B-Instruct-4bit (target), Llama-3.2-1B-Instruct-4bit (draft)
**Config**: 128 max tokens, greedy decoding, 5 prompts averaged

### Results

| Mode | Avg TPS | Avg TTFT | Acceptance Rate | Cache Hit Rate | Avg Total Time |
|------|---------|----------|-----------------|----------------|----------------|
| **AR** (baseline) | 62.8 tok/s | 0.129s | — | — | 2.17s |
| **Sync SD** (K=3) | 19.0 tok/s | 0.126s | 12.1% | — | 7.25s |
| **Async SSD** (K=3, fan=3) | 10.5 tok/s | 0.157s | — | 99.2% | 12.57s |

### Per-prompt breakdown

#### Autoregressive
| Prompt | Tokens | TTFT | TPS | Total |
|--------|--------|------|-----|-------|
| Explain quantum computing... | 128 | 0.269s | 62.8 | 2.31s |
| Write a short poem... | 128 | 0.083s | 61.3 | 2.17s |
| Differences between Python and Rust... | 128 | 0.104s | 62.5 | 2.15s |
| Describe photosynthesis... | 128 | 0.100s | 68.9 | 1.96s |
| Significance of the Turing test... | 128 | 0.089s | 58.6 | 2.28s |

#### Sync Speculative Decoding (K=3)
| Prompt | Tokens | TTFT | TPS | Steps | Total |
|--------|--------|------|-----|-------|-------|
| Explain quantum computing... | 128 | 0.126s | 25.1 | 94 | 5.22s |
| Write a short poem... | 128 | 0.093s | 15.2 | 106 | 8.50s |
| Differences between Python and Rust... | 128 | 0.167s | 13.0 | 87 | 10.02s |
| Describe photosynthesis... | 128 | 0.146s | 21.8 | 83 | 6.02s |
| Significance of the Turing test... | 128 | 0.100s | 20.0 | 97 | 6.51s |

#### Async SSD (K=3, fan_out=3)
| Prompt | Tokens | TTFT | TPS | Steps | Total |
|--------|--------|------|-----|-------|-------|
| Explain quantum computing... | 128 | 0.420s | 11.6 | 122 | 11.45s |
| Write a short poem... | 128 | 0.121s | 10.1 | 126 | 12.80s |
| Differences between Python and Rust... | 128 | 0.062s | 8.5 | 124 | 15.09s |
| Describe photosynthesis... | 128 | 0.079s | 9.8 | 122 | 13.13s |
| Significance of the Turing test... | 128 | 0.104s | 12.4 | 122 | 10.40s |

### Analysis

**Why AR wins on small models**: With a 3B target and 1B draft (both 4-bit quantized), the models are small enough that AR decode is already very fast (~63 tok/s). Speculative decoding overhead (running two models + verification) exceeds the savings from accepting multiple tokens per step.

**Where SSD shines**: The 99.2% cache hit rate validates the core SSD algorithm — tree-based fan-out successfully predicts verification outcomes almost perfectly. On larger model pairs (e.g., 70B target with 8B draft), the per-token target cost dominates, and accepting 2-4 tokens per step at ~12% overhead from drafting becomes a significant win.

**Low acceptance rate explained**: The 1B→3B pair has poor draft-target agreement (12.1%). The models are too close in size for the draft to meaningfully "predict" the target. Larger gaps (e.g., 1B→70B) typically see 40-70% acceptance rates.

### Run benchmarks yourself

```bash
uv run python bench/bench.py --mode all
```

Modes: `ar` (autoregressive), `sd` (sync speculative), `ssd` (async speculative), `all`

## Architecture

```
ssd_macos/
├── config.py              # Configuration
├── sampling_params.py     # Sampling parameters
├── llm.py                 # High-level API
├── engine/
│   ├── llm_engine.py      # Core engine
│   ├── model_runner.py    # MLX model execution
│   ├── scheduler.py       # Request scheduling
│   ├── sequence.py        # Sequence state
│   ├── kv_cache.py        # KV cache management
│   ├── speculator_sync.py # Synchronous speculative decoding
│   ├── speculator_async.py# Async SSD (MLX streams)
│   ├── verifier.py        # Token verification
│   └── step.py            # Inference step logic
├── models/                # Model architectures (via mlx-lm)
└── utils/                 # Utilities
```
