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
