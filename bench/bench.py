"""Benchmark SSD-macOS inference modes.

Usage:
    uv run python bench/bench.py --mode ar      # Autoregressive baseline
    uv run python bench/bench.py --mode sd      # Sync speculative decoding
    uv run python bench/bench.py --mode ssd     # Async SSD
    uv run python bench/bench.py --mode all     # All modes
"""
import argparse
import time

import mlx.core as mx

from ssd_macos.engine.model_runner import ModelRunner
from ssd_macos.engine.sequence import Sequence, SequenceStatus
from ssd_macos.engine.step import AutoRegressiveStep, SpecDecodeStep, SSDStep
from ssd_macos.engine.verifier import Verifier
from ssd_macos.sampling_params import SamplingParams

PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "What are the main differences between Python and Rust?",
    "Describe the process of photosynthesis step by step.",
    "What is the significance of the Turing test?",
]

TARGET_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DRAFT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

MAX_TOKENS = 128


def make_seq(prompt: str, tokenizer, max_tokens: int = MAX_TOKENS) -> Sequence:
    token_ids = tokenizer.encode(prompt)
    sp = SamplingParams(max_tokens=max_tokens)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        sp.stop_token_ids = [eos]
    return Sequence(prompt_token_ids=token_ids, sampling_params=sp)


def bench_ar(prompts: list[str] | None = None, max_tokens: int = MAX_TOKENS):
    """Benchmark autoregressive baseline."""
    prompts = prompts or PROMPTS
    print(f"\n{'='*60}")
    print("Autoregressive Baseline")
    print(f"{'='*60}")

    print(f"Loading target model: {TARGET_MODEL}")
    runner = ModelRunner(TARGET_MODEL)
    tokenizer = runner.tokenizer
    step = AutoRegressiveStep(runner)

    results = []
    for prompt in prompts:
        seq = make_seq(prompt, tokenizer, max_tokens)
        cache = runner.make_cache()

        # Prefill
        t0 = time.perf_counter()
        step.prefill(seq, cache)
        mx.eval(mx.zeros(1))  # sync
        ttft = time.perf_counter() - t0

        # Decode
        t_decode_start = time.perf_counter()
        while not seq.is_finished:
            step.decode(seq, cache)
        t_total = time.perf_counter() - t0
        t_decode = time.perf_counter() - t_decode_start

        n_tokens = seq.num_completion_tokens
        tps = n_tokens / t_decode if t_decode > 0 else 0
        text = tokenizer.decode(seq.completion_token_ids)

        results.append({
            "prompt": prompt[:50],
            "tokens": n_tokens,
            "ttft": ttft,
            "total_time": t_total,
            "decode_time": t_decode,
            "tps": tps,
        })
        print(f"  [{n_tokens} tok] TTFT={ttft:.3f}s  TPS={tps:.1f}  total={t_total:.3f}s")

    _print_summary(results)
    return results


def bench_sd(prompts: list[str] | None = None, max_tokens: int = MAX_TOKENS, lookahead: int = 3):
    """Benchmark synchronous speculative decoding."""
    prompts = prompts or PROMPTS
    print(f"\n{'='*60}")
    print(f"Synchronous Speculative Decoding (K={lookahead})")
    print(f"{'='*60}")

    print(f"Loading target: {TARGET_MODEL}")
    target_runner = ModelRunner(TARGET_MODEL)
    print(f"Loading draft:  {DRAFT_MODEL}")
    draft_runner = ModelRunner(DRAFT_MODEL)
    tokenizer = target_runner.tokenizer

    verifier = Verifier.__new__(Verifier)
    verifier.total_draft = 0
    verifier.total_accepted = 0
    verifier.lookahead = lookahead

    sd_step = SpecDecodeStep(
        draft_runner=draft_runner,
        target_runner=target_runner,
        verifier=verifier,
        lookahead=lookahead,
    )

    results = []
    for prompt in prompts:
        seq = make_seq(prompt, tokenizer, max_tokens)
        draft_cache = draft_runner.make_cache()
        target_cache = target_runner.make_cache()

        # Prefill
        t0 = time.perf_counter()
        sd_step.prefill(seq, draft_cache, target_cache)
        mx.eval(mx.zeros(1))
        ttft = time.perf_counter() - t0

        # Decode
        t_decode_start = time.perf_counter()
        steps = 0
        while not seq.is_finished:
            sd_step.decode(seq, draft_cache, target_cache)
            steps += 1
        t_total = time.perf_counter() - t0
        t_decode = time.perf_counter() - t_decode_start

        n_tokens = seq.num_completion_tokens
        tps = n_tokens / t_decode if t_decode > 0 else 0

        results.append({
            "prompt": prompt[:50],
            "tokens": n_tokens,
            "ttft": ttft,
            "total_time": t_total,
            "decode_time": t_decode,
            "tps": tps,
            "steps": steps,
        })
        print(f"  [{n_tokens} tok] TTFT={ttft:.3f}s  TPS={tps:.1f}  total={t_total:.3f}s  steps={steps}")

    acc_rate = verifier.acceptance_rate
    print(f"  Acceptance rate: {acc_rate:.1%}")
    _print_summary(results)
    return results


def bench_ssd(prompts: list[str] | None = None, max_tokens: int = MAX_TOKENS,
              lookahead: int = 3, fan_out: int = 3):
    """Benchmark async SSD."""
    prompts = prompts or PROMPTS
    print(f"\n{'='*60}")
    print(f"Async SSD (K={lookahead}, fan_out={fan_out})")
    print(f"{'='*60}")

    print(f"Loading target: {TARGET_MODEL}")
    target_runner = ModelRunner(TARGET_MODEL)
    print(f"Loading draft:  {DRAFT_MODEL}")
    draft_runner = ModelRunner(DRAFT_MODEL)
    tokenizer = target_runner.tokenizer

    ssd_step = SSDStep(
        draft_runner=draft_runner,
        target_runner=target_runner,
        lookahead=lookahead,
        fan_out=fan_out,
    )

    results = []
    for prompt in prompts:
        seq = make_seq(prompt, tokenizer, max_tokens)
        draft_cache = draft_runner.make_cache()
        target_cache = target_runner.make_cache()

        # Prefill
        t0 = time.perf_counter()
        ssd_step.prefill(seq, draft_cache, target_cache)
        mx.eval(mx.zeros(1))
        ttft = time.perf_counter() - t0

        # Decode
        t_decode_start = time.perf_counter()
        steps = 0
        while not seq.is_finished:
            ssd_step.decode(seq, draft_cache, target_cache)
            steps += 1
        t_total = time.perf_counter() - t0
        t_decode = time.perf_counter() - t_decode_start

        n_tokens = seq.num_completion_tokens
        tps = n_tokens / t_decode if t_decode > 0 else 0

        results.append({
            "prompt": prompt[:50],
            "tokens": n_tokens,
            "ttft": ttft,
            "total_time": t_total,
            "decode_time": t_decode,
            "tps": tps,
            "steps": steps,
        })
        print(f"  [{n_tokens} tok] TTFT={ttft:.3f}s  TPS={tps:.1f}  total={t_total:.3f}s  steps={steps}")

    print(f"  Cache hit rate: {ssd_step.cache_hit_rate:.1%}")
    _print_summary(results)
    return results


def _print_summary(results: list[dict]):
    if not results:
        return
    avg_tps = sum(r["tps"] for r in results) / len(results)
    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_total = sum(r["total_time"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n  Summary: avg TPS={avg_tps:.1f}  avg TTFT={avg_ttft:.3f}s  "
          f"avg total={avg_total:.3f}s  total_tokens={total_tokens}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark SSD-macOS")
    parser.add_argument("--mode", choices=["ar", "sd", "ssd", "all"], default="ar",
                        help="Inference mode to benchmark")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--lookahead", type=int, default=3, help="Speculative lookahead K")
    parser.add_argument("--fan-out", type=int, default=3, help="SSD fan-out width")
    parser.add_argument("--prompts", type=int, default=None,
                        help="Number of prompts to use (default: all 5)")
    args = parser.parse_args()

    prompts = PROMPTS[:args.prompts] if args.prompts else PROMPTS

    if args.mode in ("ar", "all"):
        bench_ar(prompts, args.max_tokens)
    if args.mode in ("sd", "all"):
        bench_sd(prompts, args.max_tokens, args.lookahead)
    if args.mode in ("ssd", "all"):
        bench_ssd(prompts, args.max_tokens, args.lookahead, args.fan_out)


if __name__ == "__main__":
    main()
