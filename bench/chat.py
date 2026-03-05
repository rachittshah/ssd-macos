"""Interactive chat with SSD-macOS.

Usage:
    uv run python bench/chat.py                  # Autoregressive (default)
    uv run python bench/chat.py --mode sd        # Sync speculative decoding
    uv run python bench/chat.py --mode ssd       # Async SSD
"""
import argparse
import sys
import time

import mlx.core as mx

from ssd_macos.engine.model_runner import ModelRunner
from ssd_macos.engine.sequence import Sequence
from ssd_macos.engine.step import AutoRegressiveStep, SpecDecodeStep, SSDStep
from ssd_macos.engine.verifier import Verifier
from ssd_macos.sampling_params import SamplingParams

TARGET_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DRAFT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def build_engine(mode: str, lookahead: int, fan_out: int):
    """Build the appropriate engine components."""
    print(f"Loading target model: {TARGET_MODEL}")
    target_runner = ModelRunner(TARGET_MODEL)
    tokenizer = target_runner.tokenizer

    draft_runner = None
    if mode in ("sd", "ssd"):
        print(f"Loading draft model: {DRAFT_MODEL}")
        draft_runner = ModelRunner(DRAFT_MODEL)

    if mode == "ar":
        step = AutoRegressiveStep(target_runner)
    elif mode == "sd":
        verifier = Verifier.__new__(Verifier)
        verifier.total_draft = 0
        verifier.total_accepted = 0
        verifier.lookahead = lookahead
        step = SpecDecodeStep(draft_runner, target_runner, verifier, lookahead)
    else:
        step = SSDStep(draft_runner, target_runner, lookahead, fan_out)

    return target_runner, draft_runner, step, tokenizer


def generate(
    prompt: str,
    target_runner: ModelRunner,
    draft_runner: ModelRunner | None,
    step,
    tokenizer,
    max_tokens: int = 256,
    temperature: float = 0.0,
    stream: bool = True,
):
    """Generate a response, optionally streaming tokens."""
    sp = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        sp.stop_token_ids = [eos]

    token_ids = tokenizer.encode(prompt)
    seq = Sequence(prompt_token_ids=token_ids, sampling_params=sp)

    target_cache = target_runner.make_cache()
    draft_cache = draft_runner.make_cache() if draft_runner else None

    t0 = time.perf_counter()

    # Prefill
    if isinstance(step, AutoRegressiveStep):
        step.prefill(seq, target_cache)
    else:
        step.prefill(seq, draft_cache, target_cache)

    if stream:
        # Print first token
        text = tokenizer.decode(seq.completion_token_ids)
        sys.stdout.write(text)
        sys.stdout.flush()
        prev_len = len(seq.completion_token_ids)

    # Decode loop
    while not seq.is_finished:
        if isinstance(step, AutoRegressiveStep):
            step.decode(seq, target_cache)
        else:
            step.decode(seq, draft_cache, target_cache)

        if stream:
            new_ids = seq.completion_token_ids[prev_len:]
            if new_ids:
                text = tokenizer.decode(new_ids)
                sys.stdout.write(text)
                sys.stdout.flush()
                prev_len = len(seq.completion_token_ids)

    elapsed = time.perf_counter() - t0
    n = seq.num_completion_tokens
    tps = n / elapsed if elapsed > 0 else 0

    if stream:
        print()

    print(f"\n[{n} tokens, {tps:.1f} tok/s, {elapsed:.2f}s]")
    return tokenizer.decode(seq.completion_token_ids)


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with SSD-macOS")
    parser.add_argument("--mode", choices=["ar", "sd", "ssd"], default="ar")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=3)
    parser.add_argument("--fan-out", type=int, default=3)
    args = parser.parse_args()

    target_runner, draft_runner, step, tokenizer = build_engine(
        args.mode, args.lookahead, args.fan_out
    )

    mode_label = {"ar": "Autoregressive", "sd": "Spec Decode", "ssd": "Async SSD"}
    print(f"\nSSD-macOS Chat ({mode_label[args.mode]})")
    print("Type 'quit' or Ctrl-C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        print("Assistant: ", end="", flush=True)
        generate(
            user_input,
            target_runner,
            draft_runner,
            step,
            tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print()


if __name__ == "__main__":
    main()
