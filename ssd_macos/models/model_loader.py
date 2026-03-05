"""Model loading using mlx-lm.

Leverages mlx-lm's existing Llama3 and Qwen3 implementations
rather than reimplementing from scratch.
"""
from mlx_lm import load


def load_model(model_path: str, tokenizer_only: bool = False):
    """Load a model and tokenizer from a HuggingFace path or local directory.

    Args:
        model_path: HuggingFace model ID or local path
        tokenizer_only: if True, only load the tokenizer (not implemented,
            use transformers.AutoTokenizer directly for this case)

    Returns:
        (model, tokenizer) tuple. The model is an mlx-lm nn.Module with
        __call__(inputs, cache=None) interface.
    """
    model, tokenizer = load(model_path)
    return model, tokenizer


def load_draft_and_target(target_path: str, draft_path: str):
    """Load both target and draft models for speculative decoding.

    Args:
        target_path: path to the larger target model
        draft_path: path to the smaller draft model

    Returns:
        (target_model, draft_model, tokenizer) - shares tokenizer from target
    """
    target_model, tokenizer = load(target_path)
    draft_model, _ = load(draft_path)
    return target_model, draft_model, tokenizer
