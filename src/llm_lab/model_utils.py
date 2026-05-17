"""Model and tokenizer helpers for native HuggingFace Transformers workflows."""

from __future__ import annotations

from typing import Any


def get_torch_dtype(dtype: str):
    """Map a CLI dtype string to a torch dtype or ``"auto"``."""
    import torch

    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def print_cuda_info() -> None:
    """Print CUDA availability and visible GPU details."""
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible GPU count: {gpu_count}")
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"GPU {idx}: {props.name} ({total_gb:.1f} GiB)")


def ensure_pad_token(tokenizer: Any) -> None:
    """Ensure a causal-LM tokenizer has a pad token for batching/training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def maybe_enable_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """Enable gradient checkpointing and input gradients when requested."""
    if not enabled:
        return
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
