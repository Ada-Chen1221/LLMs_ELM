"""Shared helpers for Unsloth SFT / RL workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UnslothLoraConfig:
    """CLI-level LoRA settings passed to ``FastLanguageModel.get_peft_model``."""

    r: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] | None = None
    bias: str = "none"
    use_gradient_checkpointing: bool | str = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False


def dtype_from_cli(dtype: str | None):
    """Map common dtype strings to torch dtypes; ``None``/``auto`` lets Unsloth decide."""
    if dtype is None or dtype.lower() in {"none", "auto"}:
        return None

    import torch

    normalized = dtype.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def parse_target_modules(value: str) -> list[str]:
    """Parse comma-separated LoRA target module names."""
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target_modules must contain at least one module name.")
    return modules


def load_unsloth_model(
    model_name_or_path: str,
    max_seq_length: int,
    dtype: str | None,
    load_in_4bit: bool,
    fast_inference: bool = False,
    gpu_memory_utilization: float | None = None,
    max_lora_rank: int | None = None,
):
    """Load a model/tokenizer through Unsloth's FastLanguageModel API."""
    from unsloth import FastLanguageModel

    kwargs: dict[str, Any] = {
        "model_name": model_name_or_path,
        "max_seq_length": max_seq_length,
        "dtype": dtype_from_cli(dtype),
        "load_in_4bit": load_in_4bit,
    }
    if fast_inference:
        kwargs["fast_inference"] = True
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    if max_lora_rank is not None:
        kwargs["max_lora_rank"] = max_lora_rank
    return FastLanguageModel.from_pretrained(**kwargs)


def add_unsloth_lora(model: Any, config: UnslothLoraConfig):
    """Attach Unsloth-optimized LoRA adapters to a loaded model."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=config.random_state,
        use_rslora=config.use_rslora,
        loftq_config=None,
    )


def apply_chat_template_if_requested(tokenizer: Any, chat_template: str | None) -> Any:
    """Apply an Unsloth chat template name when one is explicitly requested."""
    if not chat_template:
        return tokenizer

    from unsloth.chat_templates import get_chat_template

    return get_chat_template(tokenizer, chat_template=chat_template)


def enable_unsloth_inference(model: Any) -> None:
    """Switch a FastLanguageModel into Unsloth's optimized inference mode."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)


def default_precision_flags(bf16: bool | None = None, fp16: bool | None = None) -> tuple[bool, bool]:
    """Return ``(fp16, bf16)`` using Unsloth's bf16 detection when not specified."""
    from unsloth import is_bfloat16_supported

    if bf16 is None and fp16 is None:
        bf16 = is_bfloat16_supported()
        fp16 = not bf16
    elif bf16 is None:
        bf16 = not bool(fp16)
    elif fp16 is None:
        fp16 = not bool(bf16)
    if bf16:
        fp16 = False
    return bool(fp16), bool(bf16)
