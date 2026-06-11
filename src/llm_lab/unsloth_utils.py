"""Helpers for Unsloth-based SFT and RL workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UnslothLoadConfig:
    model_name_or_path: str
    max_seq_length: int
    dtype: str
    load_in_4bit: bool
    fast_inference: bool = False
    cache_dir: str | None = None
    revision: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = True


@dataclass
class UnslothLoraConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    gradient_checkpointing: bool | str
    random_state: int


def unsloth_dtype(dtype: str) -> Any:
    """Map CLI dtype strings to the representation expected by Unsloth."""
    if dtype == "auto":
        return None

    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def load_unsloth_model(config: UnslothLoadConfig):
    """Load a causal LM through Unsloth's FastLanguageModel."""
    from unsloth import FastLanguageModel

    kwargs: dict[str, Any] = {
        "model_name": config.model_name_or_path,
        "max_seq_length": config.max_seq_length,
        "dtype": unsloth_dtype(config.dtype),
        "load_in_4bit": config.load_in_4bit,
        "fast_inference": config.fast_inference,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.cache_dir is not None:
        kwargs["cache_dir"] = config.cache_dir
    if config.revision is not None:
        kwargs["revision"] = config.revision
    if config.local_files_only:
        kwargs["local_files_only"] = True
    return FastLanguageModel.from_pretrained(**kwargs)


def add_lora_adapters(model: Any, config: UnslothLoraConfig):
    """Attach Unsloth-optimized LoRA adapters without modifying model internals by hand."""
    from unsloth import FastLanguageModel

    checkpointing = "unsloth" if config.gradient_checkpointing else False
    return FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        use_gradient_checkpointing=checkpointing,
        random_state=config.random_state,
    )


def enable_unsloth_inference(model: Any) -> None:
    """Enable Unsloth's optimized inference mode on a loaded model."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
