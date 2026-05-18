"""Training helpers shared by LoRA and QLoRA scripts."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass


@dataclass
class LoraCliConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]


def parse_target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--target_modules must contain at least one module name.")
    return modules


def build_lora_config(config: LoraCliConfig):
    try:
        from peft import LoraConfig, TaskType
    except ImportError as exc:
        raise ImportError("Please install peft: pip install peft") from exc

    return LoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )


def get_training_args(
    output_dir: str,
    max_length: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    fp16: bool,
    bf16: bool,
):
    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None
    from transformers import TrainingArguments

    cls = SFTConfig or TrainingArguments
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": fp16,
        "bf16": bf16,
        "logging_steps": 1,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    if SFTConfig is not None:
        kwargs.update({"dataset_text_field": "text", "max_length": max_length})
    args = cls(**kwargs)
    if not hasattr(args, "max_length"):
        setattr(args, "max_length", max_length)
    return args


def build_sft_trainer(model, tokenizer, train_dataset, training_args, peft_config):
    try:
        from trl import SFTTrainer
    except ImportError as exc:
        raise ImportError("Please install trl: pip install trl") from exc

    signature = inspect.signature(SFTTrainer.__init__)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in signature.parameters:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = getattr(training_args, "max_length", None)
    return SFTTrainer(**kwargs)


def print_train_summary(args) -> None:
    print(f"Model: {args.model_name_or_path}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
