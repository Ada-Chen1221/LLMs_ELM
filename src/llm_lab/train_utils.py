"""Training helpers shared by LoRA and QLoRA scripts."""

from __future__ import annotations

import inspect
import os
import re
from dataclasses import dataclass
from pathlib import Path


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


def _major_minor_version(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return (0, 0)
    return int(match.group(1)), int(match.group(2))


def is_torch_at_least(torch_module, major: int, minor: int) -> bool:
    """Return whether the imported torch module is at least a major.minor version."""
    current_major, current_minor = _major_minor_version(getattr(torch_module, "__version__", "0.0"))
    return (current_major, current_minor) >= (major, minor)


def validate_trainer_checkpoint_resume(resume_from_checkpoint: str | None, torch_module) -> None:
    """Fail early with a clear message for unsafe torch.load checkpoint resume.

    Recent Transformers versions block loading Trainer optimizer/scheduler state
    with torch<2.6 because those files are commonly stored as ``*.pt`` and are
    read via ``torch.load``. Loading adapter weights only is still possible via
    ``--adapter_path`` because PEFT adapter weights are typically safetensors.
    """
    if not resume_from_checkpoint or is_torch_at_least(torch_module, 2, 6):
        return

    checkpoint_dir = Path(resume_from_checkpoint)
    torch_state_files = ["optimizer.pt", "scheduler.pt", "scaler.pt", "rng_state.pth"]
    present_files = [name for name in torch_state_files if (checkpoint_dir / name).exists()]
    if not present_files:
        return

    files = ", ".join(present_files)
    raise RuntimeError(
        f"Cannot exactly resume Trainer checkpoint {checkpoint_dir} with torch {torch_module.__version__}: "
        "Transformers now requires torch>=2.6 to load Trainer .pt state files safely. "
        f"Found checkpoint state file(s): {files}.\n"
        "Options:\n"
        "  1) Upgrade to a CUDA-compatible torch>=2.6 and rerun with --resume_from_checkpoint.\n"
        "  2) If you only need to continue adapter fine-tuning, add --resume_checkpoint_as_adapter. "
        "This loads the checkpoint adapter weights but starts a fresh optimizer/scheduler, so you can also change batch size."
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


class ResponseOnlyDataCollator:
    """Tokenize SFT rows and mask prompt tokens with ``-100`` labels.

    Each dataset row must contain:
    - ``text``: full chat-formatted prompt + assistant response
    - ``prompt_text``: the prompt prefix before the assistant response

    The model still receives the full sequence as input, but the loss is only
    computed on labels after ``prompt_text``.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        import torch

        texts = [feature["text"] for feature in features]
        prompt_texts = [feature.get("prompt_text", "") for feature in features]
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        for row_idx, prompt_text in enumerate(prompt_texts):
            if prompt_text:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    max_length=self.max_length,
                    truncation=True,
                )["input_ids"]
                nonpad_positions = batch["attention_mask"][row_idx].nonzero(as_tuple=False).flatten()
                if len(nonpad_positions) > 0:
                    prompt_start = int(nonpad_positions[0].item())
                    prompt_end = min(prompt_start + len(prompt_ids), labels.shape[1])
                    labels[row_idx, prompt_start:prompt_end] = -100
            labels[row_idx, batch["attention_mask"][row_idx] == 0] = -100
            if torch.all(labels[row_idx] == -100):
                raise ValueError(
                    "A training example has no response tokens left after prompt masking/truncation. "
                    "Increase --max_length or shorten the prompt."
                )
        batch["labels"] = labels
        return batch


def build_response_only_trainer(model, tokenizer, train_dataset, training_args, peft_config=None):
    """Build a Transformers Trainer that computes loss only on response tokens."""
    from transformers import Trainer

    if peft_config is not None:
        try:
            from peft import get_peft_model
        except ImportError as exc:
            raise ImportError("Please install peft: pip install peft") from exc
        model = get_peft_model(model, peft_config)

    max_length = getattr(training_args, "max_length", None)
    if max_length is None:
        raise ValueError("training_args must provide max_length for response-only training.")
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=ResponseOnlyDataCollator(tokenizer, max_length=max_length),
    )


def build_sft_trainer(model, tokenizer, train_dataset, training_args, peft_config=None):
    try:
        from trl import SFTTrainer
    except ImportError as exc:
        raise ImportError("Please install trl: pip install trl") from exc

    signature = inspect.signature(SFTTrainer.__init__)
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    if peft_config is not None:
        kwargs["peft_config"] = peft_config
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in signature.parameters:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in signature.parameters:
        kwargs["max_seq_length"] = getattr(training_args, "max_length", None)
    return SFTTrainer(**kwargs)


def cast_trainable_parameters_to_fp32(model) -> int:
    """Cast trainable floating-point parameters to fp32 for fp16 AMP stability.

    Some local Qwen checkpoints/configs may create LoRA adapter parameters in
    bfloat16. When ``TrainingArguments(fp16=True)`` is used, Accelerate enables
    a GradScaler, and PyTorch cannot unscale bfloat16 gradients with the fp16
    scaler (``_amp_foreach_non_finite_check_and_unscale_cuda``). Keeping the
    small trainable adapter weights in fp32 is the common, stable QLoRA/LoRA
    setup and avoids that runtime error while the frozen base model remains
    quantized/fp16.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Please install torch: pip install torch") from exc

    converted = 0
    for param in model.parameters():
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)
            converted += param.numel()
            if param.grad is not None:
                param.grad.data = param.grad.data.to(torch.float32)
    return converted


def print_train_summary(args) -> None:
    print(f"Model: {args.model_name_or_path}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    resume_from_checkpoint = getattr(args, "resume_from_checkpoint", None)
    adapter_path = getattr(args, "adapter_path", None)
    if resume_from_checkpoint:
        print(f"Resume from checkpoint: {resume_from_checkpoint}")
    if adapter_path:
        print(f"Continue from adapter: {adapter_path}")
    if hasattr(args, "loss_on_prompt"):
        loss_mode = "full prompt + response" if args.loss_on_prompt else "response only (prompt labels = -100)"
        print(f"Loss mode: {loss_mode}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
