#!/usr/bin/env python
"""LoRA SFT training with Transformers + PEFT + TRL."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import load_sft_jsonl  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, maybe_enable_gradient_checkpointing, print_cuda_info  # noqa: E402
from llm_lab.train_utils import (  # noqa: E402
    LoraCliConfig,
    build_lora_config,
    build_sft_trainer,
    get_training_args,
    parse_target_modules,
    print_train_summary,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with TRL SFTTrainer.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", default="data/toy_sft.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3_1p7b_lora")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install -r requirements.txt") from exc

    if args.bf16:
        args.fp16 = False
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Training on CPU will be very slow and is intended only for debugging imports.")
    print_train_summary(args)
    print_cuda_info()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        ensure_pad_token(tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load model/tokenizer '{args.model_name_or_path}'. "
            "Check network access, model name/path, HuggingFace permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)
    train_dataset = load_sft_jsonl(args.train_file, tokenizer)
    lora_config = build_lora_config(
        LoraCliConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=parse_target_modules(args.target_modules),
        )
    )
    training_args = get_training_args(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
    )
    trainer = build_sft_trainer(model, tokenizer, train_dataset, training_args, lora_config)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
