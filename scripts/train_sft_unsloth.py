#!/usr/bin/env python
"""Unsloth LoRA/QLoRA SFT training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import load_sft_jsonl  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, print_cuda_info  # noqa: E402
from llm_lab.train_utils import (  # noqa: E402
    build_sft_trainer,
    get_training_args,
    parse_target_modules,
    print_train_summary,
)
from llm_lab.unsloth_utils import (  # noqa: E402
    UnslothLoadConfig,
    UnslothLoraConfig,
    add_lora_adapters,
    load_unsloth_model,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA SFT adapter with Unsloth + TRL SFTTrainer.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", default="data/toy_sft.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3_1p7b_unsloth_sft")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_state", type=int, default=3407)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    # Legacy QLoRA flags are accepted for compatibility with older commands/configs;
    # Unsloth chooses the 4-bit backend from --load_in_4bit.
    parser.add_argument("--bnb_4bit_quant_type", default="nf4", choices=["nf4", "fp4"], help=argparse.SUPPRESS)
    parser.add_argument("--bnb_4bit_use_double_quant", action=argparse.BooleanOptionalAction, default=True, help=argparse.SUPPRESS)
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bf16:
        args.fp16 = False

    print_train_summary(args)
    print(f"Unsloth load_in_4bit: {args.load_in_4bit}")
    print_cuda_info()

    model, tokenizer = load_unsloth_model(
        UnslothLoadConfig(
            model_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            cache_dir=args.cache_dir,
            revision=args.revision,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
    )
    ensure_pad_token(tokenizer)
    model = add_lora_adapters(
        model,
        UnslothLoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=parse_target_modules(args.target_modules),
            gradient_checkpointing=args.gradient_checkpointing,
            random_state=args.random_state,
        ),
    )

    train_dataset = load_sft_jsonl(args.train_file, tokenizer)
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

    trainer = build_sft_trainer(model, tokenizer, train_dataset, training_args)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Unsloth SFT adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
