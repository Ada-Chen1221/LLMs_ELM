#!/usr/bin/env python
"""Unsloth LoRA/QLoRA SFT training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import load_sft_dataset  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, print_cuda_info, require_min_cuda_memory  # noqa: E402
from llm_lab.train_utils import (  # noqa: E402
    ResponseOnlyDataCollator,
    build_sft_trainer,
    get_training_args,
    print_train_summary,
    validate_trainer_checkpoint_resume,
)
from llm_lab.unsloth_utils import (  # noqa: E402
    UnslothLoraConfig,
    add_unsloth_lora,
    apply_chat_template_if_requested,
    load_unsloth_model,
    parse_target_modules,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA SFT adapter with Unsloth.")
    parser.add_argument("--model_name_or_path", default="model/Qwen3-1.7B")
    parser.add_argument("--train_file", default="data/toy_sft.jsonl")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--response_field", default="groundtruth")
    parser.add_argument("--split_field", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output_dir", default="outputs/qwen3_1p7b_unsloth_lora")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--dtype", default="auto", choices=["auto", "none", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--chat_template", default=None, help="Optional Unsloth template name, e.g. llama-3.1 or chatml.")
    parser.add_argument("--min_free_gpu_memory_gb", type=float, default=6.0, help="Fail early if visible GPU 0 has less free memory before model loading. Set 0 to disable.")
    parser.add_argument(
        "--loss_on_prompt",
        action="store_true",
        help="Compute SFT loss on prompt tokens too. Default is response-only loss.",
    )
    parser.add_argument(
        "--save_method",
        default="lora",
        choices=["lora", "merged_16bit", "merged_4bit"],
        help="How to save after training. 'lora' saves adapters only; merged_* uses Unsloth save_pretrained_merged.",
    )
    return parser.parse_args()


def build_trainer(args: argparse.Namespace, model, tokenizer, train_dataset, training_args):
    if args.loss_on_prompt:
        print("Training with loss on full prompt + response tokens (--loss_on_prompt).", flush=True)
        return build_sft_trainer(model, tokenizer, train_dataset, training_args)

    from transformers import Trainer

    print("Training with response-only loss: prompt tokens are masked with label=-100.", flush=True)
    return Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=ResponseOnlyDataCollator(tokenizer, max_length=args.max_length),
    )


def save_model(args: argparse.Namespace, model, tokenizer) -> None:
    if args.save_method == "lora":
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        model.save_pretrained_merged(args.output_dir, tokenizer, save_method=args.save_method)
    print(f"Model saved to: {args.output_dir} ({args.save_method})")


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install -r requirements.txt") from exc

    if args.bf16:
        args.fp16 = False
    try:
        validate_trainer_checkpoint_resume(args.resume_from_checkpoint, torch)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Unsloth training is intended for a CUDA GPU.")
    print_train_summary(args)
    print_cuda_info()
    require_min_cuda_memory(args.min_free_gpu_memory_gb, context="Unsloth SFT/QLoRA training")

    try:
        model, tokenizer = load_unsloth_model(
            args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = apply_chat_template_if_requested(tokenizer, args.chat_template)
        ensure_pad_token(tokenizer)
        model = add_unsloth_lora(
            model,
            UnslothLoraConfig(
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                target_modules=parse_target_modules(args.target_modules),
            ),
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load/patch Unsloth model '{args.model_name_or_path}'. "
            "Check CUDA, Unsloth installation, model path, HuggingFace permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    train_dataset = load_sft_dataset(
        args.train_file,
        tokenizer,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        split_field=args.split_field,
        split=args.split,
        response_only_loss=not args.loss_on_prompt,
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
    trainer = build_trainer(args, model, tokenizer, train_dataset, training_args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    save_model(args, trainer.model, tokenizer)


if __name__ == "__main__":
    main()
