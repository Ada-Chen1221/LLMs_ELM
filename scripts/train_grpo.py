#!/usr/bin/env python
"""Unsloth GRPO reinforcement learning training scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import load_grpo_dataset  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, configure_visible_gpu, print_cuda_info, require_min_cuda_memory  # noqa: E402
from llm_lab.unsloth_utils import (  # noqa: E402
    UnslothLoraConfig,
    add_unsloth_lora,
    apply_chat_template_if_requested,
    default_precision_flags,
    load_unsloth_model,
    parse_target_modules,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with Unsloth + TRL GRPO.")
    parser.add_argument("--model_name_or_path", default="model/Qwen3-1.7B")
    parser.add_argument("--gpu_id", default=None, help="Physical GPU id to use, e.g. 0 or 1. Sets CUDA_VISIBLE_DEVICES before torch/unsloth import.")
    parser.add_argument("--train_file", default="data/toy_sft.jsonl")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--answer_field", default="groundtruth")
    parser.add_argument("--split_field", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output_dir", default="outputs/qwen3_1p7b_unsloth_grpo")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--dtype", default="auto", choices=["auto", "none", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gpu_memory_utilization", type=float, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chat_template", default=None, help="Optional Unsloth template name, e.g. llama-3.1 or chatml.")
    parser.add_argument("--min_free_gpu_memory_gb", type=float, default=8.0, help="Fail early if visible GPU 0 has less free memory before model loading. Set 0 to disable.")
    parser.add_argument(
        "--reward_type",
        default="contains",
        choices=["contains", "exact", "numeric"],
        help="Built-in reward: substring containment, exact normalized match, or numeric answer match.",
    )
    return parser.parse_args()


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def last_number(value: str) -> str | None:
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    return matches[-1] if matches else None


def make_reward_func(reward_type: str):
    def reward_func(completions, answer, **_: Any) -> list[float]:
        rewards: list[float] = []
        for completion, expected in zip(completions, answer):
            generated = completion_to_text(completion)
            if reward_type == "exact":
                rewards.append(1.0 if normalize_text(generated) == normalize_text(expected) else 0.0)
            elif reward_type == "numeric":
                rewards.append(1.0 if last_number(generated) == last_number(expected) and last_number(expected) else 0.0)
            else:
                rewards.append(1.0 if normalize_text(expected) in normalize_text(generated) else 0.0)
        return rewards

    return reward_func


def main() -> None:
    args = parse_args()
    configure_visible_gpu(args.gpu_id)
    try:
        import torch
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as exc:
        raise SystemExit("Missing Unsloth/TRL dependency. Install with: pip install -r requirements.txt") from exc

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Unsloth GRPO is intended for a CUDA GPU.")
    print(f"Model: {args.model_name_or_path}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print_cuda_info()
    require_min_cuda_memory(args.min_free_gpu_memory_gb, context="Unsloth GRPO training")

    fp16, bf16 = default_precision_flags(args.bf16, args.fp16)
    try:
        model, tokenizer = load_unsloth_model(
            args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            fast_inference=args.fast_inference,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_lora_rank=args.lora_r,
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
        PatchFastRL("grpo", FastLanguageModel)
    except Exception as exc:
        raise SystemExit(
            f"Failed to initialize Unsloth GRPO model '{args.model_name_or_path}'. "
            "Check CUDA, Unsloth/vLLM installation, model path, permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    train_dataset = load_grpo_dataset(
        args.train_file,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        split_field=args.split_field,
        split=args.split,
    )
    grpo_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "beta": args.beta,
        "fp16": fp16,
        "bf16": bf16,
        "logging_steps": 1,
        "save_steps": 50,
        "report_to": "none",
    }
    if args.num_train_epochs is not None:
        grpo_kwargs["num_train_epochs"] = args.num_train_epochs
    else:
        grpo_kwargs["max_steps"] = args.max_steps
    if args.fast_inference:
        grpo_kwargs["use_vllm"] = True

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(**grpo_kwargs)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_func(args.reward_type)],
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"GRPO LoRA adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
