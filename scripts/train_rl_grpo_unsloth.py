#!/usr/bin/env python
"""Minimal Unsloth GRPO reinforcement-learning training scaffold."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.model_utils import ensure_pad_token, print_cuda_info  # noqa: E402
from llm_lab.train_utils import parse_target_modules, print_train_summary  # noqa: E402
from llm_lab.unsloth_utils import (  # noqa: E402
    UnslothLoadConfig,
    UnslothLoraConfig,
    add_lora_adapters,
    load_unsloth_model,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA policy with Unsloth + TRL GRPOTrainer.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_file", default="data/toy_rl.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3_1p7b_unsloth_grpo")
    parser.add_argument("--max_length", type=int, default=1024, help="Model/context length used by Unsloth.")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient. 0 avoids loading a reference model in modern TRL.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--target_modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fast_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_state", type=int, default=3407)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_grpo_jsonl(path: str | Path):
    from datasets import Dataset

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    rows: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} of {data_path} must be a JSON object.")
            prompt = obj.get("prompt")
            messages = obj.get("messages")
            answer = obj.get("answer") or obj.get("target") or obj.get("reference")
            if messages is not None:
                if not isinstance(messages, list):
                    raise ValueError(f"Line {line_no}: messages must be a list.")
                prompt_value = messages
            elif isinstance(prompt, str) and prompt.strip():
                prompt_value = prompt
            else:
                raise ValueError(f"Line {line_no}: expected non-empty prompt or messages.")
            row = {"prompt": prompt_value}
            if isinstance(answer, str) and answer.strip():
                row["answer"] = answer.strip()
            rows.append(row)

    if not rows:
        raise ValueError(f"No RL examples found in {data_path}")
    return Dataset.from_list(rows)


def exact_answer_reward(completions: list[Any], answer: list[str] | None = None, **_: Any) -> list[float]:
    """Reward exact/contained answers when an answer column is present."""
    if answer is None:
        return [0.0 for _ in completions]

    rewards: list[float] = []
    for completion, expected in zip(completions, answer):
        text = completion_to_text(completion)
        expected_text = str(expected).strip()
        if not expected_text:
            rewards.append(0.0)
        elif text.strip() == expected_text:
            rewards.append(1.0)
        elif expected_text in text:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Small generic reward for concise, non-empty completions without obvious repetition."""
    rewards: list[float] = []
    for completion in completions:
        text = completion_to_text(completion).strip()
        if not text:
            rewards.append(0.0)
            continue
        repeated = bool(re.search(r"(.{8,}?)\1{2,}", text, flags=re.DOTALL))
        length_bonus = 0.25 if len(text) <= 400 else 0.0
        rewards.append((0.25 if not repeated else 0.0) + length_bonus)
    return rewards


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = [item.get("content", "") for item in completion if isinstance(item, dict)]
        return "".join(str(part) for part in parts)
    return str(completion)


def main() -> None:
    args = parse_args()
    if args.bf16:
        args.fp16 = False

    print_train_summary(args)
    print(f"Unsloth load_in_4bit: {args.load_in_4bit}")
    print(f"Unsloth fast_inference: {args.fast_inference}")
    print_cuda_info()

    model, tokenizer = load_unsloth_model(
        UnslothLoadConfig(
            model_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            fast_inference=args.fast_inference,
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
    train_dataset = load_grpo_jsonl(args.train_file)

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(args.max_steps, 1),
        report_to="none",
    )
    trainer_kwargs = {
        "model": model,
        "reward_funcs": [exact_answer_reward, format_reward],
        "args": training_args,
        "train_dataset": train_dataset,
    }
    signature = inspect.signature(GRPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Unsloth GRPO adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
