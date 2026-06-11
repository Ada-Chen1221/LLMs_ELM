#!/usr/bin/env python
"""Run Unsloth inference from a base model or saved LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import _apply_chat_template  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, configure_visible_gpu, print_cuda_info, require_min_cuda_memory  # noqa: E402
from llm_lab.unsloth_utils import (  # noqa: E402
    apply_chat_template_if_requested,
    enable_unsloth_inference,
    load_unsloth_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsloth inference for a base model or LoRA adapter directory.")
    parser.add_argument("--model_name_or_path", default="outputs/qwen3_1p7b_unsloth_lora")
    parser.add_argument("--gpu_id", default=None, help="Physical GPU id to use, e.g. 0 or 1. Sets CUDA_VISIBLE_DEVICES before torch/unsloth import.")
    parser.add_argument("--prompt", default="请用一句话解释什么是大语言模型。")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "none", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template", default=None, help="Optional Unsloth template name, e.g. llama-3.1 or chatml.")
    parser.add_argument("--min_free_gpu_memory_gb", type=float, default=3.0, help="Fail early if visible GPU 0 has less free memory before model loading. Set 0 to disable.")
    return parser.parse_args()


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def main() -> None:
    args = parse_args()
    configure_visible_gpu(args.gpu_id)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install -r requirements.txt") from exc

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Unsloth inference is intended for a CUDA GPU.")
    print_cuda_info()
    require_min_cuda_memory(args.min_free_gpu_memory_gb, context="Unsloth inference")

    try:
        model, tokenizer = load_unsloth_model(
            args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = apply_chat_template_if_requested(tokenizer, args.chat_template)
        ensure_pad_token(tokenizer)
        enable_unsloth_inference(model)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load Unsloth model/adapter '{args.model_name_or_path}'. "
            "Check the path, CUDA, Unsloth installation, HuggingFace permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    messages = build_messages(args.prompt, args.system_prompt)
    text = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_cache": True,
    }
    if args.min_p is not None:
        generation_kwargs["min_p"] = args.min_p
    outputs = model.generate(**inputs, **generation_kwargs)
    generated = outputs[:, inputs["input_ids"].shape[-1] :]
    print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
