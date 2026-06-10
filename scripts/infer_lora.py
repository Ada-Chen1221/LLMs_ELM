#!/usr/bin/env python
"""Inference with a LoRA/QLoRA adapter on top of a base causal LM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.model_utils import get_torch_dtype, print_cuda_info  # noqa: E402

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with base model + LoRA adapter.")
    parser.add_argument("--base_model_name_or_path", required=True)
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA/QLoRA adapter output dir.")
    parser.add_argument("--prompt", default="请用一句话解释什么是大语言模型。")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--merge_and_unload",
        action="store_true",
        help="Merge LoRA weights into base model for pure Transformers inference object.",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    print(message, flush=True)


def build_prompt_text(tokenizer, prompt: str, system_prompt: str | None) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}"
    return prompt


def first_model_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return device
            if isinstance(device, int):
                return f"cuda:{device}"
    return getattr(model, "device", "cuda")


def main() -> None:
    args = parse_args()
    print_cuda_info()
    log_step(f"Base model: {args.base_model_name_or_path}")
    log_step(f"Adapter: {args.adapter_path}")

    dtype = get_torch_dtype(args.dtype)
    common = {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }

    log_step("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, **common)
    log_step("[2/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        **common,
    )
    log_step("[3/5] Loading adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    if args.merge_and_unload:
        log_step("Merging LoRA adapter into base model...")
        model = model.merge_and_unload()
    model.eval()

    text = build_prompt_text(tokenizer, args.prompt, args.system_prompt)
    model_inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        if args.device_map == "auto":
            model_inputs = model_inputs.to(first_model_device(model))
        else:
            model.to("cuda")
            model_inputs = model_inputs.to("cuda")

    do_sample = args.temperature > 0
    log_step("[4/5] Generating text...")
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    log_step("[5/5] Done.")
    print("\n===== Generated Text (With LoRA) =====")
    print(response.strip())


if __name__ == "__main__":
    main()
