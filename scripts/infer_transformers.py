#!/usr/bin/env python
"""Native HuggingFace Transformers causal-LM inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.model_utils import get_torch_dtype, print_cuda_info  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native Transformers text generation.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", default="请用一句话解释什么是大语言模型。")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    return parser.parse_args()


def build_inputs(tokenizer, prompt: str):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install -r requirements.txt") from exc

    print(f"Model: {args.model_name_or_path}")
    print_cuda_info()

    dtype = get_torch_dtype(args.dtype)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load model/tokenizer '{args.model_name_or_path}'. "
            "Check network access, model name/path, HuggingFace permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    input_ids = build_inputs(tokenizer, args.prompt)
    if args.device_map == "auto" and hasattr(model, "device"):
        input_ids = input_ids.to(model.device)
    elif torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")

    do_sample = args.temperature > 0
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("\n===== Generated Text =====")
    print(text.strip())


if __name__ == "__main__":
    main()
