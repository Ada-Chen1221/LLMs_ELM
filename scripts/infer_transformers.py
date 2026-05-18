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
    parser.add_argument("--cache_dir", default=None, help="Optional HuggingFace cache directory.")
    parser.add_argument("--revision", default=None, help="Optional model revision/branch/commit.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load only from local cache/local path; fail fast instead of downloading from HuggingFace.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow custom model/tokenizer code from the model repo. Enabled by default for Qwen compatibility.",
    )
    return parser.parse_args()


def log_step(message: str) -> None:
    print(message, flush=True)


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
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_step(f"Model: {args.model_name_or_path}")
    print_cuda_info()

    dtype = get_torch_dtype(args.dtype)
    from_pretrained_kwargs = {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }

    try:
        log_step("[1/5] Loading tokenizer... If this hangs, the model files are probably downloading or the HuggingFace connection is slow.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **from_pretrained_kwargs)
        log_step("[2/5] Loading model weights... This can take several minutes on the first run while downloading/loading checkpoints.")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map=args.device_map,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load model/tokenizer '{args.model_name_or_path}'. "
            "Check network access, model name/path, HuggingFace permissions, cache_dir/local_files_only, and local cache. "
            "If HuggingFace is slow, pre-download the model with `huggingface-cli download` or set a reachable HF_ENDPOINT. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    log_step("[3/5] Tokenizing prompt...")
    input_ids = build_inputs(tokenizer, args.prompt)
    if args.device_map == "auto" and torch.cuda.is_available():
        input_ids = input_ids.to(first_model_device(model))
    elif torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
        model.to("cuda")

    do_sample = args.temperature > 0
    log_step("[4/5] Generating text...")
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
    log_step("[5/5] Done.")
    print("\n===== Generated Text =====")
    print(text.strip())


if __name__ == "__main__":
    main()
