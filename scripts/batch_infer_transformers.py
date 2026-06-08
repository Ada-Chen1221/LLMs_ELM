#!/usr/bin/env python
"""Batch inference with a local/downloaded HuggingFace causal LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.model_utils import get_torch_dtype, print_cuda_info  # noqa: E402

DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with a local/downloaded Transformers model.")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--input_file", default="data/prompts.jsonl")
    parser.add_argument("--output_file", default="outputs/batch_outputs.jsonl")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--messages_field", default="messages")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {path} must be a JSON object.")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No prompt rows found in {path}")
    return rows


def row_to_messages(row: dict[str, Any], prompt_field: str, messages_field: str, system_prompt: str | None) -> list[dict[str, str]]:
    if messages_field in row:
        messages = row[messages_field]
        if not isinstance(messages, list):
            raise ValueError(f"'{messages_field}' must be a list of chat messages.")
        return messages

    prompt = row.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Each row must contain non-empty '{prompt_field}' or '{messages_field}'.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_text(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)


def first_model_device(model):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for device in model.hf_device_map.values():
            if isinstance(device, str) and device not in {"cpu", "disk", "meta"}:
                return device
            if isinstance(device, int):
                return f"cuda:{device}"
    return getattr(model, "device", "cuda")


def batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_file)
    print(f"Loaded {len(rows)} prompts from {args.input_file}")
    print(f"Model: {args.model_name_or_path}")
    print_cuda_info()

    common = {
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **common)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        **common,
    )
    model.eval()

    results: list[dict[str, Any]] = []
    do_sample = args.temperature > 0
    for start, chunk in batched(rows, args.batch_size):
        texts = [
            render_text(
                tokenizer,
                row_to_messages(row, args.prompt_field, args.messages_field, args.system_prompt),
            )
            for row in chunk
        ]
        model_inputs = tokenizer(texts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            model_inputs = model_inputs.to(first_model_device(model) if args.device_map == "auto" else "cuda")
            if args.device_map != "auto":
                model.to("cuda")

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature if do_sample else None,
                top_p=args.top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_token_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated)
        ]
        responses = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for offset, (row, response) in enumerate(zip(chunk, responses)):
            output = {"index": start + offset, "response": response.strip()}
            if args.id_field in row:
                output[args.id_field] = row[args.id_field]
            if args.prompt_field in row:
                output[args.prompt_field] = row[args.prompt_field]
            results.append(output)
        print(f"Processed {min(start + len(chunk), len(rows))}/{len(rows)}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
