#!/usr/bin/env python
"""Batch inference with an Unsloth base model or LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.data import _apply_chat_template  # noqa: E402
from llm_lab.model_utils import ensure_pad_token, print_cuda_info  # noqa: E402
from llm_lab.unsloth_utils import (  # noqa: E402
    apply_chat_template_if_requested,
    enable_unsloth_inference,
    load_unsloth_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with an Unsloth model or saved LoRA adapter.")
    parser.add_argument("--model_name_or_path", default="outputs/qwen3_1p7b_unsloth_lora")
    parser.add_argument("--input_file", default="data/prompts.json")
    parser.add_argument("--output_file", default="outputs/batch_unsloth_outputs.json")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--messages_field", default="messages")
    parser.add_argument("--output_field", default="output", help="Field added to each original row with model output.")
    parser.add_argument("--output_format", choices=["auto", "json", "jsonl"], default="auto")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_repeats", type=int, default=1, help="Generate this many outputs for each input row.")
    parser.add_argument("--always_list_output", action="store_true", help="Always store output_field as a list.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "none", "float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chat_template", default=None, help="Optional Unsloth template name, e.g. llama-3.1 or chatml.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    raw = input_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"No prompt rows found in {input_path}")

    if input_path.suffix.lower() == ".json" or raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"JSON input {input_path} must be a list of objects.")
        rows = data
    else:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {input_path} must be a JSON object.")
            rows.append(obj)

    if not rows:
        raise ValueError(f"No prompt rows found in {input_path}")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Input item {idx} in {input_path} must be a JSON object.")
    return rows


def infer_output_format(output_path: Path, output_format: str) -> str:
    if output_format != "auto":
        return output_format
    return "jsonl" if output_path.suffix.lower() == ".jsonl" else "json"


def write_records(path: str | Path, records: list[dict[str, Any]], output_format: str) -> None:
    output_path = Path(path)
    fmt = infer_output_format(output_path, output_format)
    with output_path.open("w", encoding="utf-8") as handle:
        if fmt == "jsonl":
            for item in records:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


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


def batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def build_work_items(texts: list[str], num_repeats: int) -> list[tuple[int, str]]:
    if num_repeats < 1:
        raise ValueError("--num_repeats must be >= 1")
    work_items: list[tuple[int, str]] = []
    for row_idx, text in enumerate(texts):
        for _ in range(num_repeats):
            work_items.append((row_idx, text))
    return work_items


def main() -> None:
    args = parse_args()
    if args.num_repeats < 1:
        raise SystemExit("--num_repeats must be >= 1")

    import torch

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_records(args.input_file)
    print(f"Loaded {len(rows)} prompts from {args.input_file}")
    print(f"Model or adapter: {args.model_name_or_path}")
    print_cuda_info()

    try:
        model, tokenizer = load_unsloth_model(
            args.model_name_or_path,
            max_seq_length=args.max_length,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
        )
        tokenizer = apply_chat_template_if_requested(tokenizer, args.chat_template)
        ensure_pad_token(tokenizer)
        tokenizer.padding_side = "left"
        enable_unsloth_inference(model)
    except Exception as exc:
        raise SystemExit(
            f"Failed to load Unsloth model/adapter '{args.model_name_or_path}'. "
            "Check the path, CUDA, Unsloth installation, HuggingFace permissions, and local cache. "
            f"Original error: {exc}"
        ) from exc

    texts = [
        _apply_chat_template(
            tokenizer,
            row_to_messages(row, args.prompt_field, args.messages_field, args.system_prompt),
            add_generation_prompt=True,
        )
        for row in rows
    ]
    work_items = build_work_items(texts, args.num_repeats)
    outputs_by_row: list[list[str]] = [[] for _ in rows]
    do_sample = args.temperature > 0

    for start, chunk in batched(work_items, args.batch_size):
        chunk_row_indices = [row_idx for row_idx, _ in chunk]
        chunk_texts = [text for _, text in chunk]
        inputs = tokenizer(chunk_texts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
            if args.min_p is not None:
                generation_kwargs["min_p"] = args.min_p

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        new_token_ids = generated[:, inputs["input_ids"].shape[-1] :]
        responses = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)

        for row_idx, response in zip(chunk_row_indices, responses):
            outputs_by_row[row_idx].append(response.strip())
        print(f"Processed generations {min(start + len(chunk), len(work_items))}/{len(work_items)}", flush=True)

    results: list[dict[str, Any]] = []
    for row, row_outputs in zip(rows, outputs_by_row):
        output = dict(row)
        output[args.output_field] = row_outputs if args.num_repeats > 1 or args.always_list_output else row_outputs[0]
        results.append(output)

    write_records(output_path, results, args.output_format)
    print(f"Wrote {len(results)} rows to {output_path}")


if __name__ == "__main__":
    main()
