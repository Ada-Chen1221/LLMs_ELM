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
    parser.add_argument("--model_name_or_path", default="models/Qwen3-4B-Instruct-2507", help="Base model path or HuggingFace repo id.")
    parser.add_argument("--adapter_path", default=None, help="Optional LoRA/QLoRA adapter directory to load on top of the base model.")
    parser.add_argument(
        "--merge_and_unload",
        action="store_true",
        help="Merge the LoRA adapter into the base model in memory before generation. Requires --adapter_path.",
    )
    parser.add_argument("--input_file", default="data/prompts.json")
    parser.add_argument("--output_file", default="outputs/batch_outputs.json")
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--messages_field", default="messages")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--output_field", default="output", help="Field added to each original row with the model output.")
    parser.add_argument("--num_repeats", type=int, default=1, help="Generate this many outputs for each input row.")
    parser.add_argument("--always_list_output", action="store_true", help="Always store output_field as a list, even when num_repeats=1.")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch random seed for reproducible sampling.")
    parser.add_argument("--output_format", choices=["auto", "json", "jsonl"], default="auto")
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


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either a JSON array file or a JSONL file into a list of objects."""
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


def build_work_items(rows: list[dict[str, Any]], texts: list[str], num_repeats: int) -> list[tuple[int, str]]:
    if num_repeats < 1:
        raise ValueError("--num_repeats must be >= 1")
    work_items: list[tuple[int, str]] = []
    for row_idx, text in enumerate(texts):
        for _ in range(num_repeats):
            work_items.append((row_idx, text))
    return work_items


def main() -> None:
    args = parse_args()
    if args.merge_and_unload and not args.adapter_path:
        raise SystemExit("--merge_and_unload requires --adapter_path.")
    if args.num_repeats < 1:
        raise SystemExit("--num_repeats must be >= 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("Missing peft dependency for --adapter_path. Install with: pip install peft") from exc
    else:
        PeftModel = None
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
    print(f"Base model: {args.model_name_or_path}")
    if args.adapter_path:
        print(f"LoRA/QLoRA adapter: {args.adapter_path}")
        print(f"Merge adapter in memory: {args.merge_and_unload}")
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
    if args.adapter_path:
        print("Loading LoRA/QLoRA adapter...", flush=True)
        model = PeftModel.from_pretrained(model, args.adapter_path)
        if args.merge_and_unload:
            print("Merging LoRA/QLoRA adapter into base model in memory...", flush=True)
            model = model.merge_and_unload()
    model.eval()

    texts = [
        render_text(
            tokenizer,
            row_to_messages(row, args.prompt_field, args.messages_field, args.system_prompt),
        )
        for row in rows
    ]
    work_items = build_work_items(rows, texts, args.num_repeats)
    outputs_by_row: list[list[str]] = [[] for _ in rows]
    do_sample = args.temperature > 0
    for start, chunk in batched(work_items, args.batch_size):
        chunk_row_indices = [row_idx for row_idx, _ in chunk]
        chunk_texts = [text for _, text in chunk]
        model_inputs = tokenizer(chunk_texts, return_tensors="pt", padding=True)
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
