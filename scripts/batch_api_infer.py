#!/usr/bin/env python
"""Batch inference against an OpenAI-compatible chat/completions endpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch prompts through a deployed OpenAI-compatible LLM service."
    )
    parser.add_argument("--base_url", required=True, help="Endpoint root, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True, help="Served model name used by the deployment.")
    parser.add_argument("--input_file", required=True, help="JSONL input file with prompt or messages per line.")
    parser.add_argument("--output_file", required=True, help="JSONL output file; one result per input row.")
    parser.add_argument("--prompt_field", default="prompt", help="Field name containing a raw user prompt.")
    parser.add_argument("--messages_field", default="messages", help="Field name containing OpenAI-style messages.")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api_key", default=None, help="Defaults to OPENAI_API_KEY, or 'EMPTY' if unset.")
    parser.add_argument("--id_field", default="id", help="Optional input id field copied to output.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it exists.")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} must be a JSON object.")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def row_to_messages(row: dict[str, Any], prompt_field: str, messages_field: str, system_prompt: str | None) -> list[dict[str, str]]:
    if messages_field in row:
        messages = row[messages_field]
        if not isinstance(messages, list):
            raise ValueError(f"{messages_field} must be a list of messages.")
        return messages

    prompt = row.get(prompt_field)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Each row must contain a non-empty '{prompt_field}' string or '{messages_field}' list.")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def post_chat_completion(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    messages = row_to_messages(row, args.prompt_field, args.messages_field, args.system_prompt)
    payload = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        return {
            "ok": True,
            "response": content,
            "raw_response": data,
            "latency_sec": round(time.time() - started, 4),
        }
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"HTTP {exc.code}: {detail}", "latency_sec": round(time.time() - started, 4)}
    except Exception as exc:  # network/service errors should be captured per row
        return {"ok": False, "error": repr(exc), "latency_sec": round(time.time() - started, 4)}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.input_file)
    print(f"Loaded {len(rows)} rows from {args.input_file}")
    print(f"Sending requests to {args.base_url.rstrip('/')}/chat/completions with concurrency={args.concurrency}")

    results: list[dict[str, Any] | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(post_chat_completion, args, row): idx for idx, row in enumerate(rows)}
        for done_count, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            row = rows[idx]
            result = future.result()
            output = {"index": idx, **result}
            if args.id_field in row:
                output[args.id_field] = row[args.id_field]
            if args.prompt_field in row:
                output[args.prompt_field] = row[args.prompt_field]
            results[idx] = output
            status = "ok" if result.get("ok") else "error"
            print(f"[{done_count}/{len(rows)}] row={idx} {status}", flush=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
