"""SFT data loading utilities for JSON/JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message must contain string 'role' and 'content' fields.")
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def _read_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    raw = data_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"No training examples found in {data_path}")

    if data_path.suffix.lower() == ".json" or raw.startswith("["):
        loaded = json.loads(raw)
        if not isinstance(loaded, list):
            raise ValueError(f"JSON training file must contain a list of objects: {data_path}")
        rows = loaded
    else:
        rows = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {data_path}: {exc}") from exc
            rows.append(obj)

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Training example {idx} in {data_path} must be a JSON object.")
    return rows


def example_to_text(
    example: dict[str, Any],
    tokenizer: Any,
    prompt_field: str = "prompt",
    response_field: str = "groundtruth",
) -> str:
    """Convert one supported JSON object to a training text string."""
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text

    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        return _format_messages(tokenizer, messages)

    prompt = example.get(prompt_field)
    response = example.get(response_field)
    if isinstance(prompt, str) and prompt.strip() and isinstance(response, str) and response.strip():
        return _format_messages(
            tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
        )

    raise ValueError(
        "Unsupported data row. Expected {'text': '...'}, "
        "{'messages': [...]}, or prompt/response fields such as "
        f"'{prompt_field}' and '{response_field}'."
    )


def load_sft_dataset(
    path: str | Path,
    tokenizer: Any,
    prompt_field: str = "prompt",
    response_field: str = "groundtruth",
    split_field: str | None = None,
    split: str | None = None,
):
    """Load JSON/JSONL SFT data and return a Dataset with a ``text`` column."""
    from datasets import Dataset

    raw_rows = _read_json_or_jsonl(path)
    if split_field and split is not None:
        raw_rows = [row for row in raw_rows if row.get(split_field) == split]
        if not raw_rows:
            raise ValueError(f"No examples found with {split_field}={split!r} in {path}")

    rows: list[dict[str, str]] = []
    for idx, obj in enumerate(raw_rows):
        try:
            rows.append({"text": example_to_text(obj, tokenizer, prompt_field, response_field)})
        except ValueError as exc:
            raise ValueError(f"Invalid SFT example at index {idx} of {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"No training examples found in {path}")
    return Dataset.from_list(rows)


def load_sft_jsonl(path: str | Path, tokenizer: Any):
    """Backward-compatible wrapper for older scripts."""
    return load_sft_dataset(path, tokenizer)
