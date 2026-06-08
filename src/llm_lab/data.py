"""JSONL SFT data loading utilities."""

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


def example_to_text(example: dict[str, Any], tokenizer: Any) -> str:
    """Convert one supported JSON object to a training text string."""
    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text

    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        return _format_messages(tokenizer, messages)

    raise ValueError(
        "Unsupported data row. Expected {'text': '...'} or "
        "{'messages': [{'role': 'user', 'content': '...'}, ...]}."
    )


def load_sft_jsonl(path: str | Path, tokenizer: Any):
    """Load a JSONL file and return a HuggingFace Dataset with a ``text`` column."""
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Please install datasets: pip install datasets") from exc

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    rows: list[dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {data_path}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} of {data_path} must be a JSON object.")
            try:
                rows.append({"text": example_to_text(obj, tokenizer)})
            except ValueError as exc:
                raise ValueError(f"Invalid SFT example on line {line_no} of {data_path}: {exc}") from exc

    if not rows:
        raise ValueError(f"No training examples found in {data_path}")
    return Dataset.from_list(rows)
