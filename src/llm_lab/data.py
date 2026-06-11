"""SFT data loading utilities for JSON/JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message must contain string 'role' and 'content' fields.")
        rendered.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered.append("assistant:")
    return "\n".join(rendered)


def _format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return _apply_chat_template(tokenizer, messages, add_generation_prompt=False)


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


def _split_messages_for_response_loss(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not messages:
        raise ValueError("'messages' must contain at least one assistant response for response-only loss.")
    last_message = messages[-1]
    if last_message.get("role") != "assistant" or not isinstance(last_message.get("content"), str):
        raise ValueError("For response-only loss, the final message must be an assistant message with string content.")
    prompt_messages = messages[:-1]
    if not prompt_messages:
        raise ValueError("For response-only loss, messages must include at least one prompt message before the assistant response.")
    return prompt_messages, last_message


def _prompt_response_to_texts(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    assistant_message: dict[str, str],
) -> tuple[str, str]:
    """Return ``(full_text, prompt_text)`` for response-only supervised loss."""
    prompt_text = _apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    full_messages = [*prompt_messages, assistant_message]
    full_text = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

    # Most chat templates make full_text start with prompt_text. Keep an explicit
    # fallback for simple/manual templates so label masking still works.
    if not full_text.startswith(prompt_text):
        content = assistant_message.get("content", "")
        full_text = f"{prompt_text}{content}"
    return full_text, prompt_text


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


def example_to_response_only_texts(
    example: dict[str, Any],
    tokenizer: Any,
    prompt_field: str = "prompt",
    response_field: str = "groundtruth",
) -> tuple[str, str]:
    """Convert one example to ``(full_text, prompt_text)`` for response-only loss.

    ``prompt_text`` is the prefix that should be masked out in labels. The loss
    is therefore computed only on tokens after this prefix (the assistant answer,
    including any assistant/end-of-turn template tokens).
    """
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        prompt_messages, assistant_message = _split_messages_for_response_loss(messages)
        return _prompt_response_to_texts(tokenizer, prompt_messages, assistant_message)

    prompt = example.get(prompt_field)
    response = example.get(response_field)
    if isinstance(prompt, str) and prompt.strip() and isinstance(response, str) and response.strip():
        return _prompt_response_to_texts(
            tokenizer,
            [{"role": "user", "content": prompt}],
            {"role": "assistant", "content": response},
        )

    text = example.get("text")
    if isinstance(text, str) and text.strip():
        # Legacy free-form rows do not identify which tokens are prompt vs
        # answer, so we cannot mask the prompt safely. Keep them trainable for
        # backward compatibility. Prefer prompt/groundtruth or messages data.
        return text, ""

    raise ValueError(
        "Unsupported data row. Expected {'messages': [...]}, prompt/response fields such as "
        f"'{prompt_field}' and '{response_field}', or legacy {'text': '...'} rows."
    )


def load_sft_dataset(
    path: str | Path,
    tokenizer: Any,
    prompt_field: str = "prompt",
    response_field: str = "groundtruth",
    split_field: str | None = None,
    split: str | None = None,
    response_only_loss: bool = True,
):
    """Load JSON/JSONL SFT data.

    By default, the returned Dataset includes both ``text`` and ``prompt_text``
    columns so the training collator can mask prompt tokens and compute loss
    only on the assistant response / groundtruth.
    """
    from datasets import Dataset

    raw_rows = _read_json_or_jsonl(path)
    if split_field and split is not None:
        raw_rows = [row for row in raw_rows if row.get(split_field) == split]
        if not raw_rows:
            raise ValueError(f"No examples found with {split_field}={split!r} in {path}")

    rows: list[dict[str, str]] = []
    for idx, obj in enumerate(raw_rows):
        try:
            if response_only_loss:
                text, prompt_text = example_to_response_only_texts(obj, tokenizer, prompt_field, response_field)
                rows.append({"text": text, "prompt_text": prompt_text})
            else:
                rows.append({"text": example_to_text(obj, tokenizer, prompt_field, response_field)})
        except ValueError as exc:
            raise ValueError(f"Invalid SFT example at index {idx} of {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"No training examples found in {path}")
    return Dataset.from_list(rows)


def load_sft_jsonl(path: str | Path, tokenizer: Any):
    """Backward-compatible wrapper for older scripts."""
    return load_sft_dataset(path, tokenizer)


def _prompt_to_grpo_prompt(example: dict[str, Any], prompt_field: str = "prompt") -> str | list[dict[str, str]]:
    """Extract a TRL GRPO prompt from supported JSON rows."""
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        if messages[-1].get("role") == "assistant":
            messages = messages[:-1]
        return messages

    prompt = example.get(prompt_field)
    if isinstance(prompt, str) and prompt.strip():
        return prompt

    text = example.get("text")
    if isinstance(text, str) and text.strip():
        return text

    raise ValueError(f"Expected a non-empty '{prompt_field}', 'messages', or 'text' prompt.")


def load_grpo_dataset(
    path: str | Path,
    prompt_field: str = "prompt",
    answer_field: str = "groundtruth",
    split_field: str | None = None,
    split: str | None = None,
):
    """Load JSON/JSONL data for GRPO-style RL.

    The returned Dataset has a ``prompt`` column for TRL's ``GRPOTrainer`` and an
    ``answer`` column that can be consumed by reward functions.
    """
    from datasets import Dataset

    raw_rows = _read_json_or_jsonl(path)
    if split_field and split is not None:
        raw_rows = [row for row in raw_rows if row.get(split_field) == split]
        if not raw_rows:
            raise ValueError(f"No examples found with {split_field}={split!r} in {path}")

    rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(raw_rows):
        try:
            prompt = _prompt_to_grpo_prompt(obj, prompt_field=prompt_field)
            answer = obj.get(answer_field)
            if answer is None:
                messages = obj.get("messages")
                if isinstance(messages, list) and messages and messages[-1].get("role") == "assistant":
                    answer = messages[-1].get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Expected a non-empty '{answer_field}' answer or final assistant message.")
            rows.append({"prompt": prompt, "answer": answer})
        except ValueError as exc:
            raise ValueError(f"Invalid GRPO example at index {idx} of {path}: {exc}") from exc

    if not rows:
        raise ValueError(f"No RL training examples found in {path}")
    return Dataset.from_list(rows)
