"""ELM metric evaluation utilities for SFT training and inference."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # Keep metric helpers importable in lightweight environments.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

from llm_lab.data import _apply_chat_template

SCORE_RE = re.compile(r"My attitude score toward this proposal is:\s*(11|10|[1-9])", re.IGNORECASE)
FALLBACK_SCORE_RE = re.compile(r"\b(?:1[01]|[1-9])\b")
REQUIRED_CONDITION_KEYS = [
    ("high", "high", "strong"),
    ("high", "high", "weak"),
    ("high", "low", "strong"),
    ("high", "low", "weak"),
    ("low", "high", "strong"),
    ("low", "high", "weak"),
    ("low", "low", "strong"),
    ("low", "low", "weak"),
]


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_likert_score(text: Any) -> int | None:
    """Parse the 1-11 attitude score from a model output string."""
    if not isinstance(text, str):
        return None
    match = SCORE_RE.search(text)
    candidates = [match.group(1)] if match else FALLBACK_SCORE_RE.findall(text)
    if not candidates:
        return None
    score = int(candidates[-1])
    return score if 1 <= score <= 11 else None


def normalize_score(score: int) -> float:
    """Normalize an 11-point Likert score to the [0, 1] interval."""
    return (score - 1) / 10


def get_pred_score(item: dict[str, Any]) -> int | None:
    """Use parsed_score first, then fall back to regex parsing model_output."""
    parsed_score = item.get("parsed_score")
    if isinstance(parsed_score, int) and 1 <= parsed_score <= 11:
        return parsed_score
    if isinstance(parsed_score, float) and parsed_score.is_integer() and 1 <= int(parsed_score) <= 11:
        return int(parsed_score)
    output = item.get("model_output", "")
    if isinstance(output, list):
        output = output[0] if output else ""
    return parse_likert_score(output)


def parse_claim_id(item: dict[str, Any]) -> int:
    if item.get("claim_id") is not None:
        return int(item["claim_id"])
    prompt_id = str(item.get("prompt_id", ""))
    match = re.search(r"claim0*(\d+)", prompt_id, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse claim_id from prompt_id: {prompt_id}")
    return int(match.group(1))


def parse_condition_key(item: dict[str, Any]) -> tuple[str, str, str]:
    condition = item.get("condition")
    if isinstance(condition, dict):
        involvement = condition.get("involvement")
        source_expertise = condition.get("source_expertise")
        argument_quality = condition.get("argument_quality")
        if involvement and source_expertise and argument_quality:
            return (str(involvement).lower(), str(source_expertise).lower(), str(argument_quality).lower())

    prompt_id = str(item.get("prompt_id", ""))
    inv_m = re.search(r"(high|low)Inv", prompt_id, re.IGNORECASE)
    src_m = re.search(r"(high|low)Expert", prompt_id, re.IGNORECASE)
    arg_m = re.search(r"(strong|weak)Arg", prompt_id, re.IGNORECASE)
    if not (inv_m and src_m and arg_m):
        raise ValueError(f"Cannot parse condition from prompt_id: {prompt_id}")
    return (inv_m.group(1).lower(), src_m.group(1).lower(), arg_m.group(1).lower())


def compute_elm_stats(predictions: list[dict[str, Any]], prediction_file: str | Path | None = None) -> dict[str, Any]:
    """Compute claim-level and summary ELM metrics from prediction rows."""
    score_groups: dict[int, dict[tuple[str, str, str], list[float]]] = {}
    invalid_items: list[dict[str, Any]] = []

    for item in predictions:
        try:
            claim_id = parse_claim_id(item)
            condition_key = parse_condition_key(item)
            pred_score = get_pred_score(item)
            if pred_score is None:
                invalid_items.append({
                    "prompt_id": item.get("prompt_id"),
                    "reason": "Cannot parse prediction score",
                    "model_output": item.get("model_output"),
                })
                continue
            score_groups.setdefault(claim_id, {}).setdefault(condition_key, []).append(normalize_score(pred_score))
        except Exception as exc:  # noqa: BLE001 - keep row-level failures in output JSON.
            invalid_items.append({
                "prompt_id": item.get("prompt_id"),
                "reason": str(exc),
                "model_output": item.get("model_output"),
            })

    claim_stats: list[dict[str, Any]] = []
    incomplete_claims: list[dict[str, Any]] = []
    for claim_id in sorted(score_groups):
        cond_scores = score_groups[claim_id]
        missing_keys = [key for key in REQUIRED_CONDITION_KEYS if key not in cond_scores or not cond_scores[key]]
        if missing_keys:
            incomplete_claims.append({
                "claim_id": claim_id,
                "missing_conditions": [list(key) for key in missing_keys],
            })
            continue

        s = {key: mean(cond_scores[key]) for key in REQUIRED_CONDITION_KEYS}
        HHs, HHw = s[("high", "high", "strong")], s[("high", "high", "weak")]
        HLs, HLw = s[("high", "low", "strong")], s[("high", "low", "weak")]
        LHs, LHw = s[("low", "high", "strong")], s[("low", "high", "weak")]
        LLs, LLw = s[("low", "low", "strong")], s[("low", "low", "weak")]

        D_H_Arg = mean([HHs - HHw, HLs - HLw])
        D_L_Arg = mean([LHs - LHw, LLs - LLw])
        Delta_Arg = D_H_Arg - D_L_Arg
        D_H_Src = mean([HHs - HLs, HHw - HLw])
        D_L_Src = mean([LHs - LLs, LHw - LLw])
        Delta_Src = D_L_Src - D_H_Src

        claim_stats.append({
            "claim_id": claim_id,
            "HHs": HHs,
            "HHw": HHw,
            "HLs": HLs,
            "HLw": HLw,
            "LHs": LHs,
            "LHw": LHw,
            "LLs": LLs,
            "LLw": LLw,
            "D_H_Arg": D_H_Arg,
            "D_L_Arg": D_L_Arg,
            "Delta_Arg": Delta_Arg,
            "D_H_Src": D_H_Src,
            "D_L_Src": D_L_Src,
            "Delta_Src": Delta_Src,
            "Arg_ELM_pass": Delta_Arg > 0,
            "Src_ELM_pass": Delta_Src > 0,
            "Both_ELM_pass": Delta_Arg > 0 and Delta_Src > 0,
        })

    if not claim_stats:
        raise RuntimeError("No complete claims found. Please check whether predictions contain all 8 conditions for each claim.")

    summary = {
        "prediction_file": str(prediction_file) if prediction_file is not None else None,
        "num_prediction_items": len(predictions),
        "num_valid_prediction_items": len(predictions) - len(invalid_items),
        "num_invalid_prediction_items": len(invalid_items),
        "num_complete_claims": len(claim_stats),
        "num_incomplete_claims": len(incomplete_claims),
        "D_H_Arg": mean(x["D_H_Arg"] for x in claim_stats),
        "D_L_Arg": mean(x["D_L_Arg"] for x in claim_stats),
        "Delta_Arg": mean(x["Delta_Arg"] for x in claim_stats),
        "D_H_Src": mean(x["D_H_Src"] for x in claim_stats),
        "D_L_Src": mean(x["D_L_Src"] for x in claim_stats),
        "Delta_Src": mean(x["Delta_Src"] for x in claim_stats),
        "Arg_ELM_pass_rate": mean(1 if x["Arg_ELM_pass"] else 0 for x in claim_stats),
        "Src_ELM_pass_rate": mean(1 if x["Src_ELM_pass"] else 0 for x in claim_stats),
        "Both_ELM_pass_rate": mean(1 if x["Both_ELM_pass"] else 0 for x in claim_stats),
    }
    return {
        "summary": summary,
        "claim_stats": claim_stats,
        "incomplete_claims": incomplete_claims,
        "invalid_items": invalid_items,
    }


def generation_prompt_from_row(tokenizer: Any, row: dict[str, Any], system_prompt: str, prompt_field: str = "prompt") -> str:
    """Render system/user context for generation while dropping assistant gold answers."""
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt_messages = [dict(message) for message in messages]
        if prompt_messages and str(prompt_messages[-1].get("role", "")).strip().lower() in {"assistant", "respondent"}:
            prompt_messages = prompt_messages[:-1]
    else:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row[prompt_field]},
        ]
    return _apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)


def left_pad_tokenize(tokenizer: Any, texts: list[str], device: Any = "cuda") -> dict[str, Any]:
    """Manually left-pad pre-rendered prompts for decoder-only batch generation."""
    import torch

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for batch inference padding.")
    encoded = [tokenizer(text, add_special_tokens=False, return_attention_mask=True) for text in texts]
    max_len = max(len(item["input_ids"]) for item in encoded)
    input_ids, attention_mask = [], []
    for item in encoded:
        ids = item["input_ids"]
        mask = item.get("attention_mask", [1] * len(ids))
        pad_len = max_len - len(ids)
        input_ids.append([pad_token_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + mask)
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    return {key: value.to(device) for key, value in batch.items()}


def batch_generate_predictions(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    system_prompt: str,
    keep_fields: list[str],
    prompt_field: str = "prompt",
    response_field: str = "groundtruth",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 1,
    num_repeats: int = 1,
    device: Any = "cuda",
    progress_prefix: str = "",
) -> list[dict[str, Any]]:
    """Run batch generation and attach model_output / parsed_score to each row."""
    import torch

    if not rows:
        return []

    old_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    was_training = getattr(model, "training", False)
    model.eval()
    try:
        results = [{field: row[field] for field in keep_fields if field in row} for row in rows]
        pending: list[tuple[int, str]] = []
        for row_idx, row in enumerate(rows):
            prompt = generation_prompt_from_row(tokenizer, row, system_prompt, prompt_field=prompt_field)
            for _ in range(num_repeats):
                pending.append((row_idx, prompt))

        outputs_by_row: list[list[str]] = [[] for _ in rows]
        do_sample = temperature > 0
        with torch.no_grad():
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                row_ids = [row_id for row_id, _ in batch]
                texts = [text for _, text in batch]
                inputs = left_pad_tokenize(tokenizer, texts, device=device)
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "use_cache": True,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if do_sample:
                    generation_kwargs.update({"temperature": temperature, "top_p": top_p})
                outputs = model.generate(**inputs, **generation_kwargs)
                new_tokens = outputs[:, inputs["input_ids"].shape[-1]:]
                decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for row_id, output_text in zip(row_ids, decoded):
                    outputs_by_row[row_id].append(output_text.strip())
                prefix = f"{progress_prefix} " if progress_prefix else ""
                print(f"{prefix}Processed {min(start + len(batch), len(pending))}/{len(pending)} generations")

        for row, outs in zip(results, outputs_by_row):
            first_output = outs[0] if outs else ""
            row["model_output"] = outs if num_repeats > 1 else first_output
            row["parsed_score"] = parse_likert_score(first_output)
            row["is_exact_match"] = first_output.strip() == str(row.get(response_field, "")).strip()
        return results
    finally:
        if old_padding_side is not None:
            tokenizer.padding_side = old_padding_side
        if was_training:
            model.train()


def elm_improved(current: dict[str, Any] | None, previous: dict[str, Any] | None) -> bool:
    """Return True only when Delta_Arg/Delta_Src improve without either metric dropping."""
    if current is None:
        return False
    if previous is None:
        return True
    cur_arg, cur_src = current["Delta_Arg"], current["Delta_Src"]
    prev_arg, prev_src = previous["Delta_Arg"], previous["Delta_Src"]
    return (cur_arg > prev_arg and cur_src >= prev_src) or (cur_src > prev_src and cur_arg >= prev_arg)


def save_best_checkpoint(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


class EpochELMEvalCallback(TrainerCallback):
    """Run ELM train/validation batch inference at each epoch end and save best adapter."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        train_rows: list[dict[str, Any]],
        valid_rows: list[dict[str, Any]],
        output_dir: str | Path,
        best_checkpoint_dir: str | Path,
        system_prompt: str,
        keep_fields: list[str],
        prompt_field: str = "prompt",
        response_field: str = "groundtruth",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 1,
        num_repeats: int = 1,
        device: Any = "cuda",
    ) -> None:
        self.tokenizer = tokenizer
        self.train_rows = train_rows
        self.valid_rows = valid_rows
        self.output_dir = Path(output_dir)
        self.best_checkpoint_dir = Path(best_checkpoint_dir)
        self.system_prompt = system_prompt
        self.keep_fields = keep_fields
        self.prompt_field = prompt_field
        self.response_field = response_field
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.num_repeats = num_repeats
        self.device = device
        self.previous_train_summary: dict[str, Any] | None = None
        self.previous_valid_summary: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):  # noqa: ANN001 - HF callback signature.
        if model is None:
            return control
        epoch = int(round(state.epoch or len(self.history) + 1))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== ELM evaluation after epoch {epoch} =====")
        train_predictions = batch_generate_predictions(
            model,
            self.tokenizer,
            self.train_rows,
            system_prompt=self.system_prompt,
            keep_fields=self.keep_fields,
            prompt_field=self.prompt_field,
            response_field=self.response_field,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            batch_size=self.batch_size,
            num_repeats=self.num_repeats,
            device=self.device,
            progress_prefix=f"epoch {epoch} train",
        )
        valid_predictions = batch_generate_predictions(
            model,
            self.tokenizer,
            self.valid_rows,
            system_prompt=self.system_prompt,
            keep_fields=self.keep_fields,
            prompt_field=self.prompt_field,
            response_field=self.response_field,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            batch_size=self.batch_size,
            num_repeats=self.num_repeats,
            device=self.device,
            progress_prefix=f"epoch {epoch} valid",
        )

        train_pred_path = self.output_dir / f"epoch_{epoch}_train_predictions.json"
        valid_pred_path = self.output_dir / f"epoch_{epoch}_valid_predictions.json"
        train_stats_path = self.output_dir / f"epoch_{epoch}_train_elm_stats.json"
        valid_stats_path = self.output_dir / f"epoch_{epoch}_valid_elm_stats.json"
        save_json(train_predictions, train_pred_path)
        save_json(valid_predictions, valid_pred_path)

        train_stats = compute_elm_stats(train_predictions, prediction_file=train_pred_path)
        valid_stats = compute_elm_stats(valid_predictions, prediction_file=valid_pred_path)
        save_json(train_stats, train_stats_path)
        save_json(valid_stats, valid_stats_path)

        train_summary = train_stats["summary"]
        valid_summary = valid_stats["summary"]
        valid_improved = elm_improved(valid_summary, self.previous_valid_summary)
        train_improved = elm_improved(train_summary, self.previous_train_summary)
        saved_as_best = bool(valid_improved or train_improved)
        save_reason = "validation improved" if valid_improved else ("train improved" if train_improved else "not improved")
        if saved_as_best:
            save_best_checkpoint(model, self.tokenizer, self.best_checkpoint_dir)

        record = {
            "epoch": epoch,
            "train_summary": train_summary,
            "valid_summary": valid_summary,
            "train_improved": train_improved,
            "valid_improved": valid_improved,
            "saved_as_best": saved_as_best,
            "save_reason": save_reason,
        }
        self.history.append(record)
        save_json(self.history, self.output_dir / "elm_training_history.json")
        self.previous_train_summary = train_summary
        self.previous_valid_summary = valid_summary

        print(f"Epoch {epoch}")
        print(f"train Delta_Arg={train_summary['Delta_Arg']:.6f}, Delta_Src={train_summary['Delta_Src']:.6f}")
        print(f"valid Delta_Arg={valid_summary['Delta_Arg']:.6f}, Delta_Src={valid_summary['Delta_Src']:.6f}")
        print(f"saved_as_best={saved_as_best}; save_reason={save_reason}")
        print(f"Best checkpoint dir: {self.best_checkpoint_dir if saved_as_best else 'unchanged'}")
        return control
