"""Claim-group REINFORCE training utilities for post-SFT ELM alignment."""

from __future__ import annotations

from collections import Counter
import json
import math
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


from llm_lab.data import _apply_chat_template, _read_json_or_jsonl
from llm_lab.elm_eval import (
    REQUIRED_CONDITION_KEYS,
    batch_generate_predictions,
    compute_elm_stats,
    parse_likert_score,
    save_json,
)

MAX_REGEN_ATTEMPTS = 3
RETRY_FAIL_PENALTY = 0.2
PARSE_FAIL_GROUP_REWARD = -8.0
CONDITION_LABELS = {
    ("high", "high", "strong"): "HHs",
    ("high", "high", "weak"): "HHw",
    ("high", "low", "strong"): "HLs",
    ("high", "low", "weak"): "HLw",
    ("low", "high", "strong"): "LHs",
    ("low", "high", "weak"): "LHw",
    ("low", "low", "strong"): "LLs",
    ("low", "low", "weak"): "LLw",
}


@dataclass
class RLConfig:
    model_name_or_path: str = "models/Qwen3-4B"
    train_file: str = "data/processed_data/processed_train_messages.json"
    output_dir: str = "outputs/qwen3_4b_elm_reinforce"
    num_train_epochs: int = 1
    groups_per_step: int = 1
    rollouts_per_group: int = 2
    learning_rate: float = 1e-6
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_new_tokens: int = 32
    temperature: float = 1.0
    top_p: float = 0.95
    max_regen_attempts: int = MAX_REGEN_ATTEMPTS
    retry_fail_penalty: float = RETRY_FAIL_PENALTY
    parse_fail_group_reward: float = PARSE_FAIL_GROUP_REWARD
    logprob_reduction: str = "sum"
    global_reward_baseline: float = 0.0
    reward_baseline_mode: str = "global"
    seed: int = 3407
    dtype: str = "auto"
    load_in_4bit: bool = False
    use_gradient_checkpointing: bool = True
    save_every_epoch: bool = True
    verbose: bool = True
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    resume_lora_path: str | None = None
    print_rollout_details: bool = False
    print_generation_progress: bool = False
    max_completion_print_chars: int = 500
    print_prompt_details: bool = True
    max_prompt_print_chars: int = 2000
    valid_file: str | None = None
    rl_valid_eval_dir: str | None = None
    best_rl_checkpoint_dir: str | None = None
    valid_eval_batch_size: int = 80
    valid_eval_max_new_tokens: int = 32
    valid_eval_temperature: float = 0.0
    valid_eval_top_p: float = 1.0
    valid_delta_arg_drop_tolerance: float = 0.03


def condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    condition = row.get("condition") or {}
    return (
        str(condition.get("involvement", "")).lower(),
        str(condition.get("source_expertise", "")).lower(),
        str(condition.get("argument_quality", "")).lower(),
    )


def build_complete_claim_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], dict[tuple[str, str, str], dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("prompt_id", "").split("_", 1)[0], row.get("claim_id"))
        grouped.setdefault(key, {})[condition_key(row)] = row
    complete = []
    for (prompt_prefix, claim_id), by_condition in grouped.items():
        if all(key in by_condition for key in REQUIRED_CONDITION_KEYS):
            complete.append({
                "prompt_prefix": prompt_prefix,
                "claim_id": claim_id,
                "items": [by_condition[key] for key in REQUIRED_CONDITION_KEYS],
            })
    return complete


def render_generation_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = [dict(message) for message in row.get("messages", [])]
    if not messages:
        raise ValueError("RL rows must contain messages.")
    if messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    for message in messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role in RL data: {message.get('role')!r}")
    return _apply_chat_template(tokenizer, messages, add_generation_prompt=True)


def penalty_high_inv_arg_effect(diff: int) -> float:
    """高涉入下 strong argument 应明显高于 weak argument；diff = strong - weak。"""
    if diff >= 2:
        return 0.0
    if diff == 1:
        return -2.0
    if diff == 0:
        return -4.0
    return -6.0


def penalty_high_inv_src_effect(diff: int) -> float:
    """高涉入下 source cue 不应该主导，但 highExpert 不能低于 lowExpert。"""
    if diff in {0, 1}:
        return 0.0
    if diff == 2:
        return -0.5
    if diff >= 3:
        return -1.5
    return -1.0


def low_arg_suppression_score(diff: int) -> float:
    """低涉入下 strong - weak 越接近 0 越好；连续 shaping score。"""
    abs_diff = abs(diff)
    if abs_diff == 0:
        return 2.5
    if abs_diff == 1:
        return 0.8
    if abs_diff == 2:
        return -0.3
    if abs_diff == 3:
        return -0.8
    if abs_diff == 4:
        return -1.3
    return -2.0


def low_src_effect_score(diff: int) -> float:
    """低涉入下 highExpert - lowExpert 应明显为正；source cue 为辅助项。"""
    if diff >= 2:
        return 0.8
    if diff == 1:
        return -0.4
    if diff == 0:
        return -0.8
    return -1.2


def detect_score_collapse(scores: dict[str, int]) -> tuple[bool, float, dict[str, Any]]:
    values = list(scores.values())
    score_range = max(values) - min(values)
    counts = Counter(values)
    unique_scores = len(counts)
    most_common_score, max_count = counts.most_common(1)[0]

    collapsed = False
    penalty = 0.0
    reason = None
    if score_range <= 1:
        collapsed = True
        penalty = -6.0
        reason = "score_range_le_1"
    elif unique_scores <= 2 and max_count >= 5:
        collapsed = True
        penalty = -6.0
        reason = "unique_le_2_and_majority_same"
    elif unique_scores <= 3 and max_count >= 6:
        collapsed = True
        penalty = -6.0
        reason = "unique_le_3_and_strong_majority_same"

    return collapsed, penalty, {
        "score_range": score_range,
        "unique_scores": unique_scores,
        "score_counts": dict(counts),
        "most_common_score": most_common_score,
        "max_count": max_count,
        "collapse_reason": reason,
    }


def compute_group_reward(scores: dict[str, int]) -> tuple[float, dict[str, Any]]:
    HHs, HHw = scores["HHs"], scores["HHw"]
    HLs, HLw = scores["HLs"], scores["HLw"]
    LHs, LHw = scores["LHs"], scores["LHw"]
    LLs, LLw = scores["LLs"], scores["LLw"]

    diffs = {
        "arg_effect_high_inv_high_src": HHs - HHw,
        "arg_effect_high_inv_low_src": HLs - HLw,
        "src_effect_high_inv_strong_arg": HHs - HLs,
        "src_effect_high_inv_weak_arg": HHw - HLw,
        "arg_effect_low_inv_high_src": LHs - LHw,
        "arg_effect_low_inv_low_src": LLs - LLw,
        "src_effect_low_inv_strong_arg": LHs - LLs,
        "src_effect_low_inv_weak_arg": LHw - LLw,
    }

    collapsed, collapse_penalty, collapse_details = detect_score_collapse(scores)
    if collapsed:
        return float(collapse_penalty), {
            "diffs": diffs,
            "collapse_detected": True,
            "collapse_details": collapse_details,
            "high_terms": {},
            "low_arg_terms": {},
            "low_src_terms": {},
            "high_arg_gate_passed": False,
            "low_arg_gate_passed": False,
            "high_reward": 0.0,
            "low_arg_reward": 0.0,
            "low_src_reward": 0.0,
            "reward": collapse_penalty,
        }

    high_terms = {
        "arg_effect_high_inv_high_src": penalty_high_inv_arg_effect(diffs["arg_effect_high_inv_high_src"]),
        "arg_effect_high_inv_low_src": penalty_high_inv_arg_effect(diffs["arg_effect_high_inv_low_src"]),
        "src_effect_high_inv_strong_arg": penalty_high_inv_src_effect(diffs["src_effect_high_inv_strong_arg"]),
        "src_effect_high_inv_weak_arg": penalty_high_inv_src_effect(diffs["src_effect_high_inv_weak_arg"]),
    }
    high_reward = sum(high_terms.values())
    high_arg_gate_passed = (
        diffs["arg_effect_high_inv_high_src"] >= 2
        and diffs["arg_effect_high_inv_low_src"] >= 2
    )
    if not high_arg_gate_passed:
        reward = min(high_reward, -1.0)
        return float(reward), {
            "diffs": diffs,
            "collapse_detected": False,
            "collapse_details": collapse_details,
            "high_terms": high_terms,
            "low_arg_terms": {},
            "low_src_terms": {},
            "high_arg_gate_passed": False,
            "low_arg_gate_passed": False,
            "high_reward": high_reward,
            "low_arg_reward": 0.0,
            "low_src_reward": 0.0,
            "reward": reward,
        }

    low_arg_gate_passed = (
        abs(diffs["arg_effect_low_inv_high_src"]) <= 1
        and abs(diffs["arg_effect_low_inv_low_src"]) <= 1
    )
    low_src_gate_passed = (
        diffs["src_effect_low_inv_strong_arg"] >= 2
        and diffs["src_effect_low_inv_weak_arg"] >= 2
    )

    low_arg_terms = {
        "arg_effect_low_inv_high_src": low_arg_suppression_score(diffs["arg_effect_low_inv_high_src"]),
        "arg_effect_low_inv_low_src": low_arg_suppression_score(diffs["arg_effect_low_inv_low_src"]),
    }
    low_src_terms = {
        "src_effect_low_inv_strong_arg": low_src_effect_score(diffs["src_effect_low_inv_strong_arg"]),
        "src_effect_low_inv_weak_arg": low_src_effect_score(diffs["src_effect_low_inv_weak_arg"]),
    }
    low_arg_reward = sum(low_arg_terms.values())
    low_src_reward = sum(low_src_terms.values())
    reward = high_reward + low_arg_reward + low_src_reward
    return float(reward), {
        "diffs": diffs,
        "collapse_detected": False,
        "collapse_details": collapse_details,
        "high_terms": high_terms,
        "low_arg_terms": low_arg_terms,
        "low_src_terms": low_src_terms,
        "high_arg_gate_passed": True,
        "low_arg_gate_passed": low_arg_gate_passed,
        "low_src_gate_passed": low_src_gate_passed,
        "high_reward": high_reward,
        "low_arg_reward": low_arg_reward,
        "low_src_reward": low_src_reward,
        "reward": reward,
    }


def _tokenize(tokenizer: Any, texts: list[str], device: Any) -> dict[str, Any]:
    import torch
    old_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        if old_side is not None:
            tokenizer.padding_side = old_side
    return {k: v.to(device) for k, v in batch.items()}


def generate_one(model: Any, tokenizer: Any, prompt: str, cfg: RLConfig, device: Any) -> tuple[str, Any]:
    inputs = _tokenize(tokenizer, [prompt], device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    completion_ids = outputs[0, inputs["input_ids"].shape[-1]:].detach()
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text, completion_ids.cpu()


def generate_many(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    cfg: RLConfig,
    device: Any,
) -> list[tuple[str, Any]]:
    """Generate completions for multiple prompts in one decoder-only batch."""
    if not prompts:
        return []
    inputs = _tokenize(tokenizer, prompts, device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[:, prompt_len:].detach()
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [(text.strip(), token_ids.cpu()) for text, token_ids in zip(texts, new_tokens)]


def completion_logprob(model: Any, tokenizer: Any, prompt: str, completion_ids: Any, cfg: RLConfig, device: Any) -> Any:
    import torch
    import torch.nn.functional as F
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    labels = input_ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = max(prompt_ids.numel() - 1, 0)
    comp_logp = logp[:, start:]
    return comp_logp.mean() if cfg.logprob_reduction == "mean" else comp_logp.sum()


def run_rollout(
    model: Any,
    tokenizer: Any,
    group: dict[str, Any],
    cfg: RLConfig,
    device: Any,
    *,
    epoch: int | None = None,
    group_index: int | None = None,
    rollout_index: int | None = None,
) -> dict[str, Any]:
    import torch

    completions, scores, completion_ids_by_label = {}, {}, {}
    retry_fail_count = 0
    prompts = {}
    pending: dict[str, str] = {}
    model.eval()
    with torch.no_grad():
        for row in group["items"]:
            label = CONDITION_LABELS[condition_key(row)]
            prompt = render_generation_prompt(tokenizer, row)
            prompts[label] = prompt
            pending[label] = prompt

        for attempt in range(1, cfg.max_regen_attempts + 1):
            if cfg.verbose and cfg.print_generation_progress:
                prefix = f"epoch={epoch} group={group_index} rollout={rollout_index}"
                print(
                    f"{prefix} batch_generation_attempt={attempt}/{cfg.max_regen_attempts} "
                    f"pending_conditions={list(pending)}",
                    flush=True,
                )
            batch_labels = list(pending)
            batch_prompts = [pending[label] for label in batch_labels]
            generated = generate_many(model, tokenizer, batch_prompts, cfg, device)
            next_pending: dict[str, str] = {}
            for label, (text, completion_ids) in zip(batch_labels, generated):
                parsed = parse_likert_score(text)
                if parsed is not None:
                    completions[label] = text
                    scores[label] = parsed
                    completion_ids_by_label[label] = completion_ids
                else:
                    retry_fail_count += 1
                    completions[label] = text
                    if attempt < cfg.max_regen_attempts:
                        next_pending[label] = pending[label]
            pending = next_pending
            if not pending:
                break
        if pending:
            failed_labels = list(pending)
            return {
                "parse_failed": True,
                "reward": cfg.parse_fail_group_reward,
                "retry_fail_count": retry_fail_count,
                "scores": scores,
                "completions": completions,
                "failed_conditions": failed_labels,
                "reward_details": {"reason": "parse failure after max regeneration attempts"},
                "prompts": prompts,
                "completion_ids": completion_ids_by_label,
            }
    elm_reward, details = compute_group_reward(scores)
    reward = elm_reward - cfg.retry_fail_penalty * retry_fail_count
    details.update({"elm_reward": elm_reward, "retry_penalty": cfg.retry_fail_penalty * retry_fail_count})
    return {
        "parse_failed": False,
        "reward": reward,
        "retry_fail_count": retry_fail_count,
        "scores": scores,
        "completions": completions,
        "reward_details": details,
        "prompts": prompts,
        "completion_ids": completion_ids_by_label,
    }


def _truncate_text(text: Any, max_chars: int) -> str:
    text = str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def print_group_rollout_details(
    *,
    epoch: int,
    group_index: int,
    num_groups: int,
    claim_id: Any,
    rewards: list[float],
    rollouts: list[dict[str, Any]],
    cfg: RLConfig,
) -> None:
    """Print rewards, reward terms, parsed scores, and completions for debugging RL behavior."""
    if not (cfg.verbose and cfg.print_rollout_details):
        return

    print("-" * 100, flush=True)
    print(
        f"RL rollout details | epoch={epoch} group={group_index}/{num_groups} "
        f"claim={claim_id} rewards={rewards} mean_reward={mean(rewards):.4f}",
        flush=True,
    )
    for rollout_idx, rollout in enumerate(rollouts, start=1):
        printable = {
            "rollout": rollout_idx,
            "reward": rollout.get("reward"),
            "parse_failed": rollout.get("parse_failed"),
            "retry_fail_count": rollout.get("retry_fail_count"),
            "scores": rollout.get("scores", {}),
            "reward_details": rollout.get("reward_details", {}),
            "prompts": {
                label: _truncate_text(text, cfg.max_prompt_print_chars)
                for label, text in sorted(rollout.get("prompts", {}).items())
            } if cfg.print_prompt_details else "disabled",
            "completions": {
                label: _truncate_text(text, cfg.max_completion_print_chars)
                for label, text in sorted(rollout.get("completions", {}).items())
            },
        }
        print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print("-" * 100, flush=True)


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a JSON list in {path}, got {type(data).__name__}.")


def infer_resume_start_epoch(cfg: RLConfig, history: list[dict[str, Any]]) -> int:
    """Infer the next epoch when resuming from an RL LoRA checkpoint."""
    parsed_epoch = 0
    if cfg.resume_lora_path:
        match = re.search(r"(?:checkpoint|best_rl_checkpoint)-epoch-(\d+)", Path(cfg.resume_lora_path).name)
        if match:
            parsed_epoch = int(match.group(1))
    history_epoch = max((int(record.get("epoch", 0)) for record in history), default=0)
    if cfg.resume_lora_path:
        return max(parsed_epoch, history_epoch) + 1
    return 1


def save_model_and_tokenizer(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    """Save a model/adapter robustly, including PEFT adapters resumed from device-mapped models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_attempts = (
        {"safe_serialization": False, "save_embedding_layers": False},
        {"safe_serialization": False},
        {},
    )
    last_error: Exception | None = None
    for kwargs in save_attempts:
        try:
            model.save_pretrained(str(output_dir), **kwargs)
            tokenizer.save_pretrained(str(output_dir))
            return
        except TypeError as exc:
            last_error = exc
            continue
        except NotImplementedError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Failed to save model/tokenizer to {output_dir}") from last_error


def reinforce_train(model: Any, tokenizer: Any, groups: list[dict[str, Any]], cfg: RLConfig) -> list[dict[str, Any]]:
    import torch

    device = next(model.parameters()).device
    if cfg.verbose:
        print(f"Starting REINFORCE training: epochs={cfg.num_train_epochs}, groups={len(groups)}, groups_per_step={cfg.groups_per_step}, rollouts_per_group={cfg.rollouts_per_group}", flush=True)
        print(f"Policy device inferred from first parameter: {device}", flush=True)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found for RL. Enable LoRA or unfreeze parameters before calling reinforce_train.")
    if cfg.verbose:
        trainable = sum(param.numel() for param in trainable_params)
        total = sum(param.numel() for param in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / max(total, 1):.4f}%)", flush=True)
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
    history_path = Path(cfg.output_dir) / "rl_training_history.json"
    history: list[dict[str, Any]] = load_json_list(history_path) if cfg.resume_lora_path else []
    valid_rows = _read_json_or_jsonl(cfg.valid_file) if cfg.valid_file else []
    valid_eval_dir = Path(cfg.rl_valid_eval_dir or Path(cfg.output_dir) / "rl_valid_eval_by_epoch")
    valid_eval_history_path = valid_eval_dir / "rl_valid_eval_history.json"
    valid_eval_history: list[dict[str, Any]] = load_json_list(valid_eval_history_path) if cfg.resume_lora_path else []
    previous_valid_summary: dict[str, Any] | None = (
        valid_eval_history[-1].get("valid_summary") if valid_eval_history else None
    )
    start_epoch = infer_resume_start_epoch(cfg, history)
    end_epoch = start_epoch + cfg.num_train_epochs - 1
    if cfg.verbose and valid_rows:
        print(f"Loaded {len(valid_rows)} validation rows for RL epoch-end ELM evaluation.", flush=True)
    if cfg.verbose and cfg.resume_lora_path:
        print(
            f"Resuming RL training from {cfg.resume_lora_path}; "
            f"next epoch={start_epoch}, running through epoch={end_epoch}.",
            flush=True,
        )
    opt_step = 0
    for epoch in range(start_epoch, end_epoch + 1):
        if cfg.verbose:
            print(f"===== RL epoch {epoch}/{end_epoch} =====", flush=True)
        random.shuffle(groups)
        for batch_start in range(0, len(groups), cfg.groups_per_step):
            group_batch = groups[batch_start:batch_start + cfg.groups_per_step]
            batch_losses = []
            batch_records = []
            for offset, group in enumerate(group_batch, start=1):
                group_index = batch_start + offset
                if cfg.verbose:
                    print(f"epoch={epoch} group={group_index}/{len(groups)} claim={group['claim_id']} starting {cfg.rollouts_per_group} rollout(s)", flush=True)
                rollouts = [
                    run_rollout(
                        model,
                        tokenizer,
                        group,
                        cfg,
                        device,
                        epoch=epoch,
                        group_index=group_index,
                        rollout_index=rollout_index,
                    )
                    for rollout_index in range(1, cfg.rollouts_per_group + 1)
                ]
                rewards = [float(r["reward"]) for r in rollouts]
                print_group_rollout_details(
                    epoch=epoch,
                    group_index=group_index,
                    num_groups=len(groups),
                    claim_id=group["claim_id"],
                    rewards=rewards,
                    rollouts=rollouts,
                    cfg=cfg,
                )
                if cfg.reward_baseline_mode == "group_mean" and len(rewards) > 1:
                    baseline = mean(rewards)
                elif cfg.reward_baseline_mode == "global":
                    baseline = cfg.global_reward_baseline
                elif cfg.reward_baseline_mode == "none":
                    baseline = 0.0
                else:
                    raise ValueError(
                        "reward_baseline_mode must be one of: 'group_mean', 'global', or 'none'."
                    )
                loss_values: list[float] = []
                model.train()
                for rollout, reward in zip(rollouts, rewards):
                    adv = reward - baseline
                    if rollout["parse_failed"]:
                        rollout["advantage"] = adv
                        continue
                    logprob_values: list[float] = []
                    loss_total = 0.0
                    num_conditions = max(len(rollout["completion_ids"]), 1)
                    for label, ids in rollout["completion_ids"].items():
                        condition_logp = completion_logprob(model, tokenizer, rollout["prompts"][label], ids, cfg, device)
                        condition_loss = -float(adv) * condition_logp / num_conditions
                        (condition_loss / cfg.gradient_accumulation_steps).backward()
                        logprob_values.append(float(condition_logp.detach().cpu()))
                        loss_total += float(condition_loss.detach().cpu())
                        del condition_logp, condition_loss
                    loss_values.append(loss_total)
                    rollout["advantage"] = adv
                    rollout["completion_logprob"] = mean(logprob_values) if logprob_values else 0.0
                    rollout["loss"] = loss_total
                group_loss_value = mean(loss_values) if loss_values else 0.0
                batch_losses.append(group_loss_value)
                batch_records.append({"epoch": epoch, "group_index": group_index, "claim_id": group["claim_id"], "rewards": rewards,
                                      "mean_reward": mean(rewards), "reward_baseline": baseline, "loss": group_loss_value, "rollouts": _json_safe_rollouts(rollouts)})
            loss_value = mean(batch_losses) if batch_losses else 0.0
            batch_number = math.ceil((batch_start + len(group_batch)) / max(cfg.groups_per_step, 1))
            if batch_number % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); opt_step += 1
            history.extend(batch_records)
            save_json(history, history_path)
            print(f"epoch={epoch} groups={batch_start + 1}-{batch_start + len(group_batch)}/{len(groups)} reward={mean([r['mean_reward'] for r in batch_records]):.3f} loss={loss_value:.4f}", flush=True)
        # Flush gradients for a final partial accumulation window at epoch end.
        if any(param.grad is not None for param in model.parameters()):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); opt_step += 1
        if valid_rows:
            previous_valid_summary = evaluate_rl_validation(
                model=model,
                tokenizer=tokenizer,
                valid_rows=valid_rows,
                cfg=cfg,
                epoch=epoch,
                device=device,
                previous_valid_summary=previous_valid_summary,
                history=valid_eval_history,
            )
        if cfg.save_every_epoch:
            ckpt = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch}"
            if cfg.verbose:
                print(f"Saving epoch checkpoint to {ckpt}", flush=True)
            save_model_and_tokenizer(model, tokenizer, ckpt)
    return history


def _json_safe_rollouts(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe = []
    for rollout in rollouts:
        item = {k: v for k, v in rollout.items() if k not in {"prompts", "completion_ids"}}
        safe.append(item)
    return safe


def load_groups_from_file(path: str | Path) -> list[dict[str, Any]]:
    groups = build_complete_claim_groups(_read_json_or_jsonl(path))
    if not groups:
        raise ValueError(f"No complete 8-condition claim groups found in {path}.")
    return groups


def save_config(cfg: RLConfig) -> None:
    save_json(asdict(cfg), Path(cfg.output_dir) / "rl_config.json")


def rl_valid_improved(
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    *,
    delta_arg_drop_tolerance: float,
) -> bool:
    """Return True when validation Delta_Arg/Delta_Src satisfy the RL best-save rule."""
    if current is None:
        return False
    if previous is None:
        return True
    cur_arg, cur_src = current["Delta_Arg"], current["Delta_Src"]
    prev_arg, prev_src = previous["Delta_Arg"], previous["Delta_Src"]
    if cur_arg > prev_arg and cur_src >= prev_src:
        return True
    if cur_src > prev_src and cur_arg >= prev_arg - delta_arg_drop_tolerance:
        return True
    return False


def save_rl_best_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    epoch: int | None = None,
) -> Path:
    output_dir = Path(output_dir)
    if epoch is not None:
        output_dir = output_dir.parent / f"{output_dir.name}-epoch-{epoch}"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_and_tokenizer(model, tokenizer, output_dir)
    return output_dir


def evaluate_rl_validation(
    *,
    model: Any,
    tokenizer: Any,
    valid_rows: list[dict[str, Any]],
    cfg: RLConfig,
    epoch: int,
    device: Any,
    previous_valid_summary: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run validation inference, save predictions/stats/history, and optionally save best LoRA."""
    eval_dir = Path(cfg.rl_valid_eval_dir or Path(cfg.output_dir) / "rl_valid_eval_by_epoch")
    best_dir = Path(cfg.best_rl_checkpoint_dir or Path(cfg.output_dir) / "best_rl_checkpoint")
    eval_dir.mkdir(parents=True, exist_ok=True)

    keep_fields = ["prompt_id", "claim_id", "groundtruth", "condition", "messages"]
    predictions = batch_generate_predictions(
        model,
        tokenizer,
        valid_rows,
        system_prompt="",
        keep_fields=keep_fields,
        max_new_tokens=cfg.valid_eval_max_new_tokens,
        temperature=cfg.valid_eval_temperature,
        top_p=cfg.valid_eval_top_p,
        batch_size=cfg.valid_eval_batch_size,
        device=device,
        progress_prefix=f"rl epoch {epoch} valid",
    )
    pred_path = eval_dir / f"epoch_{epoch}_valid_predictions.json"
    stats_path = eval_dir / f"epoch_{epoch}_valid_elm_stats.json"
    save_json(predictions, pred_path)
    stats = compute_elm_stats(predictions, prediction_file=pred_path)
    save_json(stats, stats_path)

    summary = stats["summary"]
    improved = rl_valid_improved(
        summary,
        previous_valid_summary,
        delta_arg_drop_tolerance=cfg.valid_delta_arg_drop_tolerance,
    )
    saved_best_dir = None
    if improved:
        saved_best_dir = save_rl_best_checkpoint(model, tokenizer, best_dir, epoch=epoch)

    record = {
        "epoch": epoch,
        "valid_summary": summary,
        "valid_improved": improved,
        "saved_as_best": improved,
        "best_checkpoint_dir": str(saved_best_dir) if saved_best_dir else None,
        "valid_predictions_file": str(pred_path),
        "valid_stats_file": str(stats_path),
    }
    history.append(record)
    save_json(history, eval_dir / "rl_valid_eval_history.json")

    print(
        f"RL valid epoch {epoch}: "
        f"Delta_Arg={summary['Delta_Arg']:.6f}, "
        f"Delta_Src={summary['Delta_Src']:.6f}, "
        f"saved_as_best={improved}",
        flush=True,
    )
    if improved:
        print(f"Saved best RL checkpoint to: {saved_best_dir}", flush=True)
    return summary


def apply_rl_lora(model: Any, cfg: RLConfig) -> Any:
    """Attach a small trainable LoRA adapter for memory-efficient RL updates."""
    if not cfg.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("RL LoRA training requires peft. Install with: pip install peft") from exc

    target_modules = [item.strip() for item in cfg.lora_target_modules.split(",") if item.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    if cfg.verbose and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model


def apply_or_resume_rl_lora(model: Any, cfg: RLConfig) -> Any:
    """Resume a trainable RL LoRA adapter when provided, otherwise attach a fresh adapter."""
    if cfg.resume_lora_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("Resuming RL LoRA training requires peft. Install with: pip install peft") from exc
        model = PeftModel.from_pretrained(model, cfg.resume_lora_path, is_trainable=True)
        if cfg.verbose:
            print(f"Resumed trainable RL LoRA adapter from: {cfg.resume_lora_path}", flush=True)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()
        return model
    return apply_rl_lora(model, cfg)


def maybe_prepare_kbit_training(model: Any, cfg: RLConfig) -> Any:
    """Prepare quantized policy models before attaching LoRA adapters."""
    if not cfg.load_in_4bit:
        return model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("4-bit RL LoRA training requires peft. Install with: pip install peft") from exc
    return prepare_model_for_kbit_training(model)
