"""Claim-group REINFORCE training utilities for post-SFT ELM alignment."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


from llm_lab.data import _apply_chat_template, _read_json_or_jsonl
from llm_lab.elm_eval import REQUIRED_CONDITION_KEYS, parse_likert_score, save_json

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
    temperature: float = 0.7
    top_p: float = 0.9
    max_regen_attempts: int = MAX_REGEN_ATTEMPTS
    retry_fail_penalty: float = RETRY_FAIL_PENALTY
    parse_fail_group_reward: float = PARSE_FAIL_GROUP_REWARD
    logprob_reduction: str = "mean"
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


def reward_from_diff_low_arg(diff: int) -> float:
    if diff == 0:
        return 2.0
    if diff == 1:
        return 1.5
    if diff == -1:
        return 1.0
    if diff in {2, -2}:
        return -1.0
    return -2.0


def reward_from_diff_low_src(diff: int) -> float:
    if diff >= 2:
        return 2.0
    if diff == 1:
        return 1.0
    if diff == 0:
        return -1.0
    return -2.0


def penalty_high_arg(diff: int) -> float:
    if diff >= 1:
        return 0.0
    if diff == 0:
        return -2.0
    return -3.0


def penalty_high_src(diff: int) -> float:
    if 0 <= diff <= 1:
        return 0.0
    if diff == 2:
        return -1.0
    if diff >= 3:
        return -2.0
    return -1.0


def compute_group_reward(scores: dict[str, int]) -> tuple[float, dict[str, Any]]:
    diffs = {
        "low_arg_high_source": scores["LHs"] - scores["LHw"],
        "low_arg_low_source": scores["LLs"] - scores["LLw"],
        "low_src_strong_arg": scores["LHs"] - scores["LLs"],
        "low_src_weak_arg": scores["LHw"] - scores["LLw"],
        "high_arg_high_source": scores["HHs"] - scores["HHw"],
        "high_arg_low_source": scores["HLs"] - scores["HLw"],
        "high_src_strong_arg": scores["HHs"] - scores["HLs"],
        "high_src_weak_arg": scores["HHw"] - scores["HLw"],
    }
    terms = {
        "low_arg_high_source": reward_from_diff_low_arg(diffs["low_arg_high_source"]),
        "low_arg_low_source": reward_from_diff_low_arg(diffs["low_arg_low_source"]),
        "low_src_strong_arg": reward_from_diff_low_src(diffs["low_src_strong_arg"]),
        "low_src_weak_arg": reward_from_diff_low_src(diffs["low_src_weak_arg"]),
        "high_arg_high_source": penalty_high_arg(diffs["high_arg_high_source"]),
        "high_arg_low_source": penalty_high_arg(diffs["high_arg_low_source"]),
        "high_src_strong_arg": penalty_high_src(diffs["high_src_strong_arg"]),
        "high_src_weak_arg": penalty_high_src(diffs["high_src_weak_arg"]),
    }
    return sum(terms.values()), {"diffs": diffs, "terms": terms}


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
    model.eval()
    with torch.no_grad():
        for condition_index, row in enumerate(group["items"], start=1):
            label = CONDITION_LABELS[condition_key(row)]
            prompt = render_generation_prompt(tokenizer, row)
            prompts[label] = prompt
            parsed = None
            for attempt in range(1, cfg.max_regen_attempts + 1):
                if cfg.verbose:
                    prefix = f"epoch={epoch} group={group_index} rollout={rollout_index}"
                    print(f"{prefix} condition={condition_index}/8 label={label} generation_attempt={attempt}/{cfg.max_regen_attempts}", flush=True)
                text, completion_ids = generate_one(model, tokenizer, prompt, cfg, device)
                parsed = parse_likert_score(text)
                if parsed is not None:
                    completions[label] = text
                    scores[label] = parsed
                    completion_ids_by_label[label] = completion_ids
                    break
                retry_fail_count += 1
                if attempt == cfg.max_regen_attempts:
                    return {
                        "parse_failed": True,
                        "reward": cfg.parse_fail_group_reward,
                        "retry_fail_count": retry_fail_count,
                        "scores": scores,
                        "completions": {**completions, label: text},
                        "failed_condition": label,
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
    history: list[dict[str, Any]] = []
    opt_step = 0
    for epoch in range(1, cfg.num_train_epochs + 1):
        if cfg.verbose:
            print(f"===== RL epoch {epoch}/{cfg.num_train_epochs} =====", flush=True)
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
                baseline = mean(rewards)
                std = pstdev(rewards) if len(rewards) > 1 else 0.0
                loss_values: list[float] = []
                model.train()
                for rollout, reward in zip(rollouts, rewards):
                    adv = reward - baseline
                    if cfg.rollouts_per_group > 1 and std > 1e-8:
                        adv /= std
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
                                      "mean_reward": baseline, "loss": group_loss_value, "rollouts": _json_safe_rollouts(rollouts)})
            loss_value = mean(batch_losses) if batch_losses else 0.0
            batch_number = math.ceil((batch_start + len(group_batch)) / max(cfg.groups_per_step, 1))
            if batch_number % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); opt_step += 1
            history.extend(batch_records)
            save_json(history, Path(cfg.output_dir) / "rl_training_history.json")
            print(f"epoch={epoch} groups={batch_start + 1}-{batch_start + len(group_batch)}/{len(groups)} reward={mean([r['mean_reward'] for r in batch_records]):.3f} loss={loss_value:.4f}", flush=True)
        # Flush gradients for a final partial accumulation window at epoch end.
        if any(param.grad is not None for param in model.parameters()):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); opt_step += 1
        if cfg.save_every_epoch:
            ckpt = Path(cfg.output_dir) / f"checkpoint-epoch-{epoch}"
            if cfg.verbose:
                print(f"Saving epoch checkpoint to {ckpt}", flush=True)
            model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
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


def maybe_prepare_kbit_training(model: Any, cfg: RLConfig) -> Any:
    """Prepare quantized policy models before attaching LoRA adapters."""
    if not cfg.load_in_4bit:
        return model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("4-bit RL LoRA training requires peft. Install with: pip install peft") from exc
    return prepare_model_for_kbit_training(model)
