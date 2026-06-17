#!/usr/bin/env python
"""Post-SFT claim-group REINFORCE training for ELM rewards."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_lab.model_utils import ensure_pad_token, print_cuda_info  # noqa: E402
from llm_lab.rl_reinforce import RLConfig, load_groups_from_file, reinforce_train, save_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run claim-group REINFORCE after SFT.")
    for field, default in RLConfig().__dict__.items():
        arg = "--" + field
        if isinstance(default, bool):
            p.add_argument(arg, action=argparse.BooleanOptionalAction, default=default)
        elif isinstance(default, int):
            p.add_argument(arg, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(arg, type=float, default=default)
        else:
            p.add_argument(arg, default=default)
    return p.parse_args()


def torch_dtype(name: str):
    import torch
    if name in {"auto", "none", None}:
        return "auto"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    cfg = RLConfig(**vars(args))
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    save_config(cfg)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    print_cuda_info()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if cfg.load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype(cfg.dtype),
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    if cfg.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    groups = load_groups_from_file(cfg.train_file)
    print(f"Loaded {len(groups)} complete claim groups from {cfg.train_file}")
    history = reinforce_train(model, tokenizer, groups, cfg)
    model.save_pretrained(Path(cfg.output_dir) / "final_checkpoint")
    tokenizer.save_pretrained(Path(cfg.output_dir) / "final_checkpoint")
    print(f"RL training complete. History records: {len(history)}")


if __name__ == "__main__":
    main()
