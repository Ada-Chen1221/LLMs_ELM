#!/usr/bin/env python
"""Compatibility entry point for Unsloth 4-bit QLoRA SFT training."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN_LORA = ROOT / "scripts" / "train_lora.py"

if "--load_in_4bit" not in sys.argv and "--no-load_in_4bit" not in sys.argv:
    sys.argv.append("--load_in_4bit")
if "--output_dir" not in sys.argv:
    sys.argv.extend(["--output_dir", "outputs/qwen3_1p7b_unsloth_qlora"])

runpy.run_path(str(TRAIN_LORA), run_name="__main__")
