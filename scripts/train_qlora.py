#!/usr/bin/env python
"""Compatibility wrapper: run Unsloth SFT with 4-bit QLoRA."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_sft_unsloth.py"


if __name__ == "__main__":
    if "--load_in_4bit" not in sys.argv and "--no-load_in_4bit" not in sys.argv:
        sys.argv.append("--load_in_4bit")
    runpy.run_path(str(SCRIPT), run_name="__main__")
