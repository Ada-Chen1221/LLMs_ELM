#!/usr/bin/env python
"""Launch the project notebook on a selected physical GPU.

This is a thin convenience wrapper around ``python -m notebook``. It sets
``CUDA_VISIBLE_DEVICES`` before the notebook server starts so kernels launched
from it inherit the selected GPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "unsloth_sft_grpo_qwen3.ipynb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Unsloth notebook on one selected physical GPU.")
    parser.add_argument("--gpu_id", default="0", help="Physical GPU id, e.g. 0 or 1. Exposed to notebook as cuda:0.")
    parser.add_argument("--notebook", default=str(DEFAULT_NOTEBOOK), help="Notebook path to open.")
    parser.add_argument("--no-browser", action="store_true", help="Pass --no-browser to notebook server.")
    parser.add_argument("--port", type=int, default=None, help="Optional notebook server port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [sys.executable, "-m", "notebook", args.notebook]
    if args.no_browser:
        cmd.append("--no-browser")
    if args.port is not None:
        cmd.extend(["--port", str(args.port)])

    print(f"Launching notebook with physical GPU {args.gpu_id} visible as cuda:0")
    print("CUDA_VISIBLE_DEVICES=", env["CUDA_VISIBLE_DEVICES"])
    print("Command:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Notebook exited with {exc.returncode}") from exc
    except ModuleNotFoundError as exc:
        raise SystemExit("notebook is not installed. Run: python -m pip install notebook") from exc


if __name__ == "__main__":
    main()
