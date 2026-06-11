#!/usr/bin/env python
"""Register the current Python environment as a Jupyter kernel.

Run this from the environment that already has the project dependencies
installed, for example after ``conda activate llm-lab``. It does not create or
install a new environment; it only makes the current interpreter selectable in
Jupyter's Kernel menu.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys


def default_kernel_name() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV", "").rstrip("/").split("/")[-1] or "llm-lab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the current environment as a Jupyter kernel.")
    parser.add_argument("--name", default=default_kernel_name(), help="Internal Jupyter kernel name.")
    parser.add_argument(
        "--display-name",
        default=None,
        help="Human-readable name shown in Jupyter. Defaults to 'Python (<name>)'.",
    )
    parser.add_argument("--user", action=argparse.BooleanOptionalAction, default=True, help="Install kernelspec for current user.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name = args.display_name or f"Python ({args.name})"
    if importlib.util.find_spec("ipykernel") is None:
        raise SystemExit("ipykernel is not installed in this environment. Run: python -m pip install ipykernel")

    cmd = [
        sys.executable,
        "-m",
        "ipykernel",
        "install",
        "--name",
        args.name,
        "--display-name",
        display_name,
    ]
    if args.user:
        cmd.append("--user")

    print(f"Current Python: {sys.executable}")
    print(f"Registering Jupyter kernel: {display_name!r} ({args.name})")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Failed to register kernel. Command exited with {exc.returncode}: {' '.join(cmd)}") from exc
    print("Done. In Jupyter, choose Kernel -> Change Kernel ->", display_name)


if __name__ == "__main__":
    main()
