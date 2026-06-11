#!/usr/bin/env python
"""Download a HuggingFace model snapshot to a visible local directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download a HuggingFace model for offline/local inference and training.")
    parser.add_argument("--repo_id", default="Qwen/Qwen3-1.7B", help="HuggingFace repo id, e.g. Qwen/Qwen3-1.7B.")
    parser.add_argument("--local_dir", default="model/Qwen3-1.7B", help="Directory where files will be written visibly.")
    parser.add_argument("--revision", default=None, help="Optional branch/tag/commit.")
    parser.add_argument("--cache_dir", default=None, help="Optional HuggingFace cache directory.")
    parser.add_argument(
        "--local_dir_use_symlinks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use symlinks in local_dir. False writes regular files, which is easier to inspect/copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install huggingface_hub") from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo_id} to {local_dir} ...", flush=True)
    print("If this is slow, set a working HF_ENDPOINT/proxy or download on a machine with better HuggingFace access.", flush=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_dir=str(local_dir),
        local_dir_use_symlinks=args.local_dir_use_symlinks,
    )
    print(f"Download complete: {path}", flush=True)
    print("Use it with:", flush=True)
    print(f"  python scripts/infer_transformers.py --model_name_or_path {local_dir}", flush=True)


if __name__ == "__main__":
    main()
