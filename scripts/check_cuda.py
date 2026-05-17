#!/usr/bin/env python
"""Print PyTorch/CUDA diagnostics for the current environment."""

from __future__ import annotations


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Install dependencies with: pip install -r requirements.txt") from exc

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible GPU count: {count}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU {idx}: {props.name}, capability {props.major}.{props.minor}, {props.total_memory / 1024**3:.1f} GiB")

    if not torch.cuda.is_available():
        print("提示：当前环境未检测到 CUDA。请在服务器上确认 NVIDIA 驱动、CUDA 版 PyTorch 和 CUDA_VISIBLE_DEVICES。")


if __name__ == "__main__":
    main()
