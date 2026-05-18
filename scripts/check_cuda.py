#!/usr/bin/env python
"""Print PyTorch/CUDA diagnostics for the current environment."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from dataclasses import dataclass

if importlib.util.find_spec("torch") is None:
    raise SystemExit("PyTorch is not installed. Install dependencies with: pip install -r requirements.txt")

import torch


@dataclass(frozen=True)
class NvidiaSmiInfo:
    driver_version: str | None
    cuda_version: str | None


def _run_nvidia_smi() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    return output.strip() or None


def _parse_nvidia_smi(output: str | None) -> NvidiaSmiInfo:
    if not output:
        return NvidiaSmiInfo(driver_version=None, cuda_version=None)
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", output)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return NvidiaSmiInfo(
        driver_version=driver_match.group(1) if driver_match else None,
        cuda_version=cuda_match.group(1) if cuda_match else None,
    )


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    numbers = re.findall(r"\d+", value)
    return tuple(int(part) for part in numbers[:2]) if numbers else None


def _print_mismatch_hint(torch_cuda: str | None, smi_info: NvidiaSmiInfo) -> None:
    torch_cuda_tuple = _version_tuple(torch_cuda)
    driver_cuda_tuple = _version_tuple(smi_info.cuda_version)
    if torch_cuda_tuple and driver_cuda_tuple and torch_cuda_tuple > driver_cuda_tuple:
        print("\n诊断：PyTorch 自带 CUDA runtime 版本高于当前 NVIDIA 驱动支持的最高 CUDA 版本。")
        print(f"- PyTorch CUDA build: {torch_cuda}")
        print(f"- nvidia-smi CUDA Version: {smi_info.cuda_version}")
        print("这通常不是 CUDA_VISIBLE_DEVICES 的问题，而是 PyTorch wheel 与系统驱动不兼容。")
        print("建议二选一：")
        print("1. 让管理员升级 NVIDIA 驱动，使其支持当前 PyTorch CUDA build；或")
        print("2. 保持驱动不变，重装匹配旧一些 CUDA runtime 的 PyTorch。")
        print("   对你这次日志里的 Driver CUDA 12.2，通常可优先装 cu121 版 PyTorch：")
        print("   pip uninstall -y torch torchvision torchaudio")
        print("   pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121")
        print("   pip install -r requirements.txt")


def main() -> None:
    smi_output = _run_nvidia_smi()
    smi_info = _parse_nvidia_smi(smi_output)

    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda}")
    print(f"nvidia-smi driver version: {smi_info.driver_version or '<not found>'}")
    print(f"nvidia-smi CUDA Version: {smi_info.cuda_version or '<not found>'}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    count = torch.cuda.device_count() if cuda_available else 0
    print(f"Visible GPU count: {count}")
    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        print(f"GPU {idx}: {props.name}, capability {props.major}.{props.minor}, {props.total_memory / 1024**3:.1f} GiB")

    if not cuda_available:
        print("\n提示：当前 Python 环境无法通过 PyTorch 使用 CUDA。")
        _print_mismatch_hint(torch.version.cuda, smi_info)
        if not smi_output:
            print("未找到 nvidia-smi；请确认机器有 NVIDIA 驱动，且当前 shell 能访问 nvidia-smi。")
        print("如果版本匹配后仍不可用，再检查 CUDA_VISIBLE_DEVICES、作业调度器分配和容器 GPU 挂载。")


if __name__ == "__main__":
    main()
