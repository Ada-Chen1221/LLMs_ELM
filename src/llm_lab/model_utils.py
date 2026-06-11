"""Model and tokenizer helpers for native HuggingFace Transformers workflows."""

from __future__ import annotations

from typing import Any


def configure_visible_gpu(gpu_id: str | None) -> None:
    """Restrict this process to one physical GPU before importing torch/unsloth.

    Unsloth follows PyTorch's CUDA visibility. For single-GPU training, the
    reliable way to place the run on a physical GPU is to set
    ``CUDA_VISIBLE_DEVICES`` before CUDA is initialized. The selected physical
    GPU is then exposed inside the process as ``cuda:0``.
    """
    if gpu_id is None or str(gpu_id).strip() == "":
        return
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id).strip()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_torch_dtype(dtype: str):
    """Map a CLI dtype string to a torch dtype or ``"auto"``."""
    import torch

    normalized = dtype.lower()
    if normalized == "auto":
        return "auto"
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_cuda_memory_info(device: int = 0) -> tuple[float, float]:
    """Return ``(free_gib, total_gib)`` for one visible CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return free_bytes / 1024**3, total_bytes / 1024**3


def print_cuda_info() -> None:
    """Print CUDA availability, visible GPU details, and current free memory."""
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Visible GPU count: {gpu_count}")
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        free_gb, _ = get_cuda_memory_info(idx)
        print(f"GPU {idx}: {props.name} ({total_gb:.1f} GiB total, {free_gb:.1f} GiB free)")


def require_min_cuda_memory(min_free_gb: float, device: int = 0, context: str = "this run") -> None:
    """Fail early with a clear message when the selected visible GPU is already full.

    CUDA OOM messages can be misleading when the notebook is attached to the
    wrong physical GPU. This check runs before model loading so users see an
    actionable error such as selecting another ``CUDA_VISIBLE_DEVICES`` value.
    """
    import os
    import torch

    if not torch.cuda.is_available() or min_free_gb <= 0:
        return
    free_gb, total_gb = get_cuda_memory_info(device)
    if free_gb >= min_free_gb:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    raise RuntimeError(
        f"Not enough free CUDA memory for {context}: visible GPU {device} has "
        f"{free_gb:.2f} GiB free / {total_gb:.2f} GiB total, but at least "
        f"{min_free_gb:.2f} GiB was requested. CUDA_VISIBLE_DEVICES={visible!r}.\n"
        "This usually means the selected physical GPU is already occupied. "
        "Pick a freer GPU before starting Python/Jupyter, for example:\n"
        "  python scripts/launch_notebook.py --gpu_id 1\n"
        "or inspect/kill stale processes with nvidia-smi."
    )


def ensure_pad_token(tokenizer: Any) -> None:
    """Ensure a causal-LM tokenizer has a pad token for batching/training."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def maybe_enable_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """Enable gradient checkpointing and input gradients when requested."""
    if not enabled:
        return
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
