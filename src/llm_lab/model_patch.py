"""Helpers for inspecting/modifying model internals before LoRA wrapping."""

from __future__ import annotations

from importlib import util
from pathlib import Path
from types import ModuleType


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    for _, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
    ratio = (100.0 * trainable / total) if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def preview_modules(model, limit: int = 120) -> None:
    print("\n=== Module Preview (name -> class) ===")
    for idx, (name, module) in enumerate(model.named_modules()):
        if idx >= limit:
            print(f"... truncated. Use a larger --preview_modules_limit to print more than {limit} modules.")
            break
        print(f"{name or '<root>'} -> {module.__class__.__name__}")


def load_patch_module(path: str) -> ModuleType:
    patch_path = Path(path)
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch script not found: {patch_path}")
    spec = util.spec_from_file_location("llm_lab_user_patch", patch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load patch script spec: {patch_path}")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_user_patch(model, patch_script: str | None):
    if not patch_script:
        return model
    module = load_patch_module(patch_script)
    if not hasattr(module, "apply_patch"):
        raise AttributeError(
            f"Patch script '{patch_script}' must define apply_patch(model) -> model."
        )
    print(f"Applying custom model patch: {patch_script}")
    patched = module.apply_patch(model)
    if patched is None:
        raise ValueError("apply_patch(model) returned None. Return the patched model object.")
    return patched
