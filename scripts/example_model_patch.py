"""Example patch script for --patch_script in train_lora.py/train_qlora.py."""


def apply_patch(model):
    """User hook: modify model internals before LoRA adapter injection."""
    # Example: print one specific module if present.
    target = "model.layers.0.self_attn.q_proj"
    module = dict(model.named_modules()).get(target)
    if module is not None:
        print(f"[example_model_patch] Found target module: {target} ({module.__class__.__name__})")
    else:
        print(f"[example_model_patch] Target module not found: {target}")

    # Put your own surgery here (replace blocks, wrap modules, etc.).
    return model
