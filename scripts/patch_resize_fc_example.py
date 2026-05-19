"""Example: replace one FC layer with custom bottleneck block and train only it."""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckFC(nn.Module):
    """Keep input/output dims same, but change internal FC width."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def _set_by_path(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parts = path.split('.')
    parent = root
    for key in parts[:-1]:
        parent = getattr(parent, key)
    setattr(parent, parts[-1], new_module)


def apply_patch(model):
    """Replace model.layers.0.mlp.down_proj and freeze all but new block."""
    target_name = "model.layers.0.mlp.down_proj"
    module_map = dict(model.named_modules())
    if target_name not in module_map:
        raise ValueError(f"Target module not found: {target_name}")

    old = module_map[target_name]
    if not isinstance(old, nn.Linear):
        raise TypeError(f"Expected nn.Linear at {target_name}, got {type(old)}")

    new_block = BottleneckFC(
        in_dim=old.in_features,
        out_dim=old.out_features,
        hidden_dim=1024,
    )
    _set_by_path(model, target_name, new_block)

    # freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # only train replaced block
    for p in dict(model.named_modules())[target_name].parameters():
        p.requires_grad = True

    print(f"[patch_resize_fc_example] Replaced {target_name}: Linear({old.in_features},{old.out_features}) -> BottleneckFC(hidden_dim=1024)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[patch_resize_fc_example] Trainable params after patch: {trainable}/{total}")
    return model
