from __future__ import annotations

import copy
from collections.abc import Mapping

import torch
from torch import nn


class EMA:
    """Exponential moving average with a serializable, model-like shadow copy."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"EMA decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        shadow = self.shadow.state_dict()
        if source.keys() != shadow.keys():
            raise ValueError("Raw and EMA models have different state_dict keys")
        for name, average in shadow.items():
            value = source[name].detach()
            if torch.is_floating_point(average):
                average.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                average.copy_(value)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "model": self.shadow.state_dict()}

    def load_state_dict(self, state: Mapping, *, strict: bool = True) -> None:
        if "decay" in state:
            self.decay = float(state["decay"])
        model_state = state.get("model", state)
        self.shadow.load_state_dict(model_state, strict=strict)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow.state_dict())
