"""Forward hooks that add a single steering vector to residual streams or LM-head logits."""

from __future__ import annotations

from typing import Optional

import torch

from proposals._arch import find_transformer_blocks


_MODES = ("trainable", "cem")


def _check_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")


class _SteeringWrapperBase(torch.nn.Module):
    """Common machinery: parameter ownership, current_vectors, hook offset."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        *,
        mode: str,
        param_dim: int,
        device: torch.device,
        init_scale: float,
    ) -> None:
        super().__init__()
        _check_mode(mode)
        self.model = base_model
        self.mode = mode
        self.param_dim = int(param_dim)

        if mode == "trainable":
            init = self._init_trainable(self.param_dim, device, init_scale)
            self.steering_vector = torch.nn.Parameter(init)
            self.current_vectors: Optional[torch.Tensor] = None
        else:
            self.steering_vector = None
            self.current_vectors: Optional[torch.Tensor] = None

        self._hooks: list = []
        self._register_hooks()

    @staticmethod
    def _init_trainable(param_dim: int, device: torch.device, init_scale: float) -> torch.Tensor:
        # Subclass overrides if it needs a different initialisation.
        return torch.full((param_dim,), float(init_scale), device=device, dtype=torch.float32)

    def _register_hooks(self) -> None:
        raise NotImplementedError

    def set_current_vectors(self, vectors: Optional[torch.Tensor]) -> None:
        if self.mode != "cem":
            raise RuntimeError("set_current_vectors is only valid in CEM mode.")
        self.current_vectors = vectors

    def _steering_offset(self) -> Optional[torch.Tensor]:
        if self.mode == "trainable":
            return self.steering_vector.view(1, 1, -1)
        if self.current_vectors is None:
            return None
        return self.current_vectors.unsqueeze(1)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def config(self):
        return self.model.config


class ActivationSteeringWrapper(_SteeringWrapperBase):
    """Adds a steering offset to every transformer block's residual stream."""

    def _register_hooks(self) -> None:
        for block in find_transformer_blocks(self.model):
            self._hooks.append(block.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook(_module, _input, output):
            steering = self._steering_offset()
            if steering is None:
                return output
            if isinstance(output, tuple):
                return (output[0] + steering,) + output[1:]
            return output + steering
        return hook


class LogitSteeringWrapper(_SteeringWrapperBase):
    """Adds a steering offset to the LM head's logits (exponential tilting)."""

    @staticmethod
    def _init_trainable(param_dim: int, device: torch.device, init_scale: float) -> torch.Tensor:
        # Initialize logit offsets independently across vocabulary entries.
        return torch.randn(param_dim, device=device, dtype=torch.float32) * float(init_scale)

    def _register_hooks(self) -> None:
        self._hooks.append(self.model.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook(_module, _input, output):
            steering = self._steering_offset()
            if steering is None:
                return output
            if hasattr(output, "logits"):
                output.logits = output.logits + steering
                return output
            if isinstance(output, tuple):
                return (output[0] + steering,) + output[1:]
            return output + steering
        return hook
