"""RareEvent interface and shared helpers used by every concrete event."""

from abc import ABC, abstractmethod
from typing import Optional

import torch

from is_weights import generated_predictor_logits


class RareEvent(ABC):
    """Rare-event definition used by the trainer: indicator plus surrogate loss."""

    surrogate: str
    gt_prob: float

    @abstractmethod
    def compute_indicator(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        """Return per-sequence indicator [B] (bool/float) — 1 iff the event fires."""

    @abstractmethod
    def compute_surrogate_loss(
        self,
        all_logits: torch.Tensor,
        prefix_len: int,
        generated_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sample [B] loss whose gradient pushes Q toward the rare event."""

    def diagnostic_counts(
        self,
        generated_tokens: torch.Tensor,
        event_indicator: torch.Tensor,
    ) -> dict[str, int]:
        """Optional event-specific diagnostics; default returns no extra fields."""
        return {}


def slice_generated_logits(all_logits: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Slice all_logits to the predictor positions for sampled generated tokens."""
    return generated_predictor_logits(all_logits=all_logits, prefix_len=prefix_len)


def assert_per_sample_shape(loss_values: torch.Tensor, batch_size: int) -> torch.Tensor:
    if loss_values.ndim != 1 or loss_values.shape[0] != batch_size:
        raise ValueError(
            f"Loss function must return shape [B]={batch_size}, got {tuple(loss_values.shape)}"
        )
    return loss_values
