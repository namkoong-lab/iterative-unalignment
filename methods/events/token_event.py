"""Token rare-event implementation."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from events._base import RareEvent, assert_per_sample_shape, slice_generated_logits
from proposals._base import resolve_single_token_id


@dataclass(frozen=True)
class TokenEvent(RareEvent):
    """Rare event defined by the presence of one token in generated output."""

    token: str
    token_id: int
    gt_prob: float
    surrogate: str

    @classmethod
    def from_config(cls, tokenizer, config: Mapping[str, Any]) -> "TokenEvent":
        token = config.get("token")
        if token is None:
            raise ValueError("TokenEvent config requires token=<token_string>.")
        token = str(token)
        surrogate = config.get("surrogate")
        surrogate = str(surrogate) if surrogate is not None else None
        if surrogate != "sum":
            raise ValueError(f"Unsupported surrogate {surrogate!r}. Expected 'sum'.")

        gt_raw = config.get("gt_prob", config.get("gt"))
        if gt_raw is None:
            raise ValueError("TokenEvent config requires gt_prob=<float> (or gt=<float>).")
        gt_prob = float(gt_raw)
        if gt_prob < 0.0 or gt_prob > 1.0:
            raise ValueError(f"gt_prob must be in [0, 1], got {gt_prob}.")

        token_id = resolve_single_token_id(tokenizer, token)
        return cls(token=token, token_id=token_id, gt_prob=gt_prob, surrogate=surrogate)

    def compute_indicator(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        if generated_tokens.ndim != 2:
            raise ValueError(
                f"generated_tokens must have shape [B, K], got {tuple(generated_tokens.shape)}"
            )
        return (generated_tokens == self.token_id).any(dim=-1)

    def diagnostic_counts(
        self,
        generated_tokens: torch.Tensor,
        event_indicator: torch.Tensor,
    ) -> dict[str, int]:
        if generated_tokens.ndim != 2 or generated_tokens.shape[1] == 0:
            return {}
        later_only_mask = (
            event_indicator.bool() & (generated_tokens[:, 0] != self.token_id)
        )
        return {"later_only_count": int(later_only_mask.sum().item())}

    def compute_surrogate_loss(
        self,
        all_logits: torch.Tensor,
        prefix_len: int,
        generated_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del generated_tokens
        if self.surrogate != "sum":
            raise ValueError(f"Unsupported surrogate: {self.surrogate!r}")
        generated = slice_generated_logits(all_logits, prefix_len)
        log_probs = F.log_softmax(generated, dim=-1)
        rare_log_probs = log_probs[:, :, self.token_id]
        return assert_per_sample_shape(-rare_log_probs.mean(dim=-1), generated.shape[0])
