"""Dense reverse-KL regularization comparing proposal logits to the frozen reference."""

import torch
import torch.nn.functional as F

from is_weights import generated_predictor_logits


def _check_logit_shapes(current_logits: torch.Tensor, original_logits: torch.Tensor) -> None:
    if current_logits.ndim != 3 or original_logits.ndim != 3:
        raise ValueError(
            "current_logits and original_logits must both have shape [B, seq_len, V]."
        )
    if current_logits.shape != original_logits.shape:
        raise ValueError(
            f"Shape mismatch: current_logits {tuple(current_logits.shape)} "
            f"vs original_logits {tuple(original_logits.shape)}"
        )


def kl_qp_term_per_position(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Full-vocabulary KL(Q || P) at each generated position. Returns [B, K]."""
    _check_logit_shapes(current_logits, original_logits)
    log_q = F.log_softmax(generated_predictor_logits(current_logits, prefix_len), dim=-1)
    log_p = F.log_softmax(generated_predictor_logits(original_logits, prefix_len), dim=-1)
    q_probs = torch.exp(log_q)
    return (q_probs * (log_q - log_p)).sum(dim=-1)
