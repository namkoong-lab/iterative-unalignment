"""KL and alpha regularization terms comparing proposal logits to the frozen reference."""

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


def alpha_reg_term_per_position(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    prefix_len: int,
    moment: float = 2.0,
) -> torch.Tensor:
    """Per-position log E_Q[(P/Q)^moment] over generated tokens. Returns [B, K]."""
    _check_logit_shapes(current_logits, original_logits)
    log_q = F.log_softmax(generated_predictor_logits(current_logits, prefix_len), dim=-1)
    log_p = F.log_softmax(generated_predictor_logits(original_logits, prefix_len), dim=-1)
    log_terms = (1.0 - moment) * log_q + moment * log_p
    return torch.logsumexp(log_terms, dim=-1)


def kl_qp_term_per_position(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Per-position KL(Q || P) over generated tokens. Returns [B, K]."""
    _check_logit_shapes(current_logits, original_logits)
    log_q = F.log_softmax(generated_predictor_logits(current_logits, prefix_len), dim=-1)
    log_p = F.log_softmax(generated_predictor_logits(original_logits, prefix_len), dim=-1)
    q_probs = torch.exp(log_q)
    return (q_probs * (log_q - log_p)).sum(dim=-1)
