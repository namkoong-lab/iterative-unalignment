"""Importance-weight / ESS math and proposal-vs-reference logits alignment."""

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F


def ordinary_is_log_estimate(log_weights: torch.Tensor, event_indicator: torch.Tensor) -> torch.Tensor:
    """Log of mean(1_event * p/q), without clipping or self-normalization.

    Returns -inf for no hits; keeping log space also preserves estimates too
    small to represent in ordinary float64 probability space.
    """
    if log_weights.ndim != 1 or log_weights.numel() == 0 or event_indicator.shape != log_weights.shape:
        raise ValueError("Ordinary IS requires nonempty, matching weight and event vectors")
    log_contributions = log_weights.double().masked_fill(~event_indicator.bool(), -torch.inf)
    return torch.logsumexp(log_contributions, dim=0) - math.log(log_weights.numel())


def compute_model_and_reference_logits(
    proposal,
    reference_model: torch.nn.Module,
    sampled_ids: torch.Tensor,
    *,
    q_forward_ctx=None,
    p_forward_ctx=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (current_logits, original_logits), each shaped [B, T, V].

    Optional q_forward_ctx / p_forward_ctx wrap the two forwards so a cost
    tracker can time them separately without splitting this helper.
    """
    if q_forward_ctx is None:
        q_forward_ctx = nullcontext()
    if p_forward_ctx is None:
        p_forward_ctx = nullcontext()
    with q_forward_ctx:
        current_logits = proposal.compute_logits(input_ids=sampled_ids)
    with p_forward_ctx:
        with torch.no_grad():
            original_logits = reference_model(input_ids=sampled_ids).logits
    if current_logits.shape != original_logits.shape:
        raise ValueError(
            f"Current/reference logits shape mismatch: "
            f"{tuple(current_logits.shape)} vs {tuple(original_logits.shape)}"
        )
    return current_logits, original_logits


def generated_predictor_logits(all_logits: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    Return logits aligned to sampled generated tokens.

    For generated token ids sampled_ids[:, prefix_len:],
    the matching predictor logits are all_logits[:, prefix_len - 1 : -1, :].
    """
    if all_logits.ndim != 3:
        raise ValueError(f"Expected logits to have shape [B, seq_len, V], got {tuple(all_logits.shape)}")
    seq_len = all_logits.shape[1]
    if prefix_len <= 0 or prefix_len >= seq_len:
        raise ValueError(f"prefix_len must be in [1, {seq_len - 1}], got {prefix_len}")
    return all_logits[:, prefix_len - 1 : -1, :]


def compute_population_ess_from_log_weights(log_importance_weights: torch.Tensor) -> float | None:
    """
    Compute normalized population ESS from per-sequence log importance weights.

    Args:
        log_importance_weights: Tensor [B] with log(P/Q) per sampled sequence.
    """
    if log_importance_weights.ndim != 1:
        raise ValueError(
            "log_importance_weights must have shape [B], got "
            f"{tuple(log_importance_weights.shape)}"
        )
    batch_size = log_importance_weights.shape[0]
    if batch_size < 2:
        return None

    if torch.all(log_importance_weights == log_importance_weights[0]):
        # Equal weights give normalized ESS of one.
        return 1.0

    max_log_w = log_importance_weights.max()
    stable_w = torch.exp(log_importance_weights - max_log_w)
    sum_w = stable_w.sum()
    sum_w_sq = (stable_w**2).sum()
    ess = (sum_w**2) / (sum_w_sq + 1e-12)
    return float((ess / batch_size).item())


def compute_rare_event_ess_from_log_weights(
    log_importance_weights: torch.Tensor,
    event_indicator: torch.Tensor,
) -> float | None:
    """
    Normalized ESS computed only over samples that hit the rare event.

    Filters log_importance_weights down to entries where event_indicator is
    truthy, then defers to compute_population_ess_from_log_weights. Returns
    None if fewer than two rare-event samples are present.
    """
    if log_importance_weights.ndim != 1:
        raise ValueError(
            "log_importance_weights must have shape [B], got "
            f"{tuple(log_importance_weights.shape)}"
        )
    if event_indicator.shape != log_importance_weights.shape:
        raise ValueError(
            "event_indicator must match log_importance_weights shape; got "
            f"{tuple(event_indicator.shape)} vs {tuple(log_importance_weights.shape)}"
        )
    mask = event_indicator.to(dtype=torch.bool, device=log_importance_weights.device)
    selected = log_importance_weights[mask]
    if selected.numel() < 2:
        return None
    return compute_population_ess_from_log_weights(log_importance_weights=selected)


def compute_log_p_and_log_q_per_pos(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    effective_prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-position log P(x_k | x_<k) and log Q(x_k | x_<k) over the K generated tokens.

    Returns:
        (log_p_per_pos, log_q_per_pos): each Tensor [B, K] in float64.
    """
    if sampled_ids.ndim != 2:
        raise ValueError(f"sampled_ids must have shape [B, seq_len], got {tuple(sampled_ids.shape)}")
    if current_logits.ndim != 3 or original_logits.ndim != 3:
        raise ValueError(
            "current_logits and original_logits must have shape [B, seq_len, V]; "
            f"got {tuple(current_logits.shape)} and {tuple(original_logits.shape)}"
        )
    if current_logits.shape != original_logits.shape:
        raise ValueError(
            "current_logits and original_logits must have identical shape, got "
            f"{tuple(current_logits.shape)} and {tuple(original_logits.shape)}"
        )
    if current_logits.shape[:2] != sampled_ids.shape:
        raise ValueError(
            "Logits and sampled ids must align in first two dims, got "
            f"logits[:2]={tuple(current_logits.shape[:2])} and ids={tuple(sampled_ids.shape)}"
        )

    generated_ids = sampled_ids[:, effective_prefix_len:]  # [B, K]
    q_pred_logits = generated_predictor_logits(current_logits, effective_prefix_len)  # [B, K, V]
    p_pred_logits = generated_predictor_logits(original_logits, effective_prefix_len)  # [B, K, V]

    if generated_ids.shape[1] == 0:
        raise ValueError(
            "Importance-weight computation requires at least one generated token; "
            f"got generated_ids shape {tuple(generated_ids.shape)}"
        )
    if q_pred_logits.shape[:2] != generated_ids.shape or p_pred_logits.shape[:2] != generated_ids.shape:
        raise ValueError(
            "Importance-weight logits/ids alignment mismatch: "
            f"q_pred_logits[:2]={tuple(q_pred_logits.shape[:2])}, "
            f"p_pred_logits[:2]={tuple(p_pred_logits.shape[:2])}, "
            f"generated_ids={tuple(generated_ids.shape)}"
        )

    gather_index = generated_ids.unsqueeze(-1)
    q_log_probs = F.log_softmax(q_pred_logits, dim=-1).gather(-1, gather_index).squeeze(-1)  # [B, K]
    p_log_probs = F.log_softmax(p_pred_logits, dim=-1).gather(-1, gather_index).squeeze(-1)  # [B, K]
    return p_log_probs.to(torch.float64), q_log_probs.to(torch.float64)


def compute_worst_case_log_ratio_per_pos(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    effective_prefix_len: int,
) -> torch.Tensor:
    """
    Per-position worst-case log importance weight along the realized prefix path.

    At each generated position k, given the actually-sampled history x_<k, this is
        max_v [ log P(v | x_<k) - log Q(v | x_<k) ],
    i.e. the largest per-step weight log(P/Q) any token *could* have contributed
    had it been sampled. Summed over the K positions it upper-bounds the realized
    per-sequence log weight (the worst weight the prefix path could yield).

    Returns:
        worst_log_ratio_per_pos: Tensor [B, K] in float64.
    """
    q_pred_logits = generated_predictor_logits(current_logits, effective_prefix_len)  # [B, K, V]
    p_pred_logits = generated_predictor_logits(original_logits, effective_prefix_len)  # [B, K, V]
    if q_pred_logits.shape[1] == 0:
        raise ValueError(
            "Worst-case log-ratio requires at least one generated token; "
            f"got predictor logits shape {tuple(q_pred_logits.shape)}"
        )
    # Subtract in place to avoid an additional vocabulary-sized tensor.
    log_ratio = F.log_softmax(p_pred_logits, dim=-1)
    log_ratio -= F.log_softmax(q_pred_logits, dim=-1)
    return log_ratio.max(dim=-1).values.to(torch.float64)  # [B, K]


def compute_log_p_and_log_q(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    effective_prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-sequence log P(x) and log Q(x) summed across the K generated tokens.

    Returns:
        (log_p_seq, log_q_seq): each Tensor [B] in float64.
    """
    p_pp, q_pp = compute_log_p_and_log_q_per_pos(
        current_logits=current_logits,
        original_logits=original_logits,
        sampled_ids=sampled_ids,
        effective_prefix_len=effective_prefix_len,
    )
    return p_pp.sum(dim=-1), q_pp.sum(dim=-1)


def compute_log_importance_weights(
    current_logits: torch.Tensor,
    original_logits: torch.Tensor,
    sampled_ids: torch.Tensor,
    effective_prefix_len: int,
) -> torch.Tensor:
    """
    Compute per-sequence log importance weights log(P/Q) for generated continuations.

    Returns:
        log_w: Tensor [B] in float64.
    """
    log_p_seq, log_q_seq = compute_log_p_and_log_q(
        current_logits=current_logits,
        original_logits=original_logits,
        sampled_ids=sampled_ids,
        effective_prefix_len=effective_prefix_len,
    )
    return log_p_seq - log_q_seq
