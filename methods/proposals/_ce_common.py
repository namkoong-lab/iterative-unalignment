"""Shared CE argparse flags and early-stop monitor."""

from __future__ import annotations

import argparse

import torch


def elite_weights(indices, log_importance_weights, *, weighted):
    if not weighted:
        return torch.full((indices.numel(),), 1.0 / indices.numel(),
                          device=indices.device, dtype=torch.float64)
    if log_importance_weights is None:
        raise ValueError("Weighted CE requires log importance weights")
    if log_importance_weights.ndim != 1:
        raise ValueError("Log importance weights must have shape [batch_size]")
    logw = log_importance_weights.detach().to(indices.device)[indices].double()
    if not torch.isfinite(logw).all():
        raise ValueError("CE requires finite elite log importance weights")
    return torch.softmax(logw, dim=0)


# Skip early-stop checks until CEM has had a few updates.
ESS_WARMUP_STEPS = 10
# Stop after this many consecutive post-warmup steps with collapsed in-event ESS.
ESS_CONSECUTIVE_LOW = 2


def add_ce_argparse_group(parser: argparse.ArgumentParser) -> None:
    """Register --ce_* args. Idempotent: safe to call from both CE proposals."""
    if getattr(parser, "_ce_args_registered", False):
        return
    parser._ce_args_registered = True

    from train_lib import parse_bool

    group = parser.add_argument_group(
        "CE_ACTIVATION / CE_LOGIT proposal (used when --proposal_type is CE_ACTIVATION or CE_LOGIT)"
    )
    group.add_argument(
        "--ce_importance_weighted", type=parse_bool, default=None,
        help="Weight elites by p/q (default: true for likelihood fitting, false for Gaussian search).",
    )
    group.add_argument(
        "--ce_elite_ratio", type=float, default=0.1,
        help="Fraction of population to use as elites for the EMA update.",
    )
    group.add_argument(
        "--ce_smoothing", type=float, default=0.7,
        help="EMA smoothing alpha: mu <- alpha * elite_mean + (1-alpha) * mu.",
    )
    group.add_argument(
        "--ce_sigma_init", type=float, default=0.02,
        help="Fixed standard deviation of the Gaussian search distribution.",
    )
    group.add_argument(
        "--ce_steering_init_scale", type=float, default=0.0,
        help="Initial value for every entry of mu (0.0 = start from the un-steered model).",
    )
    group.add_argument(
        "--ce_generation_temperature", type=float, default=1.0,
        help="Temperature applied during the K-step rollout under the steered model.",
    )
    group.add_argument(
        "--ce_eval_use_mean_only", type=parse_bool, default=True,
        help=(
            "When the proposal is frozen for eval, draw all rollouts with sigma=0 (mu only). "
            "Set false to keep stochastic candidate sampling during eval."
        ),
    )
    group.add_argument(
        "--ce_early_stop_on_low_ess", type=parse_bool, default=True,
        help=(
            f"Stop CE training when at least two samples hit the event and the in-event ESS "
            f"drops below the ESS target for {ESS_CONSECUTIVE_LOW} consecutive steps "
            "(after warmup), then proceed to eval. "
            "Only applies when proposal_type is CE_ACTIVATION or CE_LOGIT."
        ),
    )


class CEEarlyStopMonitor:
    """Stop after sustained in-event ESS below the configured target."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._consecutive_low = 0

    def check(
        self,
        *,
        step: int,
        rare_event_ess: float | None,
        rare_event_count: int,
        ess_target: float,
    ) -> tuple[bool, str | None]:
        if not self.enabled or step <= ESS_WARMUP_STEPS:
            return False, None

        if (
            rare_event_count >= 2
            and rare_event_ess is not None
            and rare_event_ess < ess_target
        ):
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        if self._consecutive_low >= ESS_CONSECUTIVE_LOW:
            return True, (
                f"event found ({rare_event_count} hits) and "
                f"in-event ESS {rare_event_ess:.4f} fell below the ESS target {ess_target:.4f} "
                f"for {self._consecutive_low} consecutive step(s) "
                f"(>= {ESS_CONSECUTIVE_LOW}) (warmup={ESS_WARMUP_STEPS})"
            )
        return False, None
