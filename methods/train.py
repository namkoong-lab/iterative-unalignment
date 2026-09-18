#!/usr/bin/env python3

import argparse
import inspect
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F

from adaptive_reg import AdaptiveReg
from compute_cost import ComputeCostTracker, tracking
from metrics_logging import MetricLogger
from reg_terms import alpha_reg_term_per_position, kl_qp_term_per_position
from proposals._base import rollout_batch as reference_rollout_batch
from event_config import parse_event_config_json
from is_weights import (
    compute_log_p_and_log_q_per_pos,
    compute_model_and_reference_logits,
    compute_population_ess_from_log_weights,
    compute_rare_event_ess_from_log_weights,
    compute_worst_case_log_ratio_per_pos,
    generated_predictor_logits,
)
from model_loading import add_model_loading_args, load_reference_model
from output_paths import multi_token_eval_dir, resolve_output_dir, write_run_params_txt
from train_lib import (
    add_proposal_args,
    build_proposal,
    build_rare_event,
    is_multi_token_event,
    parse_bool,
    supported_event_types,
    supported_proposal_types,
)


def _has_saved_model(model_dir: str) -> bool:
    """True iff model_dir holds a loadable full HF model (config + weights)."""
    if not os.path.isdir(model_dir):
        return False
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_weights = any(
        os.path.exists(os.path.join(model_dir, w))
        for w in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def _aggregate_per_position(x: torch.Tensor, method: str, beta: float) -> torch.Tensor:
    """Reduce a per-position reg tensor [B, K] to per-sample [B].

    method='mean': arithmetic mean over k.
    method='max':  soft-max weighted aggregate Σ_k softmax(β·x)_k · x_k.
        Differentiable; gradient flows through both the weights and the values.
        β→0 recovers mean, β→∞ recovers true max.
    """
    if method == "mean":
        return x.mean(dim=-1)
    if method == "max":
        weights = torch.softmax(beta * x, dim=-1)
        return (weights * x).sum(dim=-1)
    raise ValueError(f"unknown reg_aggregate: {method!r}")


def _aggregate_reg_terms(
    alpha_per_pos: torch.Tensor,
    kl_per_pos: torch.Tensor,
    *,
    kl_coef: float,
    method: str,
    beta: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce two [B, K] reg tensors (alpha, KL) to per-sample [B] each.

    mode='independent': each term is aggregated separately, so the soft-max in
        'max' mode picks the worst k for each term on its own.
    mode='joint': in 'max' mode the soft-max weights are computed once on the
        combined per-position signal (alpha + kl_coef * kl), then applied to
        both terms — so the gradient on both reg terms concentrates on the
        position that violates them together the most. For method='mean' the
        result is identical to 'independent' since mean is linear.
    """
    if method == "mean" or mode == "independent":
        return (
            _aggregate_per_position(alpha_per_pos, method, beta),
            _aggregate_per_position(kl_per_pos, method, beta),
        )
    if mode == "joint":
        combined_per_pos = alpha_per_pos + kl_coef * kl_per_pos
        weights = torch.softmax(beta * combined_per_pos, dim=-1)
        return (weights * alpha_per_pos).sum(dim=-1), (weights * kl_per_pos).sum(dim=-1)
    raise ValueError(f"unknown reg_aggregate_mode: {mode!r}")


def parse_args():
    event_types = supported_event_types()
    proposal_types = supported_proposal_types()
    parser = argparse.ArgumentParser(description="IU training loop")
    parser.add_argument("context", type=str, help="Prompt/context text")
    parser.add_argument("k", type=int, help="Number of generated tokens per rollout")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--steps", type=int, default=200, help="Training steps")
    train_group.add_argument(
        "--eval_steps", type=int, default=0,
        help="Frozen-model evaluation steps after training",
    )
    train_group.add_argument("--batch_size", type=int, default=8, help="Rollout batch size")
    train_group.add_argument("--lr", type=float, default=5e-6, help="Model learning rate")
    train_group.add_argument("--seed", type=int, default=120, help="Random seed")
    train_group.add_argument("--log_every", type=int, default=1, help="Print frequency in steps")
    train_group.add_argument(
        "--proposal_type", type=str, default=proposal_types[0], choices=proposal_types,
        help="Proposal implementation type",
    )
    train_group.add_argument(
        "--output_path", type=str, default="./runs",
        help="Parent output directory",
    )
    train_group.add_argument(
        "--reuse_trained_model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "multi_token only: if --output_path already contains a saved trained_model/, "
            "skip training and run eval-only against this run's indicator, loading the "
            "proposal from that dir. The check is scoped to this run's own output dir; "
            "no other directory is consulted. The proposal is determined by the surrogate, "
            "so give same-surrogate runs the same --output_path to train once and re-eval."
        ),
    )

    obj_group = parser.add_argument_group("Objective")
    obj_group.add_argument(
        "--event_type", type=str, default=event_types[0], choices=event_types,
        help="Rare event type",
    )
    obj_group.add_argument(
        "--event_config_json", type=str, required=True,
        help='JSON event config, e.g. \'{"surrogate":"sum","token":"elegant"}\'',
    )
    reg_group = parser.add_argument_group("Alpha Regularization")
    reg_group.add_argument("--alpha_reg_moment", type=float, default=2.0, help="Moment for alpha reg")
    reg_group.add_argument("--fixed_lambda", type=float, default=0.0, help="Fixed lambda (used when adaptive is off)")
    reg_group.add_argument(
        "--kl_reg_coef", type=float, default=0.0,
        help=(
            "Relative weight of KL(Q || P) within the λ-scaled reg term. The "
            "alpha reg (log E_Q[(P/Q)^α]) and KL are summed as "
            "alpha_reg + kl_reg_coef * kl_reg, then both are scaled by the "
            "adaptive λ. Both terms are log-scale divergence measures so a "
            "coef of 1.0 weights them roughly evenly. Per-sample KL reduced "
            "over K positions per --reg_aggregate. 0 disables KL (default)."
        ),
    )
    reg_group.add_argument(
        "--reg_aggregate", type=str, default="mean", choices=["mean", "max"],
        help=(
            "How to reduce the per-position [B, K] reg tensors (both alpha χ² and "
            "KL) to per-sample [B]. 'mean' averages over k. 'max' uses a soft-max "
            "weighted aggregate Σ_k softmax(β·x)_k · x_k with β=--reg_aggregate_beta, "
            "which is differentiable and concentrates gradient on the worst step "
            "as β grows."
        ),
    )
    reg_group.add_argument(
        "--reg_aggregate_beta", type=float, default=10.0,
        help=(
            "Inverse-temperature β for --reg_aggregate=max. β→0 recovers mean, "
            "β→∞ recovers true max. Ignored when --reg_aggregate=mean."
        ),
    )
    reg_group.add_argument(
        "--reg_aggregate_mode", type=str, default="independent",
        choices=["independent", "joint"],
        help=(
            "How the alpha and KL reg terms share (or don't) the soft-max in "
            "--reg_aggregate=max. 'independent': each term takes softmax over its "
            "own per-position values, so each focuses on the k that violates it "
            "alone (current behavior). 'joint': softmax weights are computed once "
            "from the combined per-position signal (alpha + kl_reg_coef * kl) and "
            "applied to both terms, so they focus together on the k that violates "
            "them together the most. No effect when --reg_aggregate=mean."
        ),
    )
    adapt_group = parser.add_argument_group("Adaptive Lambda (ESS-driven)")
    adapt_group.add_argument(
        "--adaptive_reg_enabled", type=parse_bool, default=True,
        help="Enable ESS-driven adaptive lambda (true/false)",
    )
    adapt_group.add_argument("--ess_target", type=float, default=0.2, help="Target normalized rare-event ESS")
    adapt_group.add_argument(
        "--pop_ess_target", type=float, default=0.005,
        help=(
            "Target normalized population ESS used while adaptive lambda is driven "
            "by population ESS, i.e. before the controller flips to rare-event ESS "
            "(see --rare_event_ess_min_count). After the flip, --ess_target governs."
        ),
    )
    adapt_group.add_argument(
        "--rare_event_ess_min_count", type=int, default=5,
        help=(
            "Once strictly more than this many rare-event hits appear in a batch, "
            "drive adaptive lambda from rare-event-only ESS (targeting --ess_target) "
            "instead of population ESS (targeting --pop_ess_target). Set to a very "
            "large value to disable."
        ),
    )
    adapt_group.add_argument("--dual_lr", type=float, default=0.05, help="Dual optimizer learning rate")
    adapt_group.add_argument(
        "--dual_optimizer", type=str, default="sgd", choices=["sgd", "adam"],
        help="Optimizer type for dual lambda update",
    )
    adapt_group.add_argument("--lambda_floor", type=float, default=0.01, help="Minimum lambda enforced by clamp")
    adapt_group.add_argument("--init_lambda", type=float, default=0.01, help="Initial lambda value")

    add_proposal_args(parser)
    add_model_loading_args(parser)
    return parser.parse_args()


def _run_phase(
    *,
    phase_name: str,
    total_steps: int,
    is_training: bool,
    args,
    proposal,
    reference_model,
    prefix_ids: torch.Tensor,
    rare_event,
    metrics_logger: MetricLogger,
    adaptive_reg: AdaptiveReg | None,
    fixed_lambda: float,
    cost_tracker: ComputeCostTracker | None = None,
) -> None:
    if total_steps <= 0:
        return

    print(
        f"\nStarting {phase_name} phase "
        f"({'trainable model' if is_training else 'frozen model'}) for {total_steps} steps."
    )
    if cost_tracker is not None:
        cost_tracker.begin_phase(phase_name)

    for step in range(1, total_steps + 1):
        if adaptive_reg is not None:
            lambda_value = adaptive_reg.current_lambda
        else:
            lambda_value = fixed_lambda

        with tracking(cost_tracker, "q_inference"):
            sampled_ids, prefix_len = proposal.rollout_batch(
                prefix_ids=prefix_ids, batch_size=args.batch_size, k=args.k,
            )
        generated_tokens = sampled_ids[:, prefix_len:]

        event_indicator = rare_event.compute_indicator(generated_tokens=generated_tokens)
        event_rate = float(event_indicator.float().mean().item())
        rare_event_count = int(event_indicator.sum().item())
        diag_counts = rare_event.diagnostic_counts(
            generated_tokens=generated_tokens, event_indicator=event_indicator,
        )

        # Eval freezes the proposal, so disable autograd to skip building
        # the graph; train needs grads through the proposal's compute_logits.
        grad_ctx = nullcontext() if is_training else torch.no_grad()
        with grad_ctx:
            current_logits, original_logits = compute_model_and_reference_logits(
                proposal=proposal, reference_model=reference_model, sampled_ids=sampled_ids,
                q_forward_ctx=tracking(cost_tracker, "q_forward"),
                p_forward_ctx=tracking(cost_tracker, "p_forward"),
            )
            # CE/CEM is gradient-free, so for events that expose a literal
            # climbing score (e.g. BoW's discrete classifier logit) it ranks
            # candidates by that hard score on the realized tokens rather than
            # the differentiable surrogate the IU proposals need. Negated to a
            # lower-is-better cost, matching the surrogate-loss convention the
            # rest of the loop (and the CEM elite argsort) expects.
            is_ce = args.proposal_type in ("CE_ACTIVATION", "CE_LOGIT")
            if is_ce and hasattr(rare_event, "compute_score"):
                main_loss_per_sample = -rare_event.compute_score(
                    generated_tokens=generated_tokens,
                )
            else:
                main_loss_per_sample = rare_event.compute_surrogate_loss(
                    all_logits=current_logits,
                    prefix_len=prefix_len,
                    generated_tokens=generated_tokens,
                )
            # Per-position log E_Q[(P/Q)^α] (=log(1+χ²) at α=2) reduced to
            # per-sample alpha_loss for the update/logging path; raw [B, K]
            # tensor goes to metrics.
            alpha_loss_per_pos = alpha_reg_term_per_position(
                current_logits=current_logits, original_logits=original_logits,
                prefix_len=prefix_len, moment=args.alpha_reg_moment,
            )
            # KL(Q||P) is always computed for logging; folded into main loss
            # only when the fixed coefficient is non-zero (then it is NOT
            # scaled by the adaptive-λ controller — a constant pull-back force).
            kl_loss_per_pos = kl_qp_term_per_position(
                current_logits=current_logits, original_logits=original_logits,
                prefix_len=prefix_len,
            )
            alpha_loss, kl_loss_per_sample = _aggregate_reg_terms(
                alpha_loss_per_pos, kl_loss_per_pos,
                kl_coef=float(args.kl_reg_coef),
                method=args.reg_aggregate,
                beta=args.reg_aggregate_beta,
                mode=args.reg_aggregate_mode,
            )
            main_loss = main_loss_per_sample.mean()
            kl_loss = kl_loss_per_sample.mean()

            # Token-event diagnostic: Q(rare_token | history) at each generated
            # position, [B, K]. Only defined when the event names a single token.
            rare_token_id = getattr(rare_event, "token_id", None)
            if rare_token_id is not None:
                q_generated_logits = generated_predictor_logits(
                    current_logits, prefix_len,
                )  # [B, K, V]
                rare_token_prob_per_pos = F.softmax(
                    q_generated_logits, dim=-1,
                )[:, :, int(rare_token_id)].detach()
            else:
                rare_token_prob_per_pos = None

        # Combined reg: both terms are log-scale divergence measures, so they
        # share the adaptive λ. kl_reg_coef is the relative weight between them.
        combined_reg_per_sample = alpha_loss + args.kl_reg_coef * kl_loss_per_sample

        # Per-sequence importance weights log(P/Q) over the generated
        # continuation. Computed before the proposal update so CE/CEM proposals
        # can weight their elite mean by w = P/Q; also feeds the ESS diagnostics.
        log_p_per_pos, log_q_per_pos = compute_log_p_and_log_q_per_pos(
            current_logits=current_logits.detach(),
            original_logits=original_logits.detach(),
            sampled_ids=sampled_ids,
            effective_prefix_len=prefix_len,
        )
        log_p_seq = log_p_per_pos.sum(dim=-1)
        log_q_seq = log_q_per_pos.sum(dim=-1)
        log_iw = log_p_seq - log_q_seq

        # Population worst-weight upper bound: the unconstrained per-position
        # optimum max_v(log P(v|x_<k) - log Q(v|x_<k)) summed over k upper-bounds
        # the log weight the realized prefix path could yield. The realized
        # in-event worst weight is derived in metrics from log_iw + event mask.
        worst_case_log_ratio_per_pos = compute_worst_case_log_ratio_per_pos(
            current_logits=current_logits.detach(),
            original_logits=original_logits.detach(),
            effective_prefix_len=prefix_len,
        )
        worst_case_cum_log_weight = worst_case_log_ratio_per_pos.sum(dim=-1)  # [B]

        if is_training:
            # IU gradient proposals take (main_loss, reg, lambda); CE/CEM
            # proposals additionally consume log_importance_weights for the
            # importance-weighted elite mean. Pass it only when update() takes it.
            update_kwargs = {}
            if "log_importance_weights" in inspect.signature(proposal.update).parameters:
                update_kwargs["log_importance_weights"] = log_iw
            with tracking(cost_tracker, "backprop"):
                total_loss = proposal.update(
                    main_loss_per_sample=main_loss_per_sample,
                    reg_loss=combined_reg_per_sample,
                    lambda_value=lambda_value,
                    **update_kwargs,
                )
        else:
            total_loss = main_loss + lambda_value * combined_reg_per_sample.mean()

        population_ess = compute_population_ess_from_log_weights(log_importance_weights=log_iw)
        rare_event_ess = compute_rare_event_ess_from_log_weights(
            log_importance_weights=log_iw,
            event_indicator=event_indicator,
        )

        if rare_event_count > args.rare_event_ess_min_count:
            controller_ess = rare_event_ess
            controller_target = float(args.ess_target)
        else:
            controller_ess = population_ess
            controller_target = float(args.pop_ess_target)

        if is_training and adaptive_reg is not None:
            adaptive_reg.step(controller_ess, ess_target=controller_target)

        if cost_tracker is not None:
            cost_tracker.end_step(step, is_training=is_training)

        metrics_logger.record_step(
            step=step,
            total_loss=total_loss,
            main_loss=main_loss,
            alpha_loss=alpha_loss.mean(),
            kl_loss=kl_loss,
            kl_coef=float(args.kl_reg_coef),
            alpha_loss_per_pos=alpha_loss_per_pos.detach(),
            kl_loss_per_pos=kl_loss_per_pos.detach(),
            rare_token_prob_per_pos=rare_token_prob_per_pos,
            lambda_value=lambda_value,
            population_ess=population_ess,
            rare_event_ess=rare_event_ess,
            controller_ess=controller_ess,
            generated_tokens=generated_tokens,
            rare_event=rare_event,
            event_indicator=event_indicator,
            non_cheat_count=diag_counts.get("non_cheat_count"),
            log_importance_weights=log_iw,
            log_p_seq=log_p_seq,
            log_q_seq=log_q_seq,
            log_p_per_pos=log_p_per_pos,
            log_q_per_pos=log_q_per_pos,
            worst_case_cum_log_weight=worst_case_cum_log_weight,
            proposal_metrics=None,
        )

        if is_training:
            should_stop_now, reason = proposal.should_stop(
                step=step,
                rare_event_ess=rare_event_ess,
                rare_event_count=rare_event_count,
                rare_event_ess_min_count=args.rare_event_ess_min_count,
                ess_target=float(args.ess_target),
            )
            if should_stop_now:
                print(
                    f"[{phase_name}] {reason} at step {step}; "
                    "stopping training and proceeding to eval."
                )
                break

    # Ensure the most recent steps land on disk even when log_every > 1 or the
    # loop exited early via should_stop.
    metrics_logger.flush()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Eval-only reuse (multi_token): if this run's own output dir already holds a
    # saved trained_model/, load the proposal from it and skip training. Decided
    # before build_proposal so the trained weights become the proposal source.
    # Scoped strictly to args.output_path — no other directory is consulted.
    is_mt = is_multi_token_event(str(args.event_type))
    reuse_model_dir = os.path.join(os.path.abspath(str(args.output_path)), "trained_model")
    reuse_active = (
        bool(getattr(args, "reuse_trained_model", False))
        and is_mt
        and _has_saved_model(reuse_model_dir)
    )
    if reuse_active:
        if args.use_lora:
            raise ValueError(
                "--reuse_trained_model with --use_lora is not supported: the saved "
                "artifact is a LoRA adapter, not a base model. Re-run with --no-use_lora."
            )
        args.load_proposal_from = reuse_model_dir

    proposal = build_proposal(proposal_type=args.proposal_type, args=args)
    # Reference (frozen P) always loads the base --model_id, never the trained
    # proposal — the reg terms compare Q against the untrained base.
    reference_model = load_reference_model(args, device=proposal.device)
    proposal.build_optimizer(lr=args.lr)

    prefix_ids = proposal.encode_context(args.context)
    event_config = parse_event_config_json(args.event_config_json)
    rare_event = build_rare_event(
        event_type=args.event_type, event_config=event_config, tokenizer=proposal.tokenizer,
    )

    adaptive_reg: AdaptiveReg | None = None
    if args.adaptive_reg_enabled:
        adaptive_reg = AdaptiveReg(
            ess_target=args.ess_target,
            init_lambda=args.init_lambda,
            dual_lr=args.dual_lr,
            lambda_floor=args.lambda_floor,
            dual_optimizer_type=args.dual_optimizer,
            device=proposal.device,
        )

    if adaptive_reg is not None:
        initial_lambda = adaptive_reg.current_lambda
    else:
        initial_lambda = args.fixed_lambda

    print(f"Loaded model source: {proposal.model_source}")
    print(f"use_chat_template: {getattr(proposal, 'use_chat_template', False)}")
    print(f"Using device: {proposal.device}")
    print(f"LoRA enabled: {proposal.uses_lora}")
    print(f"Model class: {proposal.model_class_name}")
    if adaptive_reg is not None:
        print("Reg scheme: AdaptiveReg (ESS-driven dual lambda)")
        print(
            f"  initial_lambda={initial_lambda:.6f}  ess_target={float(args.ess_target):.4f}"
            f"  pop_ess_target={float(args.pop_ess_target):.4f}"
        )
    else:
        print("Reg scheme: fixed lambda")
        print(f"  fixed_lambda={initial_lambda:.6f}")
    _thr = getattr(rare_event, "threshold", None)
    _thr_s = f" threshold={float(_thr):.4f}" if _thr is not None else ""
    print(
        f"Rare event: type={args.event_type} "
        f"token={rare_event.token!r} surrogate={rare_event.surrogate!r} "
        f"gt_prob={float(rare_event.gt_prob):.6f}{_thr_s}"
    )

    output_dir = resolve_output_dir(args, rare_event)
    output_pickle_path = os.path.join(output_dir, "metrics_atomic.pkl")

    cost_tracker = ComputeCostTracker(
        device=proposal.device,
        batch_size=args.batch_size,
        k=args.k,
        output_dir=output_dir,
    )
    # Naive-MC generate-one-trajectory cost is not part of the IU step, so
    # time a few frozen-p rollouts once. Everything else is timed in-loop.
    cost_tracker.calibrate_p_inference(
        lambda: reference_rollout_batch(
            model=reference_model,
            prefix_ids=prefix_ids,
            batch_size=args.batch_size,
            k=args.k,
            tokenizer=proposal.tokenizer,
        )
    )

    shared_metadata = {
        "proposal_type": args.proposal_type,
        "context": args.context,
        "k": args.k,
        "batch_size": args.batch_size,
        "alpha_reg_moment": float(args.alpha_reg_moment),
        "event_type": args.event_type,
        "event_config_json": args.event_config_json,
        "gt_prob": float(rare_event.gt_prob),
        "adaptive_reg_enabled": bool(args.adaptive_reg_enabled),
        "ess_target": float(args.ess_target),
        "pop_ess_target": float(args.pop_ess_target),
        "use_chat_template": bool(getattr(args, "use_chat_template", False)),
    }

    if reuse_active:
        # Trained proposal already loaded from reuse_model_dir; do not retrain or
        # overwrite the training run's params/metrics that live in output_dir.
        print(
            f"[reuse] Found trained model at {reuse_model_dir}; skipping training, "
            "running eval-only against this run's indicator."
        )
    else:
        write_run_params_txt(args, rare_event, output_dir, output_pickle_path)
        _run_phase(
            phase_name="train",
            total_steps=args.steps,
            is_training=True,
            args=args,
            proposal=proposal,
            reference_model=reference_model,
            prefix_ids=prefix_ids,
            rare_event=rare_event,
            metrics_logger=MetricLogger(
                output_pickle_path=output_pickle_path,
                metadata={**shared_metadata, "steps": args.steps, "phase": "train"},
                log_every=args.log_every,
                total_steps=args.steps,
                tokenizer=proposal.tokenizer,
                num_sample_sentences=3,
            ),
            adaptive_reg=adaptive_reg,
            fixed_lambda=args.fixed_lambda,
            cost_tracker=cost_tracker,
        )
        train_cost_path = cost_tracker.save(output_dir)
        if train_cost_path is not None:
            print(f"Saved train compute-cost summary to: {train_cost_path}")

        saved_model_dir = proposal.save(output_dir)
        print(f"Saved trained proposal artifacts to: {saved_model_dir}")

    if args.eval_steps > 0:
        proposal.freeze_for_eval()

        # multi_token: per-indicator eval subdir so multiple indicators reusing
        # one proposal (same output_dir) don't overwrite each other.
        if is_mt:
            eval_output_dir = multi_token_eval_dir(output_dir, rare_event)
        else:
            eval_output_dir = os.path.join(output_dir, "eval")
            os.makedirs(eval_output_dir, exist_ok=True)
        eval_output_pickle_path = os.path.join(eval_output_dir, "metrics_atomic.pkl")
        write_run_params_txt(args, rare_event, eval_output_dir, eval_output_pickle_path)

        eval_logger = MetricLogger(
            output_pickle_path=eval_output_pickle_path,
            metadata={**shared_metadata, "steps": args.eval_steps, "phase": "eval"},
            log_every=args.log_every,
            total_steps=args.eval_steps,
            tokenizer=proposal.tokenizer,
            num_sample_sentences=3,
            # Frozen model — no transient to skip, so every eval step counts.
            burn_in=0,
        )
        _run_phase(
            phase_name="eval",
            total_steps=args.eval_steps,
            is_training=False,
            args=args,
            proposal=proposal,
            reference_model=reference_model,
            prefix_ids=prefix_ids,
            rare_event=rare_event,
            metrics_logger=eval_logger,
            adaptive_reg=None,
            # Eval freezes the proposal; for the lambda * alpha_reg term we replay
            # the trained adaptive lambda when applicable, else fall back to fixed.
            fixed_lambda=(
                adaptive_reg.current_lambda if adaptive_reg is not None
                else args.fixed_lambda
            ),
            cost_tracker=cost_tracker,
        )

        eval_hit_rates_path = os.path.join(eval_output_dir, "average_hit_rates.txt")
        eval_logger.save_hit_rates_summary(file_path=eval_hit_rates_path)
        print(f"Saved eval hit-rate summary to: {eval_hit_rates_path}")

        eval_log_error_path = os.path.join(eval_output_dir, "average_log_error.txt")
        eval_logger.save_log_error_summary(file_path=eval_log_error_path)
        print(f"Saved eval log-error summary to: {eval_log_error_path}")

        if reuse_active:
            # Keep the original training-run compute_cost.txt intact.
            cost_txt_path = cost_tracker.save(eval_output_dir)
        else:
            cost_txt_path = cost_tracker.save(output_dir)
        if cost_txt_path is not None:
            print(f"Saved compute-cost summary to: {cost_txt_path}")
            print(cost_tracker.format_summary())
    elif not reuse_active:
        print(cost_tracker.format_summary())


if __name__ == "__main__":
    main()
