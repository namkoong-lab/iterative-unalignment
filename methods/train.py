#!/usr/bin/env python3

import argparse
import inspect
import os
from contextlib import nullcontext

import torch

from adaptive_reg import AdaptiveReg
from compute_cost import ComputeCostTracker, tracking
from metrics_logging import MetricLogger
from reg_terms import kl_qp_term_per_position
from proposals._base import rollout_batch as reference_rollout_batch
from event_config import parse_event_config_json
from is_weights import (
    compute_log_p_and_log_q_per_pos,
    compute_model_and_reference_logits,
    compute_population_ess_from_log_weights,
    compute_rare_event_ess_from_log_weights,
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
    """Check for a model config and an unsharded weight file."""
    if not os.path.isdir(model_dir):
        return False
    has_config = os.path.exists(os.path.join(model_dir, "config.json"))
    has_weights = any(
        os.path.exists(os.path.join(model_dir, w))
        for w in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


def parse_args():
    event_types = supported_event_types()
    proposal_types = supported_proposal_types()
    parser = argparse.ArgumentParser(description="IU training loop")
    parser.add_argument("context", type=str, help="Prompt/context text")
    parser.add_argument("k", type=int, help="Number of generated tokens per rollout")

    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--steps", type=int, default=300, help="Training steps")
    train_group.add_argument(
        "--eval_steps", type=int, default=0,
        help="Frozen-model evaluation steps after training",
    )
    train_group.add_argument("--batch_size", type=int, default=128, help="Rollout batch size")
    train_group.add_argument("--lr", type=float, default=1e-4, help="Model learning rate")
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
            "multi_token only: load --output_path/trained_model and skip training "
            "if a saved model exists. Evaluate it against the configured indicator."
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
    reg_group = parser.add_argument_group("Dense KL Regularization")
    reg_group.add_argument("--fixed_lambda", type=float, default=0.0, help="Fixed lambda (used when adaptive is off)")
    adapt_group = parser.add_argument_group("Adaptive Lambda (ESS-driven)")
    adapt_group.add_argument(
        "--adaptive_reg_enabled", type=parse_bool, default=True,
        help="Enable ESS-driven adaptive lambda (true/false)",
    )
    adapt_group.add_argument("--ess_target", type=float, default=0.1, help="Target normalized rare-event ESS")
    adapt_group.add_argument(
        "--pop_ess_target", type=float, default=0.005,
        help=(
            "Target normalized population ESS used while adaptive lambda is driven "
            "by population ESS, i.e. before the controller flips to rare-event ESS "
            "(see --rare_event_ess_min_rate). After the flip, --ess_target governs."
        ),
    )
    adapt_group.add_argument(
        "--rare_event_ess_min_rate", type=float, default=0.1,
        help=(
            "Once at least this fraction (default: 0.1 = 10%%) of the batch hits the rare event, "
            "drive adaptive lambda from rare-event-only ESS (targeting --ess_target) "
            "instead of population ESS (targeting --pop_ess_target). Set > 1.0 to disable."
        ),
    )
    adapt_group.add_argument("--dual_lr", type=float, default=0.01, help="Dual optimizer learning rate")
    adapt_group.add_argument(
        "--dual_optimizer", type=str, default="sgd", choices=["sgd", "adam"],
        help="Optimizer type for dual lambda update",
    )
    adapt_group.add_argument("--lambda_floor", type=float, default=0.1, help="Minimum lambda enforced by clamp")
    adapt_group.add_argument("--init_lambda", type=float, default=10.0, help="Initial lambda value")

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

    likelihood_fitting = getattr(proposal, "likelihood_fitting", False)
    if is_training and likelihood_fitting:
        total_steps = min(total_steps, proposal.max_rounds)
        metrics_logger.total_steps = total_steps
        proposal.adaptation_outcome = {"status": "adapting", "round_limit": total_steps}

    print(
        f"\nStarting {phase_name} phase "
        f"({'trainable model' if is_training else 'frozen model'}) for {total_steps} steps."
    )
    if cost_tracker is not None:
        cost_tracker.begin_phase(phase_name)

    for step in range(1, total_steps + 1):
        if cost_tracker is not None:
            cost_tracker.begin_step()
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
        rare_event_count = int(event_indicator.sum().item())
        rare_event_rate = float(event_indicator.float().mean().item())
        stop_for_event_rate, event_rate_stop_reason = False, None
        if is_training and likelihood_fitting:
            stop_for_event_rate, event_rate_stop_reason = proposal.check_event_rate_stop(
                step=step, rare_event_count=rare_event_count, batch_size=generated_tokens.shape[0],
            )
        # Disable autograd during frozen evaluation.
        grad_ctx = nullcontext() if is_training and not likelihood_fitting else torch.no_grad()
        with grad_ctx:
            current_logits, original_logits = compute_model_and_reference_logits(
                proposal=proposal, reference_model=reference_model, sampled_ids=sampled_ids,
                q_forward_ctx=tracking(cost_tracker, "q_forward"),
                p_forward_ctx=tracking(cost_tracker, "p_forward"),
            )
            # CE ranks realized scores when available; negate for minimization.
            is_ce = args.proposal_type in ("CE_ACTIVATION", "CE_LOGIT")
            score_threshold = None
            if likelihood_fitting and proposal.score_mode == "event":
                if hasattr(rare_event, "compute_score") and hasattr(rare_event, "logit_threshold"):
                    main_loss_per_sample = -rare_event.compute_score(generated_tokens)
                    score_threshold = float(rare_event.logit_threshold)
                else:
                    main_loss_per_sample = -event_indicator.float()
                    score_threshold = 1.0
            elif (is_ce or likelihood_fitting) and hasattr(rare_event, "compute_score"):
                main_loss_per_sample = -rare_event.compute_score(
                    generated_tokens=generated_tokens,
                )
            else:
                main_loss_per_sample = rare_event.compute_surrogate_loss(
                    all_logits=current_logits,
                    prefix_len=prefix_len,
                    generated_tokens=generated_tokens,
                )
            kl_loss_per_pos = kl_qp_term_per_position(
                current_logits=current_logits, original_logits=original_logits,
                prefix_len=prefix_len,
            )
            kl_loss_per_sample = kl_loss_per_pos.mean(dim=-1)
            main_loss = main_loss_per_sample.mean()
            kl_loss = kl_loss_per_sample.mean()

        # Score continuations under the proposal that generated them, before updating.
        log_p_per_pos, log_q_per_pos = compute_log_p_and_log_q_per_pos(
            current_logits=current_logits.detach(),
            original_logits=original_logits.detach(),
            sampled_ids=sampled_ids,
            effective_prefix_len=prefix_len,
        )
        log_p_seq = log_p_per_pos.sum(dim=-1)
        log_q_seq = log_q_per_pos.sum(dim=-1)
        log_iw = log_p_seq - log_q_seq

        if is_training and not stop_for_event_rate:
            # CE proposals also use importance weights in their elite update.
            update_kwargs = {}
            if "log_importance_weights" in inspect.signature(proposal.update).parameters:
                update_kwargs["log_importance_weights"] = log_iw
            if likelihood_fitting:
                update_kwargs.update(
                    sampled_ids=sampled_ids, prefix_len=prefix_len,
                    score_threshold=score_threshold, original_logits=original_logits,
                    log_importance_weights=log_iw,
                )
            with tracking(cost_tracker, "ce_fit" if likelihood_fitting else "backprop"):
                total_loss = proposal.update(
                    main_loss_per_sample=main_loss_per_sample,
                    reg_loss=kl_loss_per_sample,
                    lambda_value=lambda_value,
                    **update_kwargs,
                )
        else:
            total_loss = main_loss + lambda_value * kl_loss

        if is_training and likelihood_fitting:
            proposal.last_fit_metrics.update(
                training_event_rate=rare_event_rate,
                target_event_rate=proposal.stop_event_rate,
                event_rate_target_reached=stop_for_event_rate,
            )

        population_ess = compute_population_ess_from_log_weights(log_importance_weights=log_iw)
        rare_event_ess = compute_rare_event_ess_from_log_weights(
            log_importance_weights=log_iw,
            event_indicator=event_indicator,
        )

        if rare_event_rate >= args.rare_event_ess_min_rate:
            controller_ess = rare_event_ess
            controller_target = float(args.ess_target)
        else:
            controller_ess = population_ess
            controller_target = float(args.pop_ess_target)

        if is_training and adaptive_reg is not None:
            adaptive_reg.step(controller_ess, ess_target=controller_target)

        should_stop_now, reason = stop_for_event_rate, event_rate_stop_reason
        if is_training and not should_stop_now:
            should_stop_now, reason = proposal.should_stop(
                step=step,
                rare_event_ess=rare_event_ess,
                rare_event_count=rare_event_count,
                rare_event_rate=rare_event_rate,
                rare_event_ess_min_rate=args.rare_event_ess_min_rate,
                ess_target=float(args.ess_target),
            )

        metrics_logger.record_step(
            step=step,
            total_loss=total_loss,
            main_loss=main_loss,
            kl_loss=kl_loss,
            kl_loss_per_pos=kl_loss_per_pos.detach(),
            lambda_value=lambda_value,
            population_ess=population_ess,
            rare_event_ess=rare_event_ess,
            controller_ess=controller_ess,
            generated_tokens=generated_tokens,
            event_indicator=event_indicator,
            log_importance_weights=log_iw,
            log_p_seq=log_p_seq,
            log_q_seq=log_q_seq,
            log_p_per_pos=log_p_per_pos,
            log_q_per_pos=log_q_per_pos,
            proposal_metrics=(proposal.last_fit_metrics if likelihood_fitting and is_training else None),
            on_estimate_ready=(
                (lambda: cost_tracker.end_step(step, is_training=is_training))
                if cost_tracker is not None else None
            ),
        )
        if should_stop_now:
            print(f"[{phase_name}] {reason} at step {step}; proceeding to eval.")
            break

    if is_training and likelihood_fitting:
        if not proposal.adaptation_outcome.get("target_reached", False):
            proposal.adaptation_outcome["status"] = "ess_stop" if should_stop_now else "round_limit_reached"
        proposal.adaptation_outcome["updates_completed"] = proposal.update_steps

    # Save remaining metrics, including after early stopping.
    metrics_logger.flush()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Resolve saved-model reuse before constructing the proposal.
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
    # Load the frozen reference from --model_id, independently of proposal reuse.
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
    def naive_mc_batch():
        sampled_ids, prefix_len = reference_rollout_batch(
            model=reference_model,
            prefix_ids=prefix_ids,
            batch_size=args.batch_size,
            k=args.k,
            tokenizer=proposal.tokenizer,
        )
        indicator = rare_event.compute_indicator(generated_tokens=sampled_ids[:, prefix_len:])
        return indicator.float().mean().item()

    cost_tracker.calibrate_p_inference(naive_mc_batch)

    shared_metadata = {
        "proposal_type": args.proposal_type,
        "context": args.context,
        "k": args.k,
        "batch_size": args.batch_size,
        "regularizer": "kl_qp_dense_mean",
        "ordinary_is_unclipped": True,
        "ce_importance_weighted": getattr(proposal, "importance_weighted", None),
        "ce_fit_method": getattr(proposal, "fit_method", None),
        "event_type": args.event_type,
        "event_config_json": args.event_config_json,
        "gt_prob": float(rare_event.gt_prob),
        "adaptive_reg_enabled": bool(args.adaptive_reg_enabled),
        "ess_target": float(args.ess_target),
        "pop_ess_target": float(args.pop_ess_target),
        "use_chat_template": bool(getattr(args, "use_chat_template", False)),
    }

    if reuse_active:
        # Preserve the saved proposal and its training metrics.
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

        # Separate evaluation outputs for indicators sharing one proposal.
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
            # Include all frozen-proposal evaluation batches.
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
            # Report the KL term using the final training lambda.
            fixed_lambda=(
                adaptive_reg.current_lambda if adaptive_reg is not None
                else args.fixed_lambda
            ),
            cost_tracker=cost_tracker,
        )

        eval_hit_rates_path = os.path.join(eval_output_dir, "average_hit_rates.txt")
        eval_logger.save_hit_rates_summary(file_path=eval_hit_rates_path)

        eval_log_error_path = os.path.join(eval_output_dir, "average_log_error.txt")
        eval_logger.save_log_error_summary(file_path=eval_log_error_path)

        if reuse_active:
            # Keep the original training-run compute_cost.txt intact.
            cost_txt_path = cost_tracker.save(eval_output_dir)
        else:
            cost_txt_path = cost_tracker.save(output_dir)
        if cost_txt_path is not None:
            print(f"Saved compute-cost summary to: {cost_txt_path}")


if __name__ == "__main__":
    main()
