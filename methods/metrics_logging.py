import math
import os
import pickle
from typing import Any

import numpy as np
import torch
from is_weights import ordinary_is_log_estimate


def _naive_mc_n_for_log_mae(
    p: float,
    target_mae: float,
    cap: float,
    num_trials: int = 20000,
    seed: int = 0,
    n_ceiling: int = 10**15,
) -> int | None:
    """Smallest N such that naive-MC E[min(|log10(k/N) - log10(p)|, cap)] ≤ target_mae,
    with k ~ Binomial(N, p) approximated by Poisson(N·p) (Gaussian above 1e7).
    Returns None if target is unachievable within [1, n_ceiling] or inputs invalid."""
    if p is None or p <= 0.0 or target_mae is None:
        return None
    if target_mae >= cap:
        return 1
    rng = np.random.default_rng(seed)
    log_p = math.log10(p)

    def mae_at(N: int) -> float:
        if N < 1:
            return cap
        lam = N * p
        if lam < 1e7:
            k = rng.poisson(lam, num_trials).astype(np.float64)
        else:
            k = rng.normal(lam, math.sqrt(lam * (1.0 - p)), num_trials)
            np.maximum(k, 0.0, out=k)
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.where(
                k > 0,
                np.abs(np.log10(k / float(N)) - log_p),
                cap,
            )
        return float(np.minimum(err, cap).mean())

    lo, hi = 1, int(n_ceiling)
    if mae_at(hi) > target_mae:
        return None
    if mae_at(lo) <= target_mae:
        return lo
    while hi / lo > 1.05:
        mid = max(lo + 1, int(math.sqrt(float(lo) * float(hi))))
        if mae_at(mid) > target_mae:
            lo = mid
        else:
            hi = mid
    return int(math.sqrt(float(lo) * float(hi)))


def _build_samples_list(
    *,
    log_importance_weights: torch.Tensor,
    event_indicator_int: torch.Tensor,
    log_p_seq: torch.Tensor | None,
    log_q_seq: torch.Tensor | None,
    log_p_per_pos: torch.Tensor | None,
    log_q_per_pos: torch.Tensor | None,
) -> list[dict[str, Any]]:
    """Per-example records: each entry packs the scalars and per-position lists
    for one sequence (log P, log Q, log P/Q, event flag, per-token log P/Q)."""
    log_iw_list = log_importance_weights.detach().cpu().tolist()
    event_list = event_indicator_int.detach().cpu().tolist()
    B = len(log_iw_list)
    log_p_list = log_p_seq.detach().cpu().tolist() if log_p_seq is not None else [None] * B
    log_q_list = log_q_seq.detach().cpu().tolist() if log_q_seq is not None else [None] * B
    log_p_pp = log_p_per_pos.detach().cpu().tolist() if log_p_per_pos is not None else [None] * B
    log_q_pp = log_q_per_pos.detach().cpu().tolist() if log_q_per_pos is not None else [None] * B

    samples: list[dict[str, Any]] = []
    for i in range(B):
        samples.append({
            "log_p": float(log_p_list[i]) if log_p_list[i] is not None else None,
            "log_q": float(log_q_list[i]) if log_q_list[i] is not None else None,
            "log_iw": float(log_iw_list[i]),
            "event": int(event_list[i]),
            "log_p_per_pos": (
                [float(x) for x in log_p_pp[i]] if log_p_pp[i] is not None else None
            ),
            "log_q_per_pos": (
                [float(x) for x in log_q_pp[i]] if log_q_pp[i] is not None else None
            ),
        })
    return samples


class MetricLogger:
    """
    Centralized logging/persistence for training metrics.

    Stores only atomic fields required to reconstruct downstream stats and plots.
    """

    def __init__(
        self,
        *,
        output_pickle_path: str | None,
        metadata: dict[str, Any],
        log_every: int,
        total_steps: int,
        tokenizer: Any | None = None,
        num_sample_sentences: int = 3,
        burn_in: int = 20,
        window: int = 50,
    ) -> None:
        self.output_pickle_path = output_pickle_path
        self.log_every = int(log_every)
        self.total_steps = int(total_steps)
        self.tokenizer = tokenizer
        self.num_sample_sentences = max(0, int(num_sample_sentences))
        self.metrics_atomic: dict[str, Any] = {
            "metadata": metadata,
            "per_step": [],
        }
        self._is_estimate_history: list[float] = []
        self._hit_rate_epsilons: tuple[float, ...] = (0.90, 0.95, 0.99)
        self._hit_rate_window: int = int(window)
        self._hit_rate_burn_in: int = int(burn_in)
        self._hit_indicators: dict[float, list[int]] = {eps: [] for eps in self._hit_rate_epsilons}
        self._cumulative_hits: dict[float, int] = {eps: 0 for eps in self._hit_rate_epsilons}
        self._cumulative_counted_steps: int = 0
        # Per-step |log10(is_estimate) - log10(gt_prob)|, capped at log_err_cap OOM.
        # Predictions <=0 or NaN saturate at the cap. Squared version derived on demand.
        self._log_err_cap: float = 10.0
        self._log_err_history: list[float] = []

        if self.output_pickle_path:
            abs_pickle_path = os.path.abspath(self.output_pickle_path)
            output_dir = os.path.dirname(abs_pickle_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            print(f"Atomic metric pickle path: {self.output_pickle_path}")

    def record_step(
        self,
        *,
        step: int,
        total_loss: torch.Tensor,
        main_loss: torch.Tensor,
        alpha_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        kl_coef: float,
        alpha_loss_per_pos: torch.Tensor,
        kl_loss_per_pos: torch.Tensor,
        rare_token_prob_per_pos: torch.Tensor | None = None,
        lambda_value: float,
        population_ess: float | None,
        rare_event_ess: float | None = None,
        sampled_qp_ess: float | None = None,
        dense_vocab_ess: float | None = None,
        dense_qp_ess: float | None = None,
        controller_ess: float | None = None,
        generated_tokens: torch.Tensor | None = None,
        rare_event: Any | None = None,
        event_indicator: torch.Tensor | None = None,
        non_cheat_count: int | None = None,
        log_importance_weights: torch.Tensor,
        log_p_seq: torch.Tensor | None = None,
        log_q_seq: torch.Tensor | None = None,
        log_p_per_pos: torch.Tensor | None = None,
        log_q_per_pos: torch.Tensor | None = None,
        worst_case_cum_log_weight: torch.Tensor | None = None,
        proposal_metrics: dict[str, Any] | None = None,
    ) -> None:
        if event_indicator is None:
            if generated_tokens is None or rare_event is None:
                raise ValueError(
                    "record_step requires either event_indicator or both generated_tokens and rare_event."
                )
            event_indicator = rare_event.compute_indicator(generated_tokens=generated_tokens)

        alpha_loss_value = float(alpha_loss.item())
        main_loss_value = float(main_loss.item())
        total_loss_value = float(total_loss.item())
        kl_loss_value = float(kl_loss.item())
        kl_coef_value = float(kl_coef)
        lambda_value = float(lambda_value)

        event_indicator_int = event_indicator.to(torch.int64)
        alpha_loss_times_lambda = lambda_value * alpha_loss_value
        # KL is now folded into the λ-scaled reg path with kl_coef as the
        # relative weight, so its loss-contribution is λ · kl_coef · kl_loss.
        kl_term_value = lambda_value * kl_coef_value * kl_loss_value

        # Share-of-magnitude split of the three additive contributions to the
        # training loss: |c| / Σ|c|. Robust to sign mixing (unlike c/total).
        share_denom = (
            abs(main_loss_value) + abs(alpha_loss_times_lambda) + abs(kl_term_value)
        )
        if share_denom > 0.0:
            main_share = abs(main_loss_value) / share_denom
            alpha_share = abs(alpha_loss_times_lambda) / share_denom
            kl_share = abs(kl_term_value) / share_denom
        else:
            main_share = alpha_share = kl_share = float("nan")

        # Reg-only share: how the two divergence terms split the combined
        # regularization force (ignoring the main rare-event loss).
        reg_share_denom = abs(alpha_loss_times_lambda) + abs(kl_term_value)
        if reg_share_denom > 0.0:
            alpha_reg_share = abs(alpha_loss_times_lambda) / reg_share_denom
            kl_reg_share = abs(kl_term_value) / reg_share_denom
        else:
            alpha_reg_share = kl_reg_share = float("nan")

        # Per-position breakdown across the K generated tokens: how each
        # position k contributes to the total reg, batch-averaged. Returned
        # as percentages summing to 100% per reg term so position contributions
        # are directly comparable across different lambda/coef scales.
        alpha_per_pos_mean = alpha_loss_per_pos.detach().float().mean(dim=0)  # [K]
        kl_per_pos_mean = kl_loss_per_pos.detach().float().mean(dim=0)  # [K]
        alpha_per_pos_list = [float(x) for x in alpha_per_pos_mean.tolist()]
        kl_per_pos_list = [float(x) for x in kl_per_pos_mean.tolist()]
        alpha_pos_denom = sum(abs(x) for x in alpha_per_pos_list)
        kl_pos_denom = sum(abs(x) for x in kl_per_pos_list)
        if alpha_pos_denom > 0.0:
            alpha_per_pos_pct = [100.0 * abs(x) / alpha_pos_denom for x in alpha_per_pos_list]
        else:
            alpha_per_pos_pct = [float("nan") for _ in alpha_per_pos_list]
        if kl_pos_denom > 0.0:
            kl_per_pos_pct = [100.0 * abs(x) / kl_pos_denom for x in kl_per_pos_list]
        else:
            kl_per_pos_pct = [float("nan") for _ in kl_per_pos_list]

        # Token-event diagnostic: Q(rare_token | history), batch-averaged per
        # position [K] and overall mean. None for non-token events.
        if rare_token_prob_per_pos is not None:
            rare_token_prob_per_pos_mean = (
                rare_token_prob_per_pos.detach().float().mean(dim=0)
            )  # [K]
            rare_token_prob_per_pos_list = [
                float(x) for x in rare_token_prob_per_pos_mean.tolist()
            ]
            rare_token_prob_mean = float(
                rare_token_prob_per_pos.detach().float().mean().item()
            )
            rare_token_pos_denom = sum(abs(x) for x in rare_token_prob_per_pos_list)
            if rare_token_pos_denom > 0.0:
                rare_token_prob_per_pos_pct = [
                    100.0 * abs(x) / rare_token_pos_denom
                    for x in rare_token_prob_per_pos_list
                ]
            else:
                rare_token_prob_per_pos_pct = [
                    float("nan") for _ in rare_token_prob_per_pos_list
                ]
        else:
            rare_token_prob_per_pos_list = None
            rare_token_prob_mean = None
            rare_token_prob_per_pos_pct = None

        log_is_estimate = None
        if self.metrics_atomic["metadata"].get("ordinary_is_unclipped", False):
            log_estimate = ordinary_is_log_estimate(log_importance_weights, event_indicator)
            log_is_estimate = float(log_estimate)
            is_estimate = float(log_estimate.exp())
        else:
            # Preserve the historical estimator for existing methods.
            importance_weights = log_importance_weights.clamp(min=-700.0, max=700.0).exp()
            is_estimate = float((importance_weights * event_indicator.to(torch.float64)).mean())
        estimate_metrics = self.record_estimate(step, is_estimate, log_is_estimate)
        if proposal_metrics is not None:
            proposal_metrics = self.prepare_ce_metrics(proposal_metrics, event_indicator)

        self.metrics_atomic["per_step"].append(
            {
                "step": int(step),
                "main_loss": main_loss_value,
                "alpha_loss": alpha_loss_value,
                "kl_loss": kl_loss_value,
                "kl_coef": kl_coef_value,
                "kl_term": kl_term_value,
                "main_share": main_share,
                "alpha_share": alpha_share,
                "kl_share": kl_share,
                "alpha_reg_share": alpha_reg_share,
                "kl_reg_share": kl_reg_share,
                "alpha_per_pos": alpha_per_pos_list,
                "kl_per_pos": kl_per_pos_list,
                "alpha_per_pos_pct": alpha_per_pos_pct,
                "kl_per_pos_pct": kl_per_pos_pct,
                "rare_token_prob_per_pos": rare_token_prob_per_pos_list,
                "rare_token_prob_per_pos_pct": rare_token_prob_per_pos_pct,
                "rare_token_prob_mean": rare_token_prob_mean,
                "lambda": lambda_value,
                "sampled_qp_ess": sampled_qp_ess,
                "dense_vocab_ess": dense_vocab_ess,
                "dense_qp_ess": dense_qp_ess,
                "controller_ess": controller_ess,
                "log_importance_weights": [float(x) for x in log_importance_weights.tolist()],
                "log_p_seq": (
                    [float(x) for x in log_p_seq.tolist()] if log_p_seq is not None else None
                ),
                "log_q_seq": (
                    [float(x) for x in log_q_seq.tolist()] if log_q_seq is not None else None
                ),
                "event_indicator": [int(x) for x in event_indicator_int.tolist()],
                "worst_case_cum_log_weight": (
                    [float(x) for x in worst_case_cum_log_weight.tolist()]
                    if worst_case_cum_log_weight is not None
                    else None
                ),
                "samples": _build_samples_list(
                    log_importance_weights=log_importance_weights,
                    event_indicator_int=event_indicator_int,
                    log_p_seq=log_p_seq,
                    log_q_seq=log_q_seq,
                    log_p_per_pos=log_p_per_pos,
                    log_q_per_pos=log_q_per_pos,
                ),
                "non_cheat_count": non_cheat_count,
                "is_estimate": is_estimate,
                "log_is_estimate": log_is_estimate,
                "proposal_metrics": dict(proposal_metrics) if proposal_metrics is not None else None,
                **estimate_metrics,
            }
        )

        if self._should_log(step):
            self.flush()
            ess_str = "None" if population_ess is None else f"{population_ess:.4f}"
            rare_ess_str = "None" if rare_event_ess is None else f"{rare_event_ess:.4f}"
            sampled_qp_ess_str = (
                "None" if sampled_qp_ess is None else f"{sampled_qp_ess:.4f}"
            )
            dense_ess_str = (
                "None" if dense_vocab_ess is None else f"{dense_vocab_ess:.4f}"
            )
            dense_qp_ess_str = (
                "None" if dense_qp_ess is None else f"{dense_qp_ess:.4f}"
            )
            controller_ess_str = (
                "None" if controller_ess is None else f"{controller_ess:.4f}"
            )
            occurrence_in_batch_percentage = event_indicator.float().mean().item() * 100.0
            gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
            alpha_reg_moment = self.metrics_atomic["metadata"].get("alpha_reg_moment")
            alpha_reg_moment_str = (
                "None" if alpha_reg_moment is None else f"{float(alpha_reg_moment):.6e}"
            )

            phase = str(self.metrics_atomic.get("metadata", {}).get("phase", "train")).upper()
            print("")
            print(
                f"==================== ITERATION [{phase}] {step:04d}/{self.total_steps:04d} ===================="
            )
            ess_target = self.metrics_atomic["metadata"].get("ess_target")
            ess_target_str = (
                "None" if ess_target is None else f"{float(ess_target):.4f}"
            )
            print(
                f"total={total_loss_value:.6e} "
                f"alpha_reg_moment={alpha_reg_moment_str} "
                f"ess_target={ess_target_str}"
            )
            print(f"main={main_loss_value:.6e}")
            print(f"alpha_lambda={alpha_loss_times_lambda:.6e}")
            print(f"alpha={alpha_loss_value:.6e}")
            print(f"lambda={lambda_value:.6e}")
            print(f"kl_qp={kl_loss_value:.6e}  kl_coef={kl_coef_value:.6e}  kl_term={kl_term_value:.6e}")
            main_share_str = "nan" if main_share != main_share else f"{main_share * 100.0:.2f}%"
            alpha_share_str = "nan" if alpha_share != alpha_share else f"{alpha_share * 100.0:.2f}%"
            kl_share_str = "nan" if kl_share != kl_share else f"{kl_share * 100.0:.2f}%"
            alpha_reg_share_str = (
                "nan" if alpha_reg_share != alpha_reg_share
                else f"{alpha_reg_share * 100.0:.2f}%"
            )
            kl_reg_share_str = (
                "nan" if kl_reg_share != kl_reg_share
                else f"{kl_reg_share * 100.0:.2f}%"
            )
            print(
                f"loss_share (|c|/Σ|c|): main={main_share_str}  "
                f"alpha_lambda={alpha_share_str}  kl_term={kl_share_str}"
            )
            print(
                f"reg_share  (|c|/Σ|reg|): alpha_lambda={alpha_reg_share_str}  "
                f"kl_term={kl_reg_share_str}"
            )

            def _fmt_pos_summary(values: list[float], pcts: list[float]) -> str:
                if not values:
                    return "(empty)"
                abs_vals = [abs(v) for v in values]
                max_k = max(range(len(values)), key=lambda k: abs_vals[k])
                min_k = min(range(len(values)), key=lambda k: abs_vals[k])

                def _p(p: float) -> str:
                    return "nan" if p != p else f"{p:.1f}%"

                return (
                    f"max@k={max_k}:{values[max_k]:.3e}({_p(pcts[max_k])})  "
                    f"min@k={min_k}:{values[min_k]:.3e}({_p(pcts[min_k])})"
                )

            print(
                "alpha_per_pos (share-of-Σ|alpha|): "
                f"{_fmt_pos_summary(alpha_per_pos_list, alpha_per_pos_pct)}"
            )
            print(
                "kl_per_pos    (share-of-Σ|kl|):    "
                f"{_fmt_pos_summary(kl_per_pos_list, kl_per_pos_pct)}"
            )
            if rare_token_prob_per_pos_list is not None:
                print(
                    f"rare_token_prob (share-of-Σ|prob|, mean={rare_token_prob_mean:.3e}): "
                    f"{_fmt_pos_summary(rare_token_prob_per_pos_list, rare_token_prob_per_pos_pct)}"
                )
            controller_source = self.metrics_atomic["metadata"].get(
                "ess_controller_source", "sampled_switch"
            )
            print(
                f"ess={ess_str}  rare_event_ess={rare_ess_str}  "
                f"dense_vocab_ess={dense_ess_str}"
            )
            print(
                f"sampled_qp_ess={sampled_qp_ess_str}  "
                f"dense_qp_ess={dense_qp_ess_str}"
            )
            print(
                f"controller_ess={controller_ess_str}  "
                f"controller_source={controller_source}"
            )

            ww = self._worst_weight_stats(
                self.metrics_atomic["per_step"][-1], gt_prob=gt_prob
            )

            def _fmt_lw(v: float | None) -> str:
                return "None" if v is None else f"{v:.3f}"

            print("worst_log_weight spread (nat-log P/Q, range=max-min):")
            print(
                f"  population  min={_fmt_lw(ww['pop_min'])} mean={_fmt_lw(ww['pop_mean'])} "
                f"max={_fmt_lw(ww['pop_max'])} range={_fmt_lw(ww['pop_range'])}"
            )
            print(
                f"  in-Φ        min={_fmt_lw(ww['in_phi_min'])} mean={_fmt_lw(ww['in_phi_mean'])} "
                f"max={_fmt_lw(ww['in_phi_max'])} range={_fmt_lw(ww['in_phi_range'])}"
            )
            if ww["in_phi_dev_max"] is not None:
                # δ = log w - log(gt); ideal in-Φ weight is δ=0. δmax>0 ⇒ worst
                # over-weight; δmin<0 ⇒ worst under-weight; both<0 ⇒ under-coverage.
                print(
                    "  in-Φ vs GT (δ=log w-log gt; ideal 0): "
                    f"δmax={_fmt_lw(ww['in_phi_dev_max'])} "
                    f"δmin={_fmt_lw(ww['in_phi_dev_min'])} "
                    f"mean|δ|={_fmt_lw(ww['in_phi_dev_absmean'])}"
                )
            print(f"event_occur_pct={occurrence_in_batch_percentage:.2f}%")
            total_hits = int(event_indicator_int.sum().item())
            if non_cheat_count is None:
                cheat_str = "n/a (event has no token_id)"
            else:
                cheat_at_pos0 = total_hits - non_cheat_count
                cheat_str = (
                    f"cheat@pos0={cheat_at_pos0}/{total_hits}  "
                    f"non_cheat={non_cheat_count}/{total_hits}"
                )
            print(cheat_str)

            self.print_estimate_summary(is_estimate)
            print("-" * 40)
            self._print_sample_sentences(
                generated_tokens=generated_tokens,
                event_indicator=event_indicator,
            )

    def flush(self) -> None:
        """Persist the in-memory metrics dict to the pickle path, if configured."""
        if self.output_pickle_path:
            with open(self.output_pickle_path, "wb") as handle:
                pickle.dump(self.metrics_atomic, handle)

    def print_estimate_summary(self, is_estimate):
        """The original probability/accuracy block, shared by both runners."""
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        gt_prob_str = "None" if gt_prob is None else f"{float(gt_prob):.4e}"
        recent = self._is_estimate_history[-50:]
        average = sum(recent) / len(recent) if recent else is_estimate
        print("-" * 40)
        print(f"is_estimate={is_estimate:.4e}")
        print(f"is_estimate_avg50={average:.4e}")
        print(f"gt_prob={gt_prob_str}")
        print("-" * 40)
        self._print_hit_rates(gt_prob=gt_prob)
        self._print_log_error_stats()

    def _should_log(self, step: int) -> bool:
        if step == 1 or step == self.total_steps:
            return True
        return step % self.log_every == 0

    @staticmethod
    def _normalized_ess_from_log_weights(log_weights: list[float]) -> float | None:
        num_samples = len(log_weights)
        if num_samples < 2:
            return None
        log_w = torch.tensor(log_weights, dtype=torch.float64)
        max_log_w = log_w.max()
        stable_w = torch.exp(log_w - max_log_w)
        sum_w = stable_w.sum()
        sum_w_sq = (stable_w**2).sum()
        ess = (sum_w**2) / (sum_w_sq + 1e-12)
        return float((ess / num_samples).item())

    def _rare_event_normalized_ess(
        self,
        log_weights: list[float],
        event_indicator: list[int],
    ) -> float | None:
        selected_log_weights = [lw for lw, hit in zip(log_weights, event_indicator) if int(hit) == 1]
        return self._normalized_ess_from_log_weights(selected_log_weights)

    @staticmethod
    def _is_estimator(log_weights: list[float], event_indicator: list[int]) -> float | None:
        num_samples = len(log_weights)
        if num_samples == 0:
            return None
        if len(event_indicator) != num_samples:
            return None
        log_w = torch.tensor(log_weights, dtype=torch.float64)
        indicator = torch.tensor([int(x) for x in event_indicator], dtype=torch.float64)
        clipped_log_w = log_w.clamp(min=-700.0, max=700.0)
        w = torch.exp(clipped_log_w)
        return float(((w * indicator).sum() / num_samples).item())

    @staticmethod
    def _worst_weight_stats(
        row: dict[str, Any], gt_prob: float | None = None
    ) -> dict[str, float | None]:
        """Per-step worst-weight summary, all in natural-log weight space.

        IS variance is driven by the dispersion of the weights, not their level
        (the optimal proposal makes w·1_Φ constant over Φ → zero range). For each
        set we report min / mean / max and range = max - min:

        - population: spread over B of the per-sequence worst-case cumulative log
          weight Σ_k max_v(log P - log Q) (the upper-bound weights).
        - in-Φ: spread over event-hitters of the realized log P/Q (the weights
          that actually enter the estimator).

        For in-Φ we additionally anchor at GT: the zero-variance optimal proposal
        makes every in-Φ weight equal gt_prob, so δ = log w - log(gt) is each
        weight's signed distance from the ideal. δ_max>0 is the worst over-weight
        (variance blow-up), δ_min<0 the worst under-weight (collapse); this also
        captures the hit-rate offset that the anchor-free range misses, since the
        in-Φ mean sits at log(gt/hit_rate).
        """
        def _stats(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"min": None, "mean": None, "max": None, "range": None}
            vmin, vmax = min(values), max(values)
            return {
                "min": vmin,
                "mean": sum(values) / len(values),
                "max": vmax,
                "range": vmax - vmin,
            }

        ub = row.get("worst_case_cum_log_weight")
        pop_vals = [float(x) for x in ub] if ub else []
        log_iw = row.get("log_importance_weights") or []
        event = row.get("event_indicator") or []
        in_phi_vals = [float(lw) for lw, hit in zip(log_iw, event) if int(hit) == 1]

        pop = _stats(pop_vals)
        in_phi = _stats(in_phi_vals)

        # GT-anchored in-Φ deviations δ = log w - log(gt). None when gt unknown.
        gt_log = (
            math.log(float(gt_prob))
            if gt_prob is not None and float(gt_prob) > 0.0
            else None
        )
        if gt_log is not None and in_phi_vals:
            deltas = [v - gt_log for v in in_phi_vals]
            dev_min = min(deltas)
            dev_max = max(deltas)
            dev_absmean = sum(abs(d) for d in deltas) / len(deltas)
        else:
            dev_min = dev_max = dev_absmean = None

        return {
            "pop_min": pop["min"], "pop_mean": pop["mean"],
            "pop_max": pop["max"], "pop_range": pop["range"],
            "in_phi_min": in_phi["min"], "in_phi_mean": in_phi["mean"],
            "in_phi_max": in_phi["max"], "in_phi_range": in_phi["range"],
            "in_phi_dev_min": dev_min, "in_phi_dev_max": dev_max,
            "in_phi_dev_absmean": dev_absmean,
        }

    @staticmethod
    def _single_line_literal(text: str) -> str:
        parts: list[str] = []
        for ch in text:
            if ch == "\\":
                parts.append("\\\\")
            elif ch == "\n":
                parts.append("\\n")
            elif ch == "\r":
                parts.append("\\r")
            elif ch == "\t":
                parts.append("\\t")
            elif ch.isprintable():
                parts.append(ch)
            else:
                parts.append(f"\\x{ord(ch):02x}")
        return "".join(parts)

    def record_estimate(self, step: int, is_estimate: float,
                        log_is_estimate: float | None = None) -> dict[str, float | None]:
        """Shared accuracy summaries for the full trainer and chunked runners.

        Accuracy hits retain the historical relative-error intervals, burn-in,
        and rolling window. They are distinct from actual event frequency.
        """
        self._is_estimate_history.append(is_estimate)
        self._accumulate_hit_indicators(step, is_estimate)
        self._accumulate_log_error(step, is_estimate)
        result = {}
        for eps, rates in self._compute_window_hit_rates().items():
            result[f"hit_rate_cum_{eps:.2f}"] = rates["cumulative"]
            result[f"hit_rate_win{self._hit_rate_window}_{eps:.2f}"] = rates["window"]
        stats = self._compute_log_error_stats()
        for metric in ("mae", "mse"):
            result[f"log_{metric}_cap{int(self._log_err_cap)}_cum"] = stats[f"cum_{metric}"]
            result[f"log_{metric}_cap{int(self._log_err_cap)}_win{self._hit_rate_window}"] = stats[f"win_{metric}"]
        gt = self.metrics_atomic['metadata'].get('gt_prob')
        log_error = None
        if gt is not None and float(gt) > 0:
            log_est = log_is_estimate
            if log_est is None:
                log_est = math.log(is_estimate) if is_estimate > 0 else -math.inf
            log_error = abs(log_est-math.log(float(gt)))/math.log(10)
        result['log_error'] = log_error
        result['log_error_cap10'] = min(log_error, self._log_err_cap) if log_error is not None else None
        result['is_estimate_avg50'] = sum(self._is_estimate_history[-50:])/len(self._is_estimate_history[-50:])
        return result

    def _accumulate_hit_indicators(self, step: int, is_estimate: float) -> None:
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        if gt_prob is None or float(gt_prob) <= 0.0:
            return
        gt_val = float(gt_prob)
        past_burn_in = step > self._hit_rate_burn_in

        for eps in self._hit_rate_epsilons:
            lower = gt_val * (1.0 - eps)
            upper = gt_val * (1.0 + eps)
            is_hit = bool(
                is_estimate == is_estimate  # not NaN
                and lower <= is_estimate <= upper
            )
            if past_burn_in:
                self._hit_indicators[eps].append(1 if is_hit else 0)
                if is_hit:
                    self._cumulative_hits[eps] += 1

        if past_burn_in:
            self._cumulative_counted_steps += 1

    def _accumulate_log_error(self, step: int, is_estimate: float) -> None:
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        if gt_prob is None or float(gt_prob) <= 0.0:
            return
        if step <= self._hit_rate_burn_in:
            return
        gt_val = float(gt_prob)
        # Predictions <=0 or NaN are treated as fully saturated.
        if is_estimate != is_estimate or is_estimate <= 0.0:
            err = self._log_err_cap
        else:
            err = abs(math.log10(is_estimate) - math.log10(gt_val))
            err = min(err, self._log_err_cap)
        self._log_err_history.append(err)

    def _compute_log_error_stats(self) -> dict[str, float | None]:
        if not self._log_err_history:
            return {"cum_mae": None, "cum_mse": None, "win_mae": None, "win_mse": None}
        n = len(self._log_err_history)
        cum_mae = sum(self._log_err_history) / n
        cum_mse = sum(e * e for e in self._log_err_history) / n
        # Window is a max — use up to the last `_hit_rate_window` samples. As soon
        # as there's any post-burn-in data, return a real number rather than None.
        recent = self._log_err_history[-self._hit_rate_window:]
        win_n = len(recent)
        win_mae = sum(recent) / win_n
        win_mse = sum(e * e for e in recent) / win_n
        return {"cum_mae": cum_mae, "cum_mse": cum_mse, "win_mae": win_mae, "win_mse": win_mse}

    def _print_log_error_stats(self) -> None:
        stats = self._compute_log_error_stats()
        win = self._hit_rate_window
        cap = self._log_err_cap

        def fmt(v: float | None) -> str:
            return "None" if v is None else f"{v:.4f}"

        print(
            f"log_error (cap={cap:g} OOM, window={win}, after burn_in={self._hit_rate_burn_in}):"
        )
        print(f"  log_mae  win_{win}={fmt(stats['win_mae'])}")
        print(f"  log_mse  win_{win}={fmt(stats['win_mse'])}")

    def _print_hit_rates(self, gt_prob: float | None) -> None:
        window_rates = self._compute_window_hit_rates()
        win = self._hit_rate_window
        print(f"hit_rates (window={win}, after burn_in={self._hit_rate_burn_in}):")
        for eps in self._hit_rate_epsilons:
            rates = window_rates.get(eps, {"window": None})
            win_str = "None" if rates["window"] is None else f"{rates['window']:.4f}"
            print(f"  eps={eps:.2f}  win_{win}={win_str}")

    def _compute_hit_rates(self, gt_prob: float | None) -> dict[float, float | None]:
        hit_rates: dict[float, float | None] = {eps: None for eps in self._hit_rate_epsilons}
        if gt_prob is None:
            return hit_rates
        gt_prob_value = float(gt_prob)
        if gt_prob_value < 0.0:
            return hit_rates

        estimate_count = len(self._is_estimate_history)
        if estimate_count == 0:
            return hit_rates

        for eps in self._hit_rate_epsilons:
            lower = gt_prob_value - (eps * gt_prob_value)
            upper = gt_prob_value + (eps * gt_prob_value)
            hits = sum(1 for estimate in self._is_estimate_history if lower <= estimate <= upper)
            hit_rates[eps] = hits / float(estimate_count)
        return hit_rates

    def _compute_window_hit_rates(self) -> dict[float, dict[str, float | None]]:
        result: dict[float, dict[str, float | None]] = {}
        for eps in self._hit_rate_epsilons:
            indicators = self._hit_indicators[eps]
            cum_rate = None
            window_rate = None
            if self._cumulative_counted_steps > 0:
                cum_rate = self._cumulative_hits[eps] / float(self._cumulative_counted_steps)
            # Window is a max — use up to the last `_hit_rate_window` samples so
            # a value is reported as soon as there is post-burn-in data.
            if indicators:
                recent = indicators[-self._hit_rate_window:]
                window_rate = sum(recent) / float(len(recent))
            result[eps] = {"cumulative": cum_rate, "window": window_rate}
        return result

    def save_hit_rates_summary(self, file_path: str) -> None:
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        hit_rates = self._compute_hit_rates(gt_prob=gt_prob)
        window_rates = self._compute_window_hit_rates()
        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write("# Average hit rates\n")
            handle.write(f"estimate_count: {len(self._is_estimate_history)}\n")
            handle.write(f"burn_in_steps: {self._hit_rate_burn_in}\n")
            handle.write(f"window_size: {self._hit_rate_window}\n")
            handle.write(f"counted_steps_after_burn_in: {self._cumulative_counted_steps}\n")
            if gt_prob is None:
                handle.write("gt_prob: None\n")
            else:
                handle.write(f"gt_prob: {float(gt_prob):.12e}\n")
            handle.write("\n# Cumulative hit rates (all steps)\n")
            for eps in self._hit_rate_epsilons:
                hit_rate = hit_rates[eps]
                value_str = "None" if hit_rate is None else f"{hit_rate:.12f}"
                handle.write(f"hit_rate_eps_{eps:.2f}: {value_str}\n")
            handle.write(f"\n# Post-burn-in cumulative and window-{self._hit_rate_window} hit rates\n")
            for eps in self._hit_rate_epsilons:
                rates = window_rates.get(eps, {"cumulative": None, "window": None})
                cum_str = "None" if rates["cumulative"] is None else f"{rates['cumulative']:.12f}"
                win_str = "None" if rates["window"] is None else f"{rates['window']:.12f}"
                handle.write(
                    f"hit_rate_post_burn_in_{eps:.2f}: cum={cum_str} "
                    f"({self._cumulative_hits[eps]}/{self._cumulative_counted_steps})  "
                    f"window_{self._hit_rate_window}={win_str}\n"
                )

    def _aggregated_log_mae(
        self,
        is_estimates: list[float],
        gt_prob: float,
        W: int,
    ) -> tuple[float, int] | None:
        """Group is_estimates into non-overlapping chunks of W, average each chunk
        (= one IS estimate from W·batch_size samples), then return capped log MAE
        and the number of chunks. Returns None if no full chunk fits."""
        n = (len(is_estimates) // W) * W
        if n == 0:
            return None
        cap = self._log_err_cap
        log_p = math.log10(gt_prob)
        arr = np.array(is_estimates[:n], dtype=np.float64).reshape(-1, W).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            err = np.where(
                arr > 0,
                np.abs(np.log10(arr) - log_p),
                cap,
            )
        return float(np.minimum(err, cap).mean()), int(arr.shape[0])

    def save_log_error_summary(self, file_path: str) -> None:
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        stats = self._compute_log_error_stats()
        cap = self._log_err_cap
        cap_tag = int(cap)
        win = self._hit_rate_window

        def _fmt(v: float | None) -> str:
            return "None" if v is None else f"{v:.12f}"

        with open(file_path, "w", encoding="utf-8") as handle:
            handle.write(
                f"# Log-scale error vs gt_prob: |log10 p_hat - log10 p_true|, "
                f"capped at {cap:g} OOM per step (post burn-in).\n"
                f"# Predictions <=0 or NaN saturate at the cap. "
                f"log_mae has range [0, {cap:g}]; log_mse has range [0, {cap*cap:g}].\n"
            )
            handle.write(f"estimate_count: {len(self._is_estimate_history)}\n")
            handle.write(f"burn_in_steps: {self._hit_rate_burn_in}\n")
            handle.write(f"window_size: {win}\n")
            handle.write(f"log_err_cap_oom: {cap:g}\n")
            handle.write(f"log_err_count_post_burn_in: {len(self._log_err_history)}\n")
            if gt_prob is None:
                handle.write("gt_prob: None\n")
            else:
                handle.write(f"gt_prob: {float(gt_prob):.12e}\n")

            handle.write("\n# Cumulative (all post-burn-in steps)\n")
            handle.write(f"log_mae_cap{cap_tag}_cumulative: {_fmt(stats['cum_mae'])}\n")
            handle.write(f"log_mse_cap{cap_tag}_cumulative: {_fmt(stats['cum_mse'])}\n")

            handle.write(f"\n# Window-{win} (last {win} post-burn-in steps)\n")
            handle.write(f"log_mae_cap{cap_tag}_window_{win}: {_fmt(stats['win_mae'])}\n")
            handle.write(f"log_mse_cap{cap_tag}_window_{win}: {_fmt(stats['win_mse'])}\n")

            # Efficiency: smallest naive-MC N matching this log_mae under the same
            # capped metric (k ~ Poisson(N·p); k=0 saturates at cap, so the cap
            # subsidizes MC vs IS — efficiency_vs_batch is a conservative LOWER bound).
            batch_size = self.metrics_atomic["metadata"].get("batch_size")
            if gt_prob is not None and float(gt_prob) > 0.0:
                p = float(gt_prob)
                n_cum = _naive_mc_n_for_log_mae(p, stats["cum_mae"], cap=cap)
                n_win = _naive_mc_n_for_log_mae(p, stats["win_mae"], cap=cap)

                def _fmt_n(v: int | None) -> str:
                    return "None" if v is None else f"{v:d}"

                def _fmt_ratio(n: int | None) -> str:
                    if n is None or not batch_size:
                        return "None"
                    return f"{n / float(batch_size):.6e}"

                handle.write(
                    f"\n# Naive-MC equivalence (Poisson(N·p) sim, same cap={cap:g} OOM, "
                    f"num_trials=20000, seed=0).\n"
                    f"# N is the smallest sample count s.t. simulated capped log_mae "
                    f"≤ the IS log_mae above.\n"
                    f"# efficiency_vs_batch = N / batch_size — how many naive-MC "
                    f"samples one IS estimate (batch={batch_size}) is worth.\n"
                )
                handle.write(f"naive_mc_n_for_cum_mae: {_fmt_n(n_cum)}\n")
                handle.write(f"naive_mc_n_for_win_mae: {_fmt_n(n_win)}\n")
                handle.write(f"efficiency_vs_batch_cum: {_fmt_ratio(n_cum)}\n")
                handle.write(f"efficiency_vs_batch_win: {_fmt_ratio(n_win)}\n")

                # Effective-batch sweep: group W consecutive post-burn-in IS estimates
                # into one estimate from W·batch_size samples. Q is frozen in eval so
                # this is exactly equivalent to running with a larger batch_size.
                phase = str(
                    self.metrics_atomic["metadata"].get("phase", "")
                ).lower()
                if phase == "eval" and batch_size:
                    post_burn_in_estimates = (
                        self._is_estimate_history[-len(self._log_err_history):]
                        if self._log_err_history else []
                    )
                    if post_burn_in_estimates:
                        handle.write(
                            "\n# Effective-batch sweep (eval only): group W consecutive\n"
                            "# post-burn-in IS estimates → one IS estimate from W·batch_size\n"
                            "# samples. Each row gives capped log_mae across the resulting\n"
                            "# chunks, and the matching naive-MC N + efficiency ratio.\n"
                        )
                        handle.write(
                            f"{'W':>5} {'eff_batch':>10} {'groups':>7} "
                            f"{'log_mae':>9} {'naive_mc_N':>14} {'eff_vs_mc':>12}\n"
                        )
                        for W in (1, 2, 4, 8, 16, 32, 64, 128, 256):
                            result = self._aggregated_log_mae(
                                post_burn_in_estimates, p, W
                            )
                            if result is None:
                                break
                            mae_W, n_groups = result
                            n_mc_W = _naive_mc_n_for_log_mae(p, mae_W, cap=cap)
                            eff_batch_W = int(batch_size) * W
                            n_mc_str = "None" if n_mc_W is None else f"{n_mc_W:d}"
                            eff_str = (
                                "None" if n_mc_W is None
                                else f"{n_mc_W / float(eff_batch_W):.3e}"
                            )
                            handle.write(
                                f"{W:>5d} {eff_batch_W:>10d} {n_groups:>7d} "
                                f"{mae_W:>9.4f} {n_mc_str:>14} {eff_str:>12}\n"
                            )

    def _print_sample_sentences(
        self,
        generated_tokens: torch.Tensor | None,
        event_indicator: torch.Tensor | None = None,
    ) -> None:
        if (
            self.tokenizer is None
            or generated_tokens is None
            or self.num_sample_sentences <= 0
            or generated_tokens.ndim != 2
            or generated_tokens.shape[0] == 0
        ):
            return
        prefix = str(self.metrics_atomic.get("metadata", {}).get("context", ""))
        max_per_group = self.num_sample_sentences

        # Without an event indicator, fall back to a single ungrouped block.
        if event_indicator is None or event_indicator.shape[0] != generated_tokens.shape[0]:
            sample_count = min(max_per_group, int(generated_tokens.shape[0]))
            token_rows = generated_tokens[:sample_count].detach().cpu().tolist()
            decoded_rows = self.tokenizer.batch_decode(token_rows, skip_special_tokens=False)
            print("sample_sentences:")
            for idx, sample in enumerate(decoded_rows, start=1):
                escaped = self._single_line_literal(prefix + sample)
                print(f"sample_{idx}={escaped}")
            return

        ind = event_indicator.detach().to(torch.int64).cpu().tolist()
        hit_idx = [i for i, x in enumerate(ind) if int(x) == 1][:max_per_group]
        miss_idx = [i for i, x in enumerate(ind) if int(x) == 0][:max_per_group]

        def _print_group(label: str, indices: list[int]) -> None:
            if not indices:
                print(f"sample_sentences[{label}]: (none in batch)")
                return
            token_rows = generated_tokens[indices].detach().cpu().tolist()
            decoded_rows = self.tokenizer.batch_decode(token_rows, skip_special_tokens=False)
            print(f"sample_sentences[{label}] ({len(indices)} of up to {max_per_group}):")
            for idx, (batch_i, sample) in enumerate(zip(indices, decoded_rows), start=1):
                escaped = self._single_line_literal(prefix + sample)
                print(f"  {label}_{idx} (batch_idx={batch_i})={escaped}")

        _print_group("hit", hit_idx)
        _print_group("miss", miss_idx)
