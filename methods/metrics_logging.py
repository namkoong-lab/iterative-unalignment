import math
import os
import pickle
from typing import Any, Callable

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
    """Approximate the MC sample count matching a capped log MAE by simulation.

    Uses Poisson(N*p), or a normal approximation when N*p >= 1e7.
    Returns None for missing targets, nonpositive p, or an unmet target at n_ceiling.
    """
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
    """Log losses and estimates; save sample weights and metric summaries."""

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
        # Cap absolute log10 errors for the saved summaries.
        self._log_err_cap: float = 10.0
        self._log_err_history: list[float] = []

        if self.output_pickle_path:
            abs_pickle_path = os.path.abspath(self.output_pickle_path)
            output_dir = os.path.dirname(abs_pickle_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

    def record_step(
        self,
        *,
        step: int,
        total_loss: torch.Tensor,
        main_loss: torch.Tensor,
        kl_loss: torch.Tensor,
        kl_loss_per_pos: torch.Tensor,
        lambda_value: float,
        population_ess: float | None,
        rare_event_ess: float | None,
        controller_ess: float | None,
        generated_tokens: torch.Tensor,
        event_indicator: torch.Tensor,
        log_importance_weights: torch.Tensor,
        log_p_seq: torch.Tensor | None = None,
        log_q_seq: torch.Tensor | None = None,
        log_p_per_pos: torch.Tensor | None = None,
        log_q_per_pos: torch.Tensor | None = None,
        on_estimate_ready: Callable[[], None] | None = None,
        proposal_metrics: dict | None = None,
    ) -> None:
        log_estimate = ordinary_is_log_estimate(log_importance_weights, event_indicator)
        log_is_estimate = float(log_estimate)
        is_estimate = float(log_estimate.exp())
        # Algorithm timing ends after the estimate, before diagnostics and I/O.
        if on_estimate_ready is not None:
            on_estimate_ready()

        estimate_metrics = self.record_estimate(step, is_estimate, log_is_estimate)
        proposal_metrics = (
            self.prepare_ce_metrics(proposal_metrics, event_indicator)
            if proposal_metrics is not None else None
        )
        main_loss_value = float(main_loss.detach())
        total_loss_value = float(total_loss.detach())
        kl_loss_value = float(kl_loss.detach())
        lambda_value = float(lambda_value)
        kl_term_value = lambda_value * kl_loss_value
        event_indicator_int = event_indicator.to(torch.int64)
        self.metrics_atomic["per_step"].append({
            "step": int(step),
            "proposal_metrics": proposal_metrics,
            "total_loss": total_loss_value,
            "main_loss": main_loss_value,
            "kl_loss": kl_loss_value,
            "kl_term": kl_term_value,
            "kl_per_pos": kl_loss_per_pos.detach().float().mean(dim=0).tolist(),
            "lambda": lambda_value,
            "population_ess": population_ess,
            "rare_event_ess": rare_event_ess,
            "controller_ess": controller_ess,
            "log_importance_weights": log_importance_weights.detach().tolist(),
            "log_p_seq": log_p_seq.detach().tolist() if log_p_seq is not None else None,
            "log_q_seq": log_q_seq.detach().tolist() if log_q_seq is not None else None,
            "event_indicator": event_indicator_int.tolist(),
            "samples": _build_samples_list(
                log_importance_weights=log_importance_weights,
                event_indicator_int=event_indicator_int,
                log_p_seq=log_p_seq,
                log_q_seq=log_q_seq,
                log_p_per_pos=log_p_per_pos,
                log_q_per_pos=log_q_per_pos,
            ),
            "is_estimate": is_estimate,
            "log_is_estimate": log_is_estimate,
            **estimate_metrics,
        })
        if self._should_log(step):
            self.flush()
            phase = self.metrics_atomic["metadata"].get("phase", "train")
            print(
                f"[{phase} {step}/{self.total_steps}] "
                f"loss={total_loss_value:.6e} main={main_loss_value:.6e} "
                f"kl_qp={kl_loss_value:.6e} kl_term={kl_term_value:.6e} "
                f"lambda={lambda_value:.6e}"
            )
            self.print_estimate_summary(is_estimate)
            self._print_sample_sentences(generated_tokens, event_indicator)
            self._print_ce_fit(proposal_metrics)

    def flush(self) -> None:
        """Persist the in-memory metrics dict to the pickle path, if configured."""
        if self.output_pickle_path:
            temporary_path = self.output_pickle_path + ".tmp"
            with open(temporary_path, "wb") as handle:
                pickle.dump(self.metrics_atomic, handle)
            os.replace(temporary_path, self.output_pickle_path)

    def print_estimate_summary(self, is_estimate: float) -> None:
        gt_prob = self.metrics_atomic["metadata"].get("gt_prob")
        gt_prob_str = "None" if gt_prob is None else f"{float(gt_prob):.4e}"
        recent = self._is_estimate_history[-50:]
        average = sum(recent) / len(recent) if recent else is_estimate
        print(
            f"is_estimate={is_estimate:.4e} "
            f"is_estimate_avg50={average:.4e} gt_prob={gt_prob_str}"
        )

    def _should_log(self, step: int) -> bool:
        if step == 1 or step == self.total_steps:
            return True
        return step % self.log_every == 0

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
        """Record estimation errors and relative-error interval coverage."""
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
        # Use the available post-burn-in estimates, up to the window size.
        recent = self._log_err_history[-self._hit_rate_window:]
        win_n = len(recent)
        win_mae = sum(recent) / win_n
        win_mse = sum(e * e for e in recent) / win_n
        return {"cum_mae": cum_mae, "cum_mse": cum_mse, "win_mae": win_mae, "win_mse": win_mse}

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
            # Allow partial windows after burn-in.
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

            # Estimate an MC sample count under the same capped error metric.
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
                    f"\n# Approximate MC sample count matching capped log MAE (cap={cap:g}).\n"
                    "# Poisson simulation; normal approximation for N*p >= 1e7.\n"
                    "# num_trials=20000, seed=0; efficiency_vs_batch = N / batch_size.\n"
                )
                handle.write(f"naive_mc_n_for_cum_mae: {_fmt_n(n_cum)}\n")
                handle.write(f"naive_mc_n_for_win_mae: {_fmt_n(n_win)}\n")
                handle.write(f"efficiency_vs_batch_cum: {_fmt_ratio(n_cum)}\n")
                handle.write(f"efficiency_vs_batch_win: {_fmt_ratio(n_win)}\n")

                # Average groups of W frozen-proposal batches, each of equal size.
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
                            "\n# Eval estimates grouped into batches of W * batch_size samples.\n"
                            "# MC sample counts and ratios are simulation-based approximations.\n"
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

    def prepare_ce_metrics(self, metrics, event_indicator):
        """Decode the actual, frozen fitting elites for both training runners."""
        result = dict(metrics)
        samples = [dict(sample) for sample in metrics.get("elite_samples", [])]
        if samples and self.tokenizer is not None and hasattr(self.tokenizer, "batch_decode"):
            texts = self.tokenizer.batch_decode(
                [sample["token_ids"] for sample in samples], skip_special_tokens=False)
            for sample, text in zip(samples, texts):
                sample["text"] = text
        for sample in samples:
            sample["event_hit"] = bool(event_indicator[sample["batch_index"]])
        result["elite_samples"] = samples
        return result


    def _print_ce_fit(self, metrics):
        if not metrics:
            return
        if metrics.get("update_skipped"):
            print("ce_fit: skipped (event-rate target reached)")
            return
        print(f"ce_fit: method={metrics.get('fit_method')} "
              f"gradient_steps={metrics.get('gradient_steps', 0)} "
              f"improvement={metrics['objective_improvement']:.6g} "
              f"elite_ess={metrics['elite_weight_ess']:.2f}/{metrics['elite_count']}")
        samples = metrics.get("elite_samples", [])[:self.num_sample_sentences]
        if not samples:
            return
        print(f"sample_sentences[top_elites] (ranked by {metrics.get('score_mode', 'score')}; "
              "higher score is better):")
        prefix = str(self.metrics_atomic["metadata"].get("context", ""))
        for sample in samples:
            text = prefix + sample["text"] if "text" in sample else str(sample["token_ids"])
            print(f"  elite_{sample['rank']} (batch_idx={sample['batch_index']}, "
                  f"score={sample['score']:.5g}, elite_weight={sample['weight']:.4%}, "
                  f"event_hit={sample.get('event_hit')})={self._single_line_literal(text)}")


    def print_ce_iteration(self, row, generated_tokens=None, event_indicator=None):
        """Use the original logger layout for the memory-bounded CE runner."""
        phase = str(self.metrics_atomic["metadata"].get("phase", "train")).upper()
        print(f"\n==================== ITERATION [{phase}] {row['step']:04d}/{self.total_steps:04d} ====================")
        print(f"main={row['main_loss']:.6e}")
        rare_ess = row.get("rare_event_ess")
        rare_ess_str = "None" if rare_ess is None else f"{rare_ess:.4f}"
        print(f"ess={row['population_ess']:.4f}  rare_event_ess={rare_ess_str}")
        print(f"event_occur_pct={row['event_rate'] * 100:.2f}%")
        self.print_estimate_summary(row['is_estimate'])
        print("-" * 40)
        self._print_sample_sentences(generated_tokens, event_indicator)
        self._print_ce_fit(row.get('proposal_metrics'))
        print("", flush=True)
