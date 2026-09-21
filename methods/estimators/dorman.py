"""Annealed TPS and MBAR for fixed-length completions, after Dorman et al. (2026)."""

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol

import numpy as np


class Backend(Protocol):
    """Fixed-length base-model sampling and a deterministic trajectory score."""

    length: int

    def sample(self, batch_size: int) -> np.ndarray: ...
    def regenerate(self, states: np.ndarray, cut: int) -> np.ndarray: ...
    def evaluate(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def synchronize(self) -> None: ...


@dataclass(frozen=True)
class TPSConfig:
    schedules: tuple[tuple[float, ...], ...] = (
        tuple(i / 10 for i in range(1, 11)),
        tuple(-i / 10 for i in range(1, 11)),
    )
    chains: int = 10
    steps_per_bias: int = 40_000
    direct_samples: int = 200_000
    direct_batch_size: int = 128
    burnin_fraction: float = 0.1
    gr_threshold: float = 1.1
    filter_gr: bool = True

    def __post_init__(self):
        for name in ("chains", "steps_per_bias", "direct_samples", "direct_batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.chains < 2:
            raise ValueError("At least two independent chains are required for GR diagnostics")
        if not math.isfinite(self.burnin_fraction) or not 0 <= self.burnin_fraction < 1:
            raise ValueError("burnin_fraction must be in [0, 1)")
        if self.steps_per_bias - int(self.steps_per_bias * self.burnin_fraction) < 2:
            raise ValueError("At least two steps per chain must remain after burn-in")
        if not math.isfinite(self.gr_threshold) or self.gr_threshold <= 1:
            raise ValueError("gr_threshold must be finite and greater than one")
        flat = [b for schedule in self.schedules for b in schedule]
        if not self.schedules or any(not schedule for schedule in self.schedules):
            raise ValueError("Provide at least one nonempty bias schedule")
        if any(not math.isfinite(b) or b == 0 for b in flat) or len(set(flat)) != len(flat):
            raise ValueError("Biases must be finite, nonzero and distinct; direct samples supply b=0")
        for schedule in self.schedules:
            if any(b * schedule[0] <= 0 for b in schedule) or any(
                abs(a) >= abs(b) for a, b in zip(schedule, schedule[1:])
            ):
                raise ValueError("Each schedule must have one sign and increasing magnitude")


def gelman_rubin(scores: np.ndarray) -> float:
    """Classical R-hat; undefined diagnostics return infinity."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        return math.inf
    n = x.shape[1]
    within = x.var(axis=1, ddof=1).mean()
    if within <= 0:
        return math.inf
    between = n * x.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) * within / n + between / n) / within))


def mbar_indicator_estimate(biases, score_groups, indicator_groups) -> dict:
    """Estimate the binary event with MBAR normalizers and per-source IS weights."""
    from pymbar import MBAR
    from pymbar.utils import ConvergenceError, ParameterError

    biases = np.asarray(biases, dtype=np.float64)
    if biases.ndim != 1 or len(biases) != len(score_groups) or len(biases) != len(indicator_groups):
        raise ValueError("One score and indicator group is required per bias")
    if not np.isfinite(biases).all() or len(set(biases)) != len(biases) or 0.0 not in biases:
        raise ValueError("Distinct finite biases including the unbiased state b=0 are required")
    scores = [np.asarray(s, dtype=np.float64).reshape(-1) for s in score_groups]
    hits = [np.asarray(h, dtype=np.float64).reshape(-1) for h in indicator_groups]
    if any(len(s) == 0 or s.shape != h.shape for s, h in zip(scores, hits)):
        raise ValueError("Nonempty aligned score/indicator groups are required")
    s, h = np.concatenate(scores), np.concatenate(hits)
    if not np.isfinite(s).all() or not np.isin(h, [0, 1]).all():
        raise ValueError("Scores must be finite and indicators binary")
    counts = np.array([len(group) for group in scores], dtype=int)
    # Centering the score changes only state normalizers, not any weights.
    score_center = s.mean()
    u_kn = -biases[:, None] * (s - score_center)[None, :]
    if not np.isfinite(u_kn).all():
        raise ValueError("Nonfinite reduced potentials; reduce the bias magnitude")
    try:
        fit = MBAR(u_kn, counts, solver_protocol="robust", relative_tolerance=1e-10)
    except (ConvergenceError, ParameterError) as exc:
        raise RuntimeError(f"MBAR solver failed: {exc}") from exc
    all_weights = fit.W_nk
    if (
        not np.isfinite(all_weights).all()
        or not np.allclose(all_weights.sum(axis=0), 1.0, rtol=0, atol=1e-7)
        or not np.allclose(all_weights @ counts, 1.0, rtol=0, atol=1e-7)
    ):
        raise RuntimeError("MBAR weights do not satisfy the normalization equations")
    zero_index = int(np.flatnonzero(biases == 0)[0])
    # log(p/q_b) = log(Z_b/Z_0) - b*score; MBAR stores f_b = -log(Z_b).
    log_weights = np.concatenate([
        fit.f_k[zero_index] - fit.f_k[index] - bias * (group - score_center)
        for index, (bias, group) in enumerate(zip(biases, scores))
    ])
    if not np.isfinite(log_weights).all():
        raise RuntimeError("Nonfinite source-state importance weights")
    weights = np.exp(log_weights - log_weights.max())
    return {
        "estimate": float((weights @ h) / weights.sum()),
        "retained_samples": int(counts.sum()),
        "retained_event_hits": int(h.sum()),
        "state_biases": biases.tolist(),
        "state_counts": counts.tolist(),
        "overlap_matrix": fit.compute_overlap()["matrix"].tolist() if len(biases) > 1 else [[1.0]],
    }


def estimate(
    backend: Backend, config: TPSConfig, seed: int, *,
    progress: Callable[[dict], None] | None = None,
    log_every: int = 200,
) -> tuple[dict, dict]:
    """Return one estimate and full traces. Seed backend sampling separately."""
    if backend.length <= 0 or log_every <= 0:
        raise ValueError("The completion length and log_every must be positive")
    rng = np.random.default_rng(seed)
    backend.synchronize()
    started = time.perf_counter()
    generated_tokens = evaluated_trajectories = 0
    attempted = accepted_total = 0
    reporting_seconds = 0.0
    diagnostics = []

    def report(phase, **fields):
        nonlocal reporting_seconds
        if progress is not None:
            backend.synchronize()
            before = time.perf_counter()
            try:
                progress({"phase": phase, "generated_tokens": generated_tokens, **fields})
            finally:
                reporting_seconds += time.perf_counter() - before

    def evaluate(states):
        nonlocal evaluated_trajectories
        scores, hits = backend.evaluate(states)
        scores, hits = np.asarray(scores, dtype=float), np.asarray(hits, dtype=float)
        if scores.shape != (len(states),) or hits.shape != scores.shape:
            raise ValueError("Backend must return one score and indicator per trajectory")
        if not np.isfinite(scores).all() or not np.isin(hits, [0, 1]).all():
            raise ValueError("Backend returned a nonfinite score or nonbinary indicator")
        evaluated_trajectories += len(states)
        return scores, hits

    direct_scores = np.empty(config.direct_samples)
    direct_hits = np.empty(config.direct_samples, dtype=np.uint8)
    for batch, start in enumerate(range(0, config.direct_samples, config.direct_batch_size), 1):
        stop = min(start + config.direct_batch_size, config.direct_samples)
        states = backend.sample(stop - start)
        generated_tokens += (stop - start) * backend.length
        direct_scores[start:stop], direct_hits[start:stop] = evaluate(states)
        if batch % log_every == 0 or stop == config.direct_samples:
            report("direct", samples_done=stop, samples_total=config.direct_samples)
    biases, score_groups, hit_groups = [0.0], [direct_scores], [direct_hits]
    traces = {"direct_scores": direct_scores, "direct_indicators": direct_hits}
    burnin = int(config.steps_per_bias * config.burnin_fraction)
    for schedule_index, schedule in enumerate(config.schedules):
        states = backend.sample(config.chains)
        generated_tokens += config.chains * backend.length
        current_scores, current_hits = evaluate(states)
        for bias_index, bias in enumerate(schedule):
            scores = np.empty((config.chains, config.steps_per_bias))
            hits = np.empty(scores.shape, dtype=np.uint8)
            accepted = 0
            for step in range(config.steps_per_bias):
                # Independent cuts include zero, so the first token can change.
                cuts = rng.integers(0, backend.length, size=config.chains)
                proposed = states.copy()
                for cut in np.unique(cuts):
                    indices = np.flatnonzero(cuts == cut)
                    proposed[indices] = backend.regenerate(states[indices], int(cut))
                    generated_tokens += len(indices) * (backend.length - int(cut))
                proposed_scores, proposed_hits = evaluate(proposed)
                # Base-law suffix probabilities cancel in the MH ratio.
                log_alpha = np.minimum(0.0, bias * (proposed_scores - current_scores))
                keep = np.log(rng.random(config.chains)) < log_alpha
                states[keep] = proposed[keep]
                current_scores[keep] = proposed_scores[keep]
                current_hits[keep] = proposed_hits[keep]
                accepted += int(keep.sum())
                accepted_total += int(keep.sum())
                attempted += config.chains
                # Rejections must repeat the current state in the trace.
                scores[:, step] = current_scores
                hits[:, step] = current_hits
                if (step + 1) % log_every == 0:
                    report("tps", schedule_index=schedule_index, bias=bias,
                           steps_done=step + 1, steps_total=config.steps_per_bias)
            gr = gelman_rubin(scores[:, burnin:])
            retained = not config.filter_gr or gr < config.gr_threshold
            if retained:
                biases.append(bias)
                score_groups.append(scores[:, burnin:].reshape(-1))
                hit_groups.append(hits[:, burnin:].reshape(-1))
            diagnostic = {
                "schedule_index": schedule_index, "bias_index": bias_index, "bias": bias,
                "gr": gr, "retained": retained,
                "acceptance_rate": accepted / (config.chains * config.steps_per_bias),
                "post_burnin_event_hits": int(hits[:, burnin:].sum()),
                "post_burnin_samples": int(hits[:, burnin:].size),
            }
            diagnostics.append(diagnostic)
            key = f"schedule_{schedule_index}_bias_{bias_index}"
            traces[key + "_scores"] = scores
            traces[key + "_indicators"] = hits
            report("bias_complete", **diagnostic)
    backend.synchronize()
    sampling_seconds = time.perf_counter() - started - reporting_seconds
    report("mbar")
    try:
        result = mbar_indicator_estimate(biases, score_groups, hit_groups)
        result["status"] = "ok" if len(biases) > 1 else "direct_only"
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        result = {"estimate": None, "status": "mbar_failed", "error": str(exc)}
    backend.synchronize()
    total_seconds = time.perf_counter() - started - reporting_seconds
    result.update({
        "seed": int(seed), "total_seconds": total_seconds,
        "sampling_seconds": sampling_seconds,
        "reconstruction_seconds": total_seconds - sampling_seconds,
        "reporting_seconds": reporting_seconds,
        "wall_seconds": total_seconds + reporting_seconds,
        "generated_tokens": generated_tokens,
        "evaluated_trajectories": evaluated_trajectories,
        "attempted_moves": attempted, "accepted_moves": accepted_total,
        "direct_event_hits": int(direct_hits.sum()), "diagnostics": diagnostics,
    })
    return result, traces
