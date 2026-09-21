"""Time training, frozen importance sampling, and naive Monte Carlo batches."""

from __future__ import annotations

import math
import os
import pickle
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator

import torch


TRAIN_OPS = ("q_inference", "q_forward", "p_forward", "backprop", "ce_fit")
INFERENCE_OPS = ("q_inference", "q_forward", "p_forward")


def computational_efficiency_improvement(
    *,
    n_mc_star: float,
    n_is_star: float,
    cost_mc: float,
    cost_is: float,
    cost_train: float,
) -> float:
    """Compute the MC-to-IS cost ratio, including proposal training."""
    if n_mc_star <= 0.0 or n_is_star <= 0.0:
        raise ValueError("N_MC* and N_IS* must be positive.")
    if cost_mc <= 0.0 or cost_is <= 0.0:
        raise ValueError("cost_MC and cost_IS must be positive.")
    if cost_train < 0.0:
        raise ValueError("cost_train must be non-negative.")
    denom = cost_train + n_is_star * cost_is
    if denom <= 0.0:
        raise ValueError("denominator cost_train + N_IS* · cost_IS must be positive.")
    return (n_mc_star * cost_mc) / denom


def tracking(tracker: "ComputeCostTracker | None", op: str):
    """Context manager: time `op` if a tracker is present, else a no-op."""
    if tracker is None:
        return nullcontext()
    return tracker.track(op)


def _mean(values: list[float], *, warmup: int) -> float | None:
    usable = values[warmup:] if len(values) > warmup else values
    if not usable:
        return None
    return float(sum(usable) / len(usable))


def _device_label(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


class ComputeCostTracker:
    """Full-step and per-operation timings for training and frozen evaluation."""

    def __init__(
        self,
        *,
        device: torch.device,
        batch_size: int,
        k: int,
        warmup_steps: int = 2,
        enabled: bool = True,
        output_dir: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.k = int(k)
        self.warmup_steps = max(0, int(warmup_steps))
        self.enabled = bool(enabled)
        self.output_dir = os.path.abspath(output_dir) if output_dir else None
        self.device_label = _device_label(self.device)
        self.steps: list[dict[str, Any]] = []
        self.p_inference_times: list[float] = []
        self._current: dict[str, float] = {}
        self._phase: str | None = None
        self._step_started: float | None = None

    def _sync(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextmanager
    def track(self, op: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._current[op] = self._current.get(op, 0.0) + (time.perf_counter() - t0)

    def begin_phase(self, phase_name: str) -> None:
        self._phase = str(phase_name)

    def begin_step(self) -> None:
        if not self.enabled:
            return
        if self._step_started is not None:
            raise RuntimeError("Previous compute-cost step has not ended")
        self._current = {}
        self._sync()
        self._step_started = time.perf_counter()

    def end_step(self, step: int, *, is_training: bool) -> None:
        if not self.enabled:
            self._current = {}
            return
        if self._step_started is None:
            raise RuntimeError("begin_step() must precede end_step()")
        self._sync()
        self._current["algorithm_total"] = time.perf_counter() - self._step_started
        self._step_started = None
        phase = self._phase or ("train" if is_training else "eval")
        record = {
            "step": int(step),
            "phase": phase,
            "is_training": bool(is_training),
            **{op: float(self._current[op]) for op in self._current},
        }
        self.steps.append(record)
        self._current = {}

    def calibrate_p_inference(
        self,
        rollout_fn: Callable[[], Any],
        *,
        n_warmup: int = 2,
        n_timed: int = 4,
    ) -> None:
        """Time frozen-p generation, event scoring, and the batch mean."""
        if not self.enabled:
            return
        if n_timed <= 0:
            raise ValueError(f"n_timed must be positive, got {n_timed}")
        for _ in range(max(0, int(n_warmup))):
            rollout_fn()
            self._sync()
        times: list[float] = []
        for _ in range(int(n_timed)):
            self._sync()
            t0 = time.perf_counter()
            rollout_fn()
            self._sync()
            times.append(time.perf_counter() - t0)
        self.p_inference_times.extend(times)

    def _phase_op_times(self, phase: str, op: str) -> list[float]:
        return [
            float(row[op])
            for row in self.steps
            if row.get("phase") == phase and op in row
        ]

    def _phase_means(self, phase: str) -> dict[str, float | None]:
        ops = TRAIN_OPS if phase == "train" else INFERENCE_OPS
        return {
            op: _mean(self._phase_op_times(phase, op), warmup=self.warmup_steps)
            for op in ops
        }

    def composed_costs(self) -> dict[str, Any]:
        """Full algorithm costs and per-operation timing diagnostics."""
        train_means = self._phase_means("train")
        eval_means = self._phase_means("eval")
        train_steps = [row for row in self.steps if row.get("phase") == "train"]
        eval_steps = [row for row in self.steps if row.get("phase") == "eval"]
        train_totals = self._phase_op_times("train", "algorithm_total")
        eval_totals = self._phase_op_times("eval", "algorithm_total")
        # Full costs include every completed step, including warmup steps.
        cost_train = sum(train_totals) if train_totals else None
        cost_is_batch = _mean(eval_totals, warmup=0)
        cost_is = cost_is_batch / self.batch_size if cost_is_batch is not None else None
        per_step_train = _mean(train_totals, warmup=0)
        p_inf_batch = _mean(self.p_inference_times, warmup=0)
        cost_mc = p_inf_batch / self.batch_size if p_inf_batch is not None else None

        def _per_sample(mean_batch: float | None) -> float | None:
            if mean_batch is None:
                return None
            return mean_batch / self.batch_size

        return {
            "device": self.device_label,
            "batch_size": self.batch_size,
            "k": self.k,
            "warmup_steps": self.warmup_steps,
            "n_train_steps_timed": len(train_steps),
            "n_eval_steps_timed": len(eval_steps),
            "n_p_inference_timed": len(self.p_inference_times),
            "train_mean_s_per_batch": train_means,
            "eval_mean_s_per_batch": eval_means,
            "train_mean_s_per_trajectory": {
                op: _per_sample(train_means.get(op)) for op in TRAIN_OPS
            },
            "p_inference_mean_s_per_batch": p_inf_batch,
            "cost_MC": cost_mc,
            "cost_IS": cost_is,
            "cost_IS_includes_q_forward": True,
            "cost_train": cost_train,
            "cost_train_per_step": per_step_train,
            "cost_train_full": cost_train,
            "cost_IS_full": cost_is,
            "full_cost_scope": "generation through estimate; excludes logging, I/O, and model loading",
        }

    def format_summary(self) -> str:
        costs = self.composed_costs()

        def _fmt(v: float | None) -> str:
            if v is None or not math.isfinite(v):
                return "None"
            return f"{v:.6e}"

        lines = [
            "# Wall-clock seconds after CUDA synchronize. N_MC* and N_IS* come from the Winkler analysis.",
            "",
            f"device: {costs['device']}",
            f"batch_size: {costs['batch_size']}",
            f"k: {costs['k']}",
            f"warmup_steps (per-op means only): {costs['warmup_steps']}",
            f"full_cost_scope: {costs['full_cost_scope']}",
            f"n_train_steps_timed: {costs['n_train_steps_timed']}",
            f"n_eval_steps_timed: {costs['n_eval_steps_timed']}",
            f"n_p_inference_timed: {costs['n_p_inference_timed']}",
            f"cost_IS_includes_q_forward: {costs['cost_IS_includes_q_forward']}",
            "",
        ]
        for op in TRAIN_OPS:
            lines.append(
                f"train_{op}_s_per_batch: {_fmt(costs['train_mean_s_per_batch'].get(op))}"
            )
        lines.append(
            "p_inference_s_per_batch: "
            f"{_fmt(costs['p_inference_mean_s_per_batch'])}"
        )
        eval_means = costs["eval_mean_s_per_batch"]
        for op in ("q_inference", "q_forward", "p_forward"):
            lines.append(f"eval_{op}_s_per_batch: {_fmt(eval_means.get(op))}")

        lines.append("")
        for op in TRAIN_OPS:
            lines.append(
                f"train_{op}_s_per_trajectory: "
                f"{_fmt(costs['train_mean_s_per_trajectory'].get(op))}"
            )

        lines.extend(
            [
                "",
                f"cost_MC: {_fmt(costs['cost_MC'])}",
                f"cost_IS: {_fmt(costs['cost_IS'])}",
                f"cost_train: {_fmt(costs['cost_train'])}",
                f"cost_train_per_step: {_fmt(costs['cost_train_per_step'])}",
            ]
        )
        return "\n".join(lines) + "\n"

    def save(self, output_dir: str | None = None) -> str | None:
        """Write compute_cost.txt and compute_cost.pkl. Returns the txt path."""
        if not self.enabled:
            return None
        directory = output_dir or self.output_dir
        if not directory:
            raise ValueError("save() requires output_dir")
        directory = os.path.abspath(directory)
        os.makedirs(directory, exist_ok=True)
        txt_path = os.path.join(directory, "compute_cost.txt")
        pkl_path = os.path.join(directory, "compute_cost.pkl")
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(self.format_summary())
        payload = {
            "metadata": {
                "device": self.device_label,
                "batch_size": self.batch_size,
                "k": self.k,
                "warmup_steps": self.warmup_steps,
            },
            "steps": self.steps,
            "p_inference_times": list(self.p_inference_times),
            "composed": self.composed_costs(),
        }
        with open(pkl_path, "wb") as handle:
            pickle.dump(payload, handle)
        return txt_path
