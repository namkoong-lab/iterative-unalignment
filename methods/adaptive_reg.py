"""ESS-driven dual controller that updates the regularization strength lambda."""

from __future__ import annotations

import math

import torch


class AdaptiveReg:
    """Raises lambda when population ESS is below target, lowers it when above."""

    def __init__(
        self,
        *,
        ess_target: float,
        init_lambda: float,
        dual_lr: float = 0.05,
        lambda_floor: float = 0.01,
        dual_optimizer_type: str = "sgd",
        device: torch.device | str = "cpu",
    ) -> None:
        if lambda_floor <= 0.0:
            raise ValueError(f"lambda_floor must be > 0, got {lambda_floor}")
        if dual_lr <= 0.0:
            raise ValueError(f"dual_lr must be > 0, got {dual_lr}")
        if not (0.0 <= ess_target <= 1.0):
            raise ValueError(f"ess_target should be in [0, 1], got {ess_target}")

        self.ess_target = float(ess_target)
        self.lambda_floor = float(lambda_floor)

        self._min_log_lambda = math.log(max(self.lambda_floor, 1e-12))

        safe_init = max(float(init_lambda), self.lambda_floor)
        self._log_lambda = torch.tensor(
            [math.log(safe_init)],
            device=torch.device(device) if isinstance(device, str) else device,
            requires_grad=True,
        )

        if dual_optimizer_type == "sgd":
            self._optimizer: torch.optim.Optimizer = torch.optim.SGD(
                [self._log_lambda], lr=dual_lr,
            )
        elif dual_optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                [self._log_lambda], lr=dual_lr,
            )
        else:
            raise ValueError(
                f"Unknown dual_optimizer_type: {dual_optimizer_type!r}. "
                f"Supported: 'sgd', 'adam'."
            )

    @property
    def current_lambda(self) -> float:
        """Current lambda value (always >= lambda_floor)."""
        return max(float(torch.exp(self._log_lambda.detach()).item()), self.lambda_floor)

    def step(self, population_ess: float | None, ess_target: float | None = None) -> None:
        """Take one dual step from population ESS. No-op if ESS is missing or non-finite."""
        if population_ess is None or not math.isfinite(float(population_ess)):
            return

        target = self.ess_target if ess_target is None else float(ess_target)
        ess_error = float(population_ess) - target

        self._optimizer.zero_grad(set_to_none=True)
        dual_loss = ess_error * self._log_lambda
        dual_loss.backward()
        self._optimizer.step()

        with torch.no_grad():
            self._log_lambda.clamp_(min=self._min_log_lambda)
