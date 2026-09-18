"""CEM steering proposals: sample candidate vectors, score the rare-event surrogate, and update (mu, sigma) without gradients."""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from model_loading import (
    _load_causal_lm,
    _load_tokenizer,
    _resolve_device,
    resolve_model_source,
)
from proposals._arch import hidden_size_of, vocab_size_of
from proposals._base import Proposal
from proposals._ce_common import CEEarlyStopMonitor, add_ce_argparse_group
from proposals._steering_wrappers import (
    ActivationSteeringWrapper,
    LogitSteeringWrapper,
)


class CEProposal(Proposal):
    """CEM proposal over a single steering vector. Subclasses set MODE."""

    MODE: str = ""  # set by concrete subclass

    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        model_source: str,
        elite_ratio: float = 0.1,
        smoothing: float = 0.7,
        sigma_init: float = 0.02,
        sigma_min: float = 0.001,
        steering_init_scale: float = 0.0,
        generation_temperature: float = 1.0,
        eval_use_mean_only: bool = True,
        early_stop_on_low_ess: bool = True,
        use_chat_template: bool = False,
    ) -> None:
        if self.MODE not in ("activation", "logit"):
            raise ValueError(
                "CEProposal must be subclassed with MODE in {'activation','logit'}; "
                f"got {self.MODE!r}"
            )
        if not (0.0 < elite_ratio <= 1.0):
            raise ValueError(f"elite_ratio must be in (0, 1], got {elite_ratio}")
        if not (0.0 < smoothing <= 1.0):
            raise ValueError(f"smoothing must be in (0, 1], got {smoothing}")
        if sigma_init <= 0.0:
            raise ValueError(f"sigma_init must be > 0, got {sigma_init}")
        if sigma_min < 0.0:
            raise ValueError(f"sigma_min must be >= 0, got {sigma_min}")

        self.tokenizer = tokenizer
        self.device = device
        self.model_source = model_source
        self.uses_lora = False
        self.use_chat_template = bool(use_chat_template)

        self.elite_ratio = float(elite_ratio)
        self.smoothing = float(smoothing)
        self.sigma_init = float(sigma_init)
        self.sigma_min = float(sigma_min)
        self.steering_init_scale = float(steering_init_scale)
        self.generation_temperature = float(generation_temperature)
        self.eval_use_mean_only = bool(eval_use_mean_only)

        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.to(device)

        if self.MODE == "activation":
            self.param_dim = hidden_size_of(base_model)
            wrapper_cls = ActivationSteeringWrapper
        else:
            self.param_dim = vocab_size_of(base_model)
            wrapper_cls = LogitSteeringWrapper

        # CEM mode: wrapper has no trainable parameter; per-row vectors are
        # fed in each step. ``init_scale`` is unused because the wrapper
        # owns no parameter, but the constructor still requires it.
        self._wrapper = wrapper_cls(
            base_model=base_model,
            mode="cem",
            param_dim=self.param_dim,
            device=device,
            init_scale=0.0,
        )
        self.model = self._wrapper
        self.model_class_name = base_model.__class__.__name__

        # mu = constant init is a softmax no-op for logit mode (shift-
        # invariance); randomness comes per-step from sigma * randn during
        # candidate sampling. Activation mode also accepts a constant init.
        self.mu = torch.full(
            (self.param_dim,), self.steering_init_scale, device=device, dtype=torch.float32,
        )
        self.sigma = torch.full(
            (self.param_dim,), self.sigma_init, device=device, dtype=torch.float32,
        )

        self._current_candidates: Optional[torch.Tensor] = None
        self._frozen = False
        self.update_steps = 0

        self._early_stop_monitor = CEEarlyStopMonitor(enabled=early_stop_on_low_ess)

    @classmethod
    def add_argparse_group(cls, parser) -> None:
        add_ce_argparse_group(parser)

    @classmethod
    def from_args(cls, args) -> "CEProposal":
        source = resolve_model_source(args)
        device = _resolve_device(args.device)
        tokenizer = _load_tokenizer(source)
        base_model = _load_causal_lm(source)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id

        return cls(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            model_source=source,
            elite_ratio=getattr(args, "ce_elite_ratio", 0.1),
            smoothing=getattr(args, "ce_smoothing", 0.7),
            sigma_init=getattr(args, "ce_sigma_init", 0.02),
            sigma_min=getattr(args, "ce_sigma_min", 0.001),
            steering_init_scale=getattr(args, "ce_steering_init_scale", 0.0),
            generation_temperature=getattr(args, "ce_generation_temperature", 1.0),
            eval_use_mean_only=getattr(args, "ce_eval_use_mean_only", True),
            early_stop_on_low_ess=getattr(args, "ce_early_stop_on_low_ess", True),
            use_chat_template=bool(getattr(args, "use_chat_template", False)),
        )

    def should_stop(
        self,
        *,
        step: int,
        rare_event_ess: float | None,
        rare_event_count: int,
        rare_event_ess_min_count: int,
        ess_target: float,
    ) -> tuple[bool, str | None]:
        return self._early_stop_monitor.check(
            step=step,
            rare_event_ess=rare_event_ess,
            rare_event_count=rare_event_count,
            rare_event_ess_min_count=rare_event_ess_min_count,
            ess_target=ess_target,
        )

    def build_optimizer(self, lr: float) -> None:
        # CEM is gradient-free; lr is unused but the trainer expects this hook.
        return None

    def _sample_candidates(self, batch_size: int) -> torch.Tensor:
        if self._frozen and self.eval_use_mean_only:
            return self.mu.unsqueeze(0).expand(batch_size, -1).contiguous()
        epsilon = torch.randn(batch_size, self.param_dim, device=self.device)
        return self.mu.unsqueeze(0) + epsilon * self.sigma.unsqueeze(0)

    @torch.no_grad()
    def rollout_batch(
        self,
        prefix_ids: torch.Tensor,
        batch_size: int,
        k: int,
    ) -> tuple[torch.Tensor, int]:
        if prefix_ids is None or prefix_ids.shape[1] == 0:
            raise ValueError(
                "rollout_batch requires a non-empty prefix_ids tensor. "
                "Provide an explicit prefix that includes the appropriate BOS token for your model."
            )

        candidates = self._sample_candidates(batch_size)
        self._current_candidates = candidates
        self._wrapper.set_current_vectors(candidates)
        self._wrapper.eval()

        ids = prefix_ids.repeat(batch_size, 1)
        prefix_len = prefix_ids.shape[1]

        for _ in range(k):
            next_logits = self._wrapper(input_ids=ids).logits[:, -1, :]
            if self.generation_temperature != 1.0:
                next_logits = next_logits / self.generation_temperature
            next_id = torch.distributions.Categorical(logits=next_logits).sample()
            ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)

        return ids, prefix_len

    @torch.no_grad()
    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._current_candidates is None:
            raise RuntimeError(
                "compute_logits called before rollout_batch sampled candidates."
            )
        if self._current_candidates.shape[0] != input_ids.shape[0]:
            raise RuntimeError(
                "Batch dim mismatch between current candidates "
                f"({self._current_candidates.shape[0]}) and input_ids "
                f"({input_ids.shape[0]})."
            )
        self._wrapper.set_current_vectors(self._current_candidates)
        return self._wrapper(input_ids=input_ids).logits

    def update(
        self,
        main_loss_per_sample: torch.Tensor,
        reg_loss: torch.Tensor,
        lambda_value: float,
        log_importance_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # reg_loss / lambda_value are part of the Proposal contract but the
        # CE baseline scores on the surrogate alone — alpha-reg is an IU
        # concern. ESS-based early stopping is what keeps Q close to P here.
        del reg_loss, lambda_value

        if self._current_candidates is None:
            raise RuntimeError("update() called before rollout_batch produced candidates.")

        log_prefix = f"[ce_{self.MODE} step {self.update_steps + 1}]"

        scores = main_loss_per_sample.detach().to(self.device)
        if scores.ndim != 1 or scores.shape[0] != self._current_candidates.shape[0]:
            raise ValueError(
                "main_loss_per_sample must have shape [batch_size]; got "
                f"{tuple(scores.shape)} vs candidates {self._current_candidates.shape[0]}"
            )

        if not torch.isfinite(scores).all():
            raise RuntimeError(f"{log_prefix} non-finite CEM scores")

        if not self._frozen:
            with torch.no_grad():
                pop_size = scores.shape[0]
                num_elites = max(1, int(round(pop_size * self.elite_ratio)))
                elite_idx = torch.argsort(scores)[:num_elites]
                elites = self._current_candidates.index_select(0, elite_idx)

                # CE update: importance-weighted mean of the elites. Each elite
                # is weighted by its rollout likelihood ratio w = p/q (the same
                # log_iw = log p - log q the trainer uses for ESS), self-
                # normalized over the elite set via softmax(log w) -- which is
                # exactly w_i / sum_j w_j but numerically stable. Falls back to
                # the uniform mean when the trainer supplies no weights.
                if log_importance_weights is not None:
                    elite_logw = (
                        log_importance_weights.detach()
                        .to(device=self.device, dtype=elites.dtype)
                        .index_select(0, elite_idx)
                    )
                    weights = torch.softmax(elite_logw, dim=0)  # [num_elites]
                else:
                    weights = torch.full(
                        (num_elites,), 1.0 / num_elites, device=self.device,
                    )

                w = weights.unsqueeze(1)  # [num_elites, 1] broadcast over dim
                mu_elite = (w * elites).sum(dim=0)
                if num_elites > 1:
                    # Weighted (biased) std around the weighted elite mean.
                    diff = elites - mu_elite.unsqueeze(0)
                    sigma_elite = torch.sqrt(
                        (w * diff * diff).sum(dim=0).clamp_min(0.0)
                    )
                else:
                    sigma_elite = torch.zeros_like(self.mu)

                self.mu = self.smoothing * mu_elite + (1.0 - self.smoothing) * self.mu
                self.sigma = self.smoothing * sigma_elite + (1.0 - self.smoothing) * self.sigma
                self.sigma = torch.clamp(self.sigma, min=self.sigma_min)

        # `total_loss` is only used by the trainer for logging; CEM does not
        # backward through it.
        total_loss = scores.mean()
        self.update_steps += 1
        return total_loss.detach()

    def freeze_for_eval(self) -> None:
        self._frozen = True
        self._wrapper.eval()

    def save(self, output_dir: str) -> str:
        artifact_dir = os.path.join(os.path.abspath(output_dir), "trained_model")
        os.makedirs(artifact_dir, exist_ok=True)

        save_dict = {
            "steering_vector": self.mu.detach().cpu(),
            "sigma": self.sigma.detach().cpu(),
            "param_dim": self.param_dim,
            "steering_mode": self.MODE,
            "steering_init_scale": self.steering_init_scale,
            "elite_ratio": self.elite_ratio,
            "smoothing": self.smoothing,
            "sigma_init": self.sigma_init,
            "sigma_min": self.sigma_min,
            "model_source": self.model_source,
            "update_steps": self.update_steps,
        }
        torch.save(save_dict, os.path.join(artifact_dir, "steering_vector.pt"))
        self.tokenizer.save_pretrained(artifact_dir)

        with open(os.path.join(artifact_dir, "save_mode.txt"), "w", encoding="utf-8") as f:
            f.write(f"mode: ce_{self.MODE}_steering_vector\n")

        return artifact_dir


class CEActivationProposal(CEProposal):
    """CEM steering on the residual stream (hidden-size vector)."""

    MODE = "activation"


class CELogitProposal(CEProposal):
    """CEM steering on the LM head's logits (vocab-size vector)."""

    MODE = "logit"
