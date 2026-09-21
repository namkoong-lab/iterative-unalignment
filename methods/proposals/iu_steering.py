"""Gradient-based single-vector steering trained with the IU loss."""

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
from proposals._base import Proposal, rollout_batch as _rollout_batch_helper
from proposals._steering_wrappers import (
    ActivationSteeringWrapper,
    LogitSteeringWrapper,
)


class IUSteeringProposal(Proposal):
    """IU-style proposal where Q is parameterised by a single learnable
    steering vector. Subclasses select MODE = 'activation' | 'logit'."""

    MODE: str = ""  # set by concrete subclass

    def __init__(
        self,
        *,
        base_model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        model_source: str,
        steering_init_scale: float = 0.01,
        use_chat_template: bool = False,
    ) -> None:
        if self.MODE not in ("activation", "logit"):
            raise ValueError(
                "IUSteeringProposal must be subclassed with MODE in {'activation','logit'}; "
                f"got {self.MODE!r}"
            )

        self.tokenizer = tokenizer
        self.device = device
        self.model_source = model_source
        self.uses_lora = False
        self.use_chat_template = bool(use_chat_template)

        self.steering_init_scale = float(steering_init_scale)

        base_model.to(device)
        for param in base_model.parameters():
            param.requires_grad = False

        if self.MODE == "activation":
            self.param_dim = hidden_size_of(base_model)
            wrapper_cls = ActivationSteeringWrapper
        else:
            self.param_dim = vocab_size_of(base_model)
            wrapper_cls = LogitSteeringWrapper

        self._wrapper = wrapper_cls(
            base_model=base_model,
            mode="trainable",
            param_dim=self.param_dim,
            device=device,
            init_scale=self.steering_init_scale,
        )
        self._wrapper.to(device)
        self.model = self._wrapper
        self.model_class_name = base_model.__class__.__name__

        self._optimizer: Optional[torch.optim.Optimizer] = None
        self.update_steps = 0

    @classmethod
    def _init_scale_attr(cls) -> str:
        return (
            "iu_act_steering_init_scale"
            if cls.MODE == "activation"
            else "iu_logit_steering_init_scale"
        )

    @classmethod
    def from_args(cls, args) -> "IUSteeringProposal":
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
            steering_init_scale=getattr(args, cls._init_scale_attr(), 0.01),
            use_chat_template=bool(getattr(args, "use_chat_template", False)),
        )

    def build_optimizer(self, lr: float) -> None:
        self._optimizer = torch.optim.AdamW([self._wrapper.steering_vector], lr=lr)

    def rollout_batch(
        self,
        prefix_ids: torch.Tensor,
        batch_size: int,
        k: int,
    ) -> tuple[torch.Tensor, int]:
        return _rollout_batch_helper(
            model=self._wrapper,
            prefix_ids=prefix_ids,
            batch_size=batch_size,
            k=k,
            tokenizer=self.tokenizer,
        )

    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._wrapper(input_ids=input_ids).logits

    def update(
        self,
        main_loss_per_sample: torch.Tensor,
        reg_loss: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor:
        if self._optimizer is None:
            raise RuntimeError("Optimizer is not initialized. Call build_optimizer() before update().")

        log_prefix = f"[iu_{self.MODE} step {self.update_steps + 1}]"

        self._wrapper.train()
        total_loss = main_loss_per_sample.mean() + (lambda_value * reg_loss.mean())

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"{log_prefix} non-finite loss: {total_loss.item():.4g}")

        self._optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        steering = self._wrapper.steering_vector
        if steering.grad is None:
            raise RuntimeError(
                f"IUSteeringProposal({self.MODE}).update() produced no gradient on the "
                "steering vector. Check that the loss flows through the wrapper's hook."
            )
        if not torch.isfinite(steering.grad).all():
            raise RuntimeError(f"{log_prefix} non-finite gradient on steering vector")

        self._optimizer.step()

        self.update_steps += 1
        return total_loss

    def freeze_for_eval(self) -> None:
        self._wrapper.steering_vector.requires_grad = False
        self._wrapper.eval()

    def save(self, output_dir: str) -> str:
        artifact_dir = os.path.join(os.path.abspath(output_dir), "trained_model")
        os.makedirs(artifact_dir, exist_ok=True)

        save_dict = {
            "steering_vector": self._wrapper.steering_vector.detach().cpu(),
            "param_dim": self.param_dim,
            "steering_init_scale": self.steering_init_scale,
            "model_source": self.model_source,
            "update_steps": self.update_steps,
        }
        # Save the dimension under its mode-specific name as well.
        if self.MODE == "activation":
            save_dict["hidden_size"] = self.param_dim
        else:
            save_dict["vocab_size"] = self.param_dim

        torch.save(save_dict, os.path.join(artifact_dir, "steering_vector.pt"))
        self.tokenizer.save_pretrained(artifact_dir)

        with open(os.path.join(artifact_dir, "save_mode.txt"), "w", encoding="utf-8") as f:
            f.write(f"mode: iu_{self.MODE}_steering_vector\n")

        return artifact_dir


class IUActivationProposal(IUSteeringProposal):
    """IU steering on the residual stream (hidden-size vector)."""

    MODE = "activation"

    @classmethod
    def add_argparse_group(cls, parser) -> None:
        group = parser.add_argument_group(
            "IU_ACTIVATION proposal (used only when --proposal_type IU_ACTIVATION)"
        )
        group.add_argument(
            "--iu_act_steering_init_scale", type=float, default=0.01,
            help="Initial value for every entry of the learnable activation steering vector.",
        )


class IULogitProposal(IUSteeringProposal):
    """IU steering on the LM head's logits (vocab-size vector)."""

    MODE = "logit"

    @classmethod
    def add_argparse_group(cls, parser) -> None:
        group = parser.add_argument_group(
            "IU_LOGIT proposal (used only when --proposal_type IU_LOGIT)"
        )
        group.add_argument(
            "--iu_logit_steering_init_scale", type=float, default=0.01,
            help=(
                "Std of the Gaussian used to initialize the vocab-sized logit steering "
                "vector (entries are randn * init_scale; constant init would be a softmax no-op)."
            ),
        )
