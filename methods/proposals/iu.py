from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from model_loading import load_trainable_components
from proposals._base import Proposal, rollout_batch


class IUProposal(Proposal):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: torch.device,
        model_source: str,
        uses_lora: bool,
        use_chat_template: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_source = model_source
        self.uses_lora = uses_lora
        self.use_chat_template = use_chat_template
        self.model_class_name = model.__class__.__name__
        self.update_steps = 0
        self._optimizer: Optional[torch.optim.Optimizer] = None

    @classmethod
    def from_args(cls, args) -> "IUProposal":
        model, tokenizer, device, model_source, uses_lora = load_trainable_components(args)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_source=model_source,
            uses_lora=uses_lora,
            use_chat_template=bool(getattr(args, "use_chat_template", False)),
        )

    def build_optimizer(self, lr: float) -> None:
        self._optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def rollout_batch(
        self,
        prefix_ids: torch.Tensor,
        batch_size: int,
        k: int,
    ) -> tuple[torch.Tensor, int]:
        return rollout_batch(
            model=self.model,
            prefix_ids=prefix_ids,
            batch_size=batch_size,
            k=k,
            tokenizer=self.tokenizer,
        )

    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids).logits

    def update(
        self,
        main_loss_per_sample: torch.Tensor,
        reg_loss: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor:
        if self._optimizer is None:
            raise RuntimeError("Optimizer is not initialized. Call build_optimizer() before update().")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Proposal model has no trainable parameters.")

        self.model.train()
        total_loss = main_loss_per_sample.mean() + (lambda_value * reg_loss.mean())

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"[step {self.update_steps + 1}] non-finite loss: {total_loss.item():.4g}"
            )

        self._optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Guard against disconnected objectives (all trainable grads are None),
        # while allowing legitimate zero-gradient steps.
        if not any(param.grad is not None for param in trainable_params):
            raise RuntimeError(
                "Proposal.update() produced no gradients for any trainable parameter. "
                "Check objective graph wiring."
            )

        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable_params):
            raise RuntimeError(
                f"[step {self.update_steps + 1}] non-finite gradient on trainable params"
            )

        self._optimizer.step()

        self.update_steps += 1
        return total_loss

    def freeze_for_eval(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def save(self, output_dir: str) -> str:
        model_dir = os.path.join(os.path.abspath(output_dir), "trained_model")
        os.makedirs(model_dir, exist_ok=True)

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        mode_label = "lora_adapter" if self.uses_lora else "full_model"
        with open(os.path.join(model_dir, "save_mode.txt"), "w", encoding="utf-8") as f:
            f.write(f"mode: {mode_label}\n")

        return model_dir
