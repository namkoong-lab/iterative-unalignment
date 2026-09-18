"""Proposal interface plus shared encode and rollout helpers."""

from abc import ABC, abstractmethod

import torch


class Proposal(ABC):
    """Proposal interface used by train.py."""

    tokenizer: object
    device: torch.device
    model_source: str
    uses_lora: bool
    model_class_name: str
    use_chat_template: bool = False

    @classmethod
    @abstractmethod
    def from_args(cls, args) -> "Proposal":
        """Construct from an argparse Namespace."""

    @classmethod
    def add_argparse_group(cls, parser) -> None:
        """Register proposal-specific argparse flags. Default: none."""
        return

    @abstractmethod
    def build_optimizer(self, lr: float) -> None:
        """Initialize the proposal's optimizer (gradient-free proposals can no-op)."""

    @abstractmethod
    def rollout_batch(
        self,
        prefix_ids: torch.Tensor,
        batch_size: int,
        k: int,
    ) -> tuple[torch.Tensor, int]:
        """Sample K continuations under the current proposal; return (ids, prefix_len)."""

    @abstractmethod
    def compute_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning [B, T, V] logits under the current proposal."""

    @abstractmethod
    def update(
        self,
        main_loss_per_sample: torch.Tensor,
        reg_loss: torch.Tensor,
        lambda_value: float,
    ) -> torch.Tensor:
        """Run one training step; return the (logged) total loss tensor."""

    @abstractmethod
    def freeze_for_eval(self) -> None:
        """Disable training-side mutation so subsequent rollouts are fixed."""

    @abstractmethod
    def save(self, output_dir: str) -> str:
        """Persist proposal artifacts under output_dir; return the artifact dir."""

    def encode_context(self, context: str) -> torch.Tensor:
        """Tokenize the prompt for rollouts; honours self.use_chat_template."""
        if getattr(self, "use_chat_template", False):
            if not hasattr(self.tokenizer, "apply_chat_template"):
                raise ValueError(
                    "use_chat_template=True but tokenizer has no apply_chat_template; "
                    "use an instruction-tuned model with a chat template."
                )
            messages = [{"role": "user", "content": context}]
            prefix_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            return prefix_ids.to(self.device)
        return encode_prefix(self.tokenizer, context, self.device)

    def should_stop(
        self,
        *,
        step: int,
        rare_event_ess: float | None,
        rare_event_count: int,
        rare_event_ess_min_count: int,
        ess_target: float,
    ) -> tuple[bool, str | None]:
        """Optional early-stop hook; default never stops."""
        return (False, None)


def encode_prefix(tokenizer, context: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids
    return encoded.to(device)


def resolve_single_token_id(tokenizer, token_str: str) -> int:
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"target_token must map to exactly one token. "
            f"Got ids={token_ids} for token_str={token_str!r}"
        )
    return token_ids[0]


@torch.no_grad()
def rollout_batch(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    batch_size: int,
    k: int,
    tokenizer,
) -> tuple[torch.Tensor, int]:
    """
    Sample K tokens from the current model.

    Returns:
        ids: [B, effective_prefix_len + K]
        effective_prefix_len: prefix length used for downstream loss/reg slicing.
    """
    model.eval()

    if prefix_ids is None or prefix_ids.shape[1] == 0:
        raise ValueError(
            "rollout_batch requires a non-empty prefix_ids tensor. "
            "Provide an explicit prefix that includes the appropriate BOS token for your model."
        )

    ids = prefix_ids.repeat(batch_size, 1)
    effective_prefix_len = prefix_ids.shape[1]

    for _ in range(k):
        next_logits = model(input_ids=ids).logits[:, -1, :]
        next_id = torch.distributions.Categorical(logits=next_logits).sample()
        ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)

    return ids, effective_prefix_len
