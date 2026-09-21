"""AND, OR, and ordered-subsequence events over a set of tokens.

The indicator (tokens/mode) is the estimand. An optional surrogate token set
and mode can shape training toward a different event. Sequence mode scores
the next unmatched target at each generated position.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from events._base import RareEvent, assert_per_sample_shape, slice_generated_logits
from proposals._base import resolve_single_token_id


_SUPPORTED_MODES = ("and", "or", "sequence")
_SUPPORTED_SURROGATES = ("sum",)


def _resolve_dedup_tokens(tokenizer, raw_tokens) -> tuple[list[str], list[int]]:
    """Resolve token strings to ids, dedup by id, preserve first-seen order."""
    if isinstance(raw_tokens, (str, bytes)):
        raise ValueError("token list must be a JSON list of strings, not a single string.")
    tokens = [str(t) for t in raw_tokens]
    if not tokens:
        raise ValueError("token list must contain at least one token.")
    kept_tokens: list[str] = []
    token_ids: list[int] = []
    seen: set[int] = set()
    for tok in tokens:
        tid = resolve_single_token_id(tokenizer, tok)
        if tid in seen:
            continue
        seen.add(tid)
        token_ids.append(int(tid))
        kept_tokens.append(tok)
    return kept_tokens, token_ids


def _resolve_ordered_tokens(tokenizer, raw_tokens) -> tuple[list[str], list[int]]:
    """Resolve an ordered token tuple, preserving order and repetitions."""
    if isinstance(raw_tokens, (str, bytes)):
        raise ValueError("token list must be a JSON list of strings, not a single string.")
    tokens = [str(t) for t in raw_tokens]
    if not tokens:
        raise ValueError("token list must contain at least one token.")
    return tokens, [int(resolve_single_token_id(tokenizer, tok)) for tok in tokens]


@dataclass(frozen=True)
class MultiTokenEvent(RareEvent):
    """Rare event defined by AND, OR, or an ordered token subsequence."""

    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    mode: str
    gt_prob: float
    surrogate: str
    # Independent surrogate token set + mode driving the differentiable objective.
    # Default to the indicator (tokens, mode) when not supplied in the config.
    surrogate_tokens: tuple[str, ...]
    surrogate_token_ids: tuple[int, ...]
    surrogate_mode: str
    # Display string for output-path naming and logging (e.g. " a & b" / " a | b").
    token: str

    @classmethod
    def from_config(cls, tokenizer, config: Mapping[str, Any]) -> "MultiTokenEvent":
        raw_tokens = config.get("tokens")
        if raw_tokens is None:
            raise ValueError("MultiTokenEvent config requires tokens=[<token_string>, ...].")
        mode = str(config.get("mode", "and")).strip().lower()
        if mode not in _SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode!r}. Expected one of: {_SUPPORTED_MODES}.")
        resolver = _resolve_ordered_tokens if mode == "sequence" else _resolve_dedup_tokens
        kept_tokens, token_ids = resolver(tokenizer, raw_tokens)

        surrogate = config.get("surrogate")
        surrogate = str(surrogate) if surrogate is not None else None
        if surrogate not in _SUPPORTED_SURROGATES:
            raise ValueError(
                f"Unsupported surrogate {surrogate!r}. Expected one of: {_SUPPORTED_SURROGATES}."
            )

        gt_raw = config.get("gt_prob", config.get("gt"))
        if gt_raw is None:
            raise ValueError("MultiTokenEvent config requires gt_prob=<float> (or gt=<float>).")
        gt_prob = float(gt_raw)
        if gt_prob < 0.0 or gt_prob > 1.0:
            raise ValueError(f"gt_prob must be in [0, 1], got {gt_prob}.")

        # Independent surrogate set + mode. Each defaults to the indicator's.
        surrogate_mode = str(config.get("surrogate_mode", mode)).strip().lower()
        if surrogate_mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported surrogate_mode {surrogate_mode!r}. "
                f"Expected one of: {_SUPPORTED_MODES}."
            )
        raw_surrogate_tokens = config.get("surrogate_tokens")
        if raw_surrogate_tokens is None:
            surrogate_tokens = list(kept_tokens)
            surrogate_token_ids = list(token_ids)
        else:
            surrogate_resolver = (
                _resolve_ordered_tokens if surrogate_mode == "sequence" else _resolve_dedup_tokens
            )
            surrogate_tokens, surrogate_token_ids = surrogate_resolver(
                tokenizer, raw_surrogate_tokens
            )

        joiner = {"and": " & ", "or": " | ", "sequence": " -> "}[mode]
        display = joiner.join(kept_tokens)

        return cls(
            tokens=tuple(kept_tokens),
            token_ids=tuple(token_ids),
            mode=mode,
            gt_prob=gt_prob,
            surrogate=surrogate,
            surrogate_tokens=tuple(surrogate_tokens),
            surrogate_token_ids=tuple(surrogate_token_ids),
            surrogate_mode=surrogate_mode,
            token=display,
        )

    def compute_indicator(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        if generated_tokens.ndim != 2:
            raise ValueError(
                f"generated_tokens must have shape [B, K], got {tuple(generated_tokens.shape)}"
            )
        if self.mode == "sequence":
            progress = torch.zeros(
                generated_tokens.shape[0], device=generated_tokens.device, dtype=torch.long
            )
            targets = torch.as_tensor(
                self.token_ids, device=generated_tokens.device, dtype=generated_tokens.dtype
            )
            for position in range(generated_tokens.shape[1]):
                active = progress < len(self.token_ids)
                expected = targets[progress.clamp_max(len(self.token_ids) - 1)]
                progress = progress + (active & (generated_tokens[:, position] == expected)).long()
            return progress == len(self.token_ids)

        # Per-token "appears at least once" over the K positions: [n_tokens, B].
        per_token = torch.stack(
            [(generated_tokens == tid).any(dim=-1) for tid in self.token_ids], dim=0
        )
        if self.mode == "and":
            return per_token.all(dim=0)  # co-occurrence
        return per_token.any(dim=0)  # union

    def compute_surrogate_loss(
        self,
        all_logits: torch.Tensor,
        prefix_len: int,
        generated_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.surrogate != "sum":
            raise ValueError(f"Unsupported surrogate: {self.surrogate!r}")

        generated = slice_generated_logits(all_logits, prefix_len)  # [B, K, V]
        batch_size = generated.shape[0]

        log_probs = F.log_softmax(generated, dim=-1)  # [B, K, V]
        # Surrogate optimizes the (independent) surrogate token set + mode.
        idx = torch.as_tensor(
            self.surrogate_token_ids, device=generated.device, dtype=torch.long
        )
        rare_log_probs = log_probs.index_select(dim=-1, index=idx)  # [B, K, n_surrogate]

        if self.surrogate_mode == "sequence":
            if generated_tokens is None:
                raise ValueError("sequence surrogate requires generated_tokens.")
            if generated_tokens.shape[:2] != generated.shape[:2]:
                raise ValueError(
                    "generated_tokens must match generated logits in batch and length; "
                    f"got {tuple(generated_tokens.shape)} vs {tuple(generated.shape[:2])}."
                )
            progress = torch.zeros(batch_size, device=generated.device, dtype=torch.long)
            targets = torch.as_tensor(
                self.surrogate_token_ids, device=generated.device, dtype=torch.long
            )
            per_position = []
            for position in range(generated.shape[1]):
                active = progress < len(self.surrogate_token_ids)
                target_ids = targets[progress.clamp_max(len(self.surrogate_token_ids) - 1)]
                selected = log_probs[:, position, :].gather(1, target_ids.unsqueeze(1)).squeeze(1)
                per_position.append(selected * active.to(selected.dtype))
                emitted = generated_tokens[:, position].to(device=generated.device)
                progress = progress + (active & (emitted == target_ids)).long()
            # Average over the full length; completed sequences contribute zero.
            loss_values = -torch.stack(per_position, dim=1).mean(dim=1)
        elif self.surrogate_mode == "and":
            # -mean_k sum_i log p_k^i  (== sum over tokens of single-token sum loss).
            loss_values = -rare_log_probs.sum(dim=-1).mean(dim=-1)  # [B]
        else:  # "or"
            # Sum the probabilities of distinct target tokens at each position.
            log_union_mass = torch.logsumexp(rare_log_probs, dim=-1)  # [B, K]
            loss_values = -log_union_mass.mean(dim=-1)  # [B]

        return assert_per_sample_shape(loss_values, batch_size)
