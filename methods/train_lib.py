"""Factories, registry, and CLI helpers for events and proposals."""

import argparse
import inspect


EVENT_TYPE_TOKEN = "token"
EVENT_TYPE_MULTI_TOKEN = "multi_token"
EVENT_TYPE_BOW_THRESHOLD = "bow_threshold"
SUPPORTED_EVENT_TYPES = (EVENT_TYPE_TOKEN, EVENT_TYPE_MULTI_TOKEN, EVENT_TYPE_BOW_THRESHOLD)


def supported_event_types() -> tuple[str, ...]:
    return SUPPORTED_EVENT_TYPES


def is_token_event(event_type: str) -> bool:
    return event_type == EVENT_TYPE_TOKEN


def is_multi_token_event(event_type: str) -> bool:
    return event_type == EVENT_TYPE_MULTI_TOKEN


def is_bow_threshold_event(event_type: str) -> bool:
    return event_type == EVENT_TYPE_BOW_THRESHOLD


def _proposal_registry() -> dict[str, type]:
    """Map proposal type → class. Single source of truth used by build_proposal,
    add_proposal_args, and supported_proposal_types."""
    from proposals.iu import IUProposal
    from proposals.iu_steering import IUActivationProposal, IULogitProposal
    from proposals.ce_steering import CEActivationProposal, CELogitProposal
    from proposals.ce_mle import CEMLEActivationProposal, CEMLELogitProposal
    from proposals.ce_mle_lora import CEMLELoRAProposal
    return {
        "IU": IUProposal,
        "IU_ACTIVATION": IUActivationProposal,
        "IU_LOGIT": IULogitProposal,
        "CE_ACTIVATION": CEActivationProposal,
        "CE_LOGIT": CELogitProposal,
        "CE_MLE_ACTIVATION": CEMLEActivationProposal,
        "CE_MLE_LOGIT": CEMLELogitProposal,
        "CE_MLE_LORA": CEMLELoRAProposal,
    }


def supported_proposal_types() -> tuple[str, ...]:
    return tuple(_proposal_registry().keys())


def add_proposal_args(parser: argparse.ArgumentParser) -> None:
    """Let each proposal class register its own argparse group."""
    for cls in _proposal_registry().values():
        cls.add_argparse_group(parser)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true/false")


def build_rare_event(event_type: str, event_config: dict, tokenizer):
    if is_token_event(event_type):
        from events.token_event import TokenEvent

        event = TokenEvent.from_config(tokenizer=tokenizer, config=event_config)
    elif is_multi_token_event(event_type):
        from events.multi_token_event import MultiTokenEvent

        event = MultiTokenEvent.from_config(tokenizer=tokenizer, config=event_config)
    elif is_bow_threshold_event(event_type):
        from events.bow_threshold_event import BowThresholdEvent

        event = BowThresholdEvent.from_config(tokenizer=tokenizer, config=event_config)
    else:
        raise ValueError(f"Unsupported event_type: {event_type!r}.")
    _validate_rare_event_interface(event)
    return event


def build_proposal(proposal_type: str, args):
    registry = _proposal_registry()
    if proposal_type not in registry:
        raise ValueError(f"Unsupported proposal_type: {proposal_type!r}.")
    proposal = registry[proposal_type].from_args(args)
    _validate_proposal_interface(proposal)
    return proposal


def _validate_rare_event_interface(rare_event) -> None:
    required_attrs = ("surrogate", "gt_prob")
    required_methods = ("compute_indicator", "compute_surrogate_loss")
    for attr in required_attrs:
        if not hasattr(rare_event, attr):
            raise TypeError(f"Rare event is missing required attribute: {attr}")
    for method_name in required_methods:
        method = getattr(rare_event, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"Rare event is missing required method: {method_name}()")


def _validate_proposal_interface(proposal) -> None:
    required_attrs = ("tokenizer", "device", "model_source", "uses_lora", "model_class_name")
    required_methods = ("build_optimizer", "rollout_batch", "compute_logits", "update", "encode_context")
    for attr in required_attrs:
        if not hasattr(proposal, attr):
            raise TypeError(f"Proposal is missing required attribute: {attr}")
    for method_name in required_methods:
        method = getattr(proposal, method_name, None)
        if method is None or not callable(method):
            raise TypeError(f"Proposal is missing required method: {method_name}()")
    required_update_args = ("main_loss_per_sample", "reg_loss", "lambda_value")
    param_names = tuple(inspect.signature(proposal.update).parameters.keys())
    if param_names[: len(required_update_args)] != required_update_args:
        raise TypeError(
            "Proposal.update() must start with positional args "
            f"{required_update_args}; got {param_names}."
        )
    allowed_optional_update_args = (
        "log_importance_weights", "sampled_ids", "prefix_len",
        "score_threshold", "original_logits", "elite_logits_provider",
    )
    extras = param_names[len(required_update_args):]
    unexpected = [p for p in extras if p not in allowed_optional_update_args]
    if unexpected:
        raise TypeError(
            f"Proposal.update() has unsupported extra args {unexpected}; "
            f"only {list(required_update_args)} (plus optional "
            f"{list(allowed_optional_update_args)}) are allowed."
        )
