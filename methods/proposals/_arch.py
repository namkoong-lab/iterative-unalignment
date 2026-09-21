"""Locate transformer blocks and hidden/vocab sizes for GPT-2 and Gemma."""

import torch


_BLOCK_CANDIDATES = [
    ("transformer", "h"),
    ("model", "layers"),
    ("gpt_neox", "layers"),
    ("transformer", "blocks"),
]


def find_transformer_blocks(model: torch.nn.Module) -> list:
    for outer, inner in _BLOCK_CANDIDATES:
        outer_module = getattr(model, outer, None)
        if outer_module is None:
            continue
        inner_module = getattr(outer_module, inner, None)
        if inner_module is None:
            continue
        return list(inner_module)
    raise ValueError(
        "Could not locate transformer blocks for "
        f"{type(model).__name__}; extend _BLOCK_CANDIDATES for this arch."
    )


def hidden_size_of(model: torch.nn.Module) -> int:
    config = model.config
    size = getattr(config, "n_embd", None) or getattr(config, "hidden_size", None)
    if size is None:
        raise ValueError(
            f"Could not determine hidden size on {type(config).__name__}; "
            "expected n_embd or hidden_size."
        )
    return int(size)


def vocab_size_of(model: torch.nn.Module) -> int:
    config = model.config
    size = getattr(config, "vocab_size", None)
    if size is None:
        raise ValueError(
            f"Could not determine vocab size on {type(config).__name__}; "
            "expected vocab_size."
        )
    return int(size)
