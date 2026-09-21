"""Model / tokenizer / LoRA / device loading for the trainable proposal and frozen reference."""

import argparse
import os
import sys
from typing import Optional

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    PeftModel = None


def add_model_loading_args(parser):
    parser.add_argument("--model_id", type=str, default="gpt2", help="Base model id")
    parser.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LoRA for efficient fine-tuning (default: enabled). Use --no-use_lora to disable.",
    )
    parser.add_argument(
        "--load_proposal_from",
        type=str,
        default=None,
        help=(
            "If set, load the trainable proposal's weights + tokenizer from this directory "
            "(a previously saved trained_model) instead of --model_id. The frozen reference "
            "model still loads from --model_id. Used for eval-only reuse of a trained proposal."
        ),
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to place the model on",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Format context as a chat user turn (tokenizer.apply_chat_template) before rollouts. "
            "Use for instruction-tuned models (e.g. google/gemma-2b-it)."
        ),
    )


def resolve_model_source(args) -> str:
    return args.model_id


def _load_tokenizer(source: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(source)
    except (TypeError, ValueError) as e:
        if "NoneType" in str(e) and "gpt2" in source.lower():
            vocab_path = hf_hub_download(source, "vocab.json")
            merges_path = hf_hub_download(source, "merges.txt")
            tokenizer = GPT2Tokenizer(vocab_file=vocab_path, merges_file=merges_path)
        else:
            raise
    return tokenizer


def _load_causal_lm(source: str):
    try:
        return AutoModelForCausalLM.from_pretrained(source)
    except OSError as e:
        if "no file named" in str(e) and "found in directory" in str(e):
            if os.path.isdir(source):
                print(f"Warning: local directory '{source}' exists and may shadow hub id", file=sys.stderr)
            model_path = snapshot_download(source)
            return AutoModelForCausalLM.from_pretrained(model_path)
        raise


def _infer_lora_target_modules(model) -> list[str]:
    module_names = {name for name, _ in model.named_modules()}
    candidate_groups = [
        ["q_proj", "k_proj", "v_proj", "o_proj"],
        ["c_attn", "c_proj"],
        ["query_key_value", "dense"],
        ["Wqkv"],
    ]
    for group in candidate_groups:
        group_matches = [m for m in group if any(name.endswith(m) for name in module_names)]
        if group_matches:
            return group_matches
    raise ValueError(
        "Could not infer LoRA target modules. Expected one of: "
        "q_proj/k_proj/v_proj/o_proj, c_attn/c_proj, query_key_value/dense, or Wqkv."
    )


def _setup_lora_model(model, args):
    if not args.use_lora:
        return model
    if not LORA_AVAILABLE:
        raise ImportError("LoRA requested but 'peft' is not installed. Install with: pip install peft")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_infer_lora_target_modules(model),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _configure_trainable_params(model, use_lora: bool):
    model.train()
    if use_lora:
        if PeftModel is not None and isinstance(model, PeftModel):
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable_params == 0:
            raise ValueError("No trainable parameters found. Check LoRA configuration.")
    else:
        for param in model.parameters():
            param.requires_grad = True


def _resolve_device(device_flag: str) -> torch.device:
    if device_flag == "cpu":
        return torch.device("cpu")
    if device_flag == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trainable_components(
    args,
) -> tuple[torch.nn.Module, AutoTokenizer, torch.device, str, bool]:
    # Saved proposal weights can be loaded independently of the reference model.
    source = getattr(args, "load_proposal_from", None) or resolve_model_source(args)
    device = _resolve_device(args.device)

    tokenizer = _load_tokenizer(source)
    model = _load_causal_lm(source)
    model = _setup_lora_model(model, args)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    _configure_trainable_params(model, args.use_lora)

    return model, tokenizer, device, source, bool(args.use_lora)


def load_reference_model(args, device: Optional[torch.device] = None):
    source = resolve_model_source(args)
    model = _load_causal_lm(source)
    ref_device = device if device is not None else _resolve_device(args.device)
    model.to(ref_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
