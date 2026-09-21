"""Run output directory layout and run_params.txt writer."""

import hashlib
import os
from datetime import datetime

# Local copies of train_lib event-type names, so this module stays a light import.
_EVENT_TYPE_BOW_THRESHOLD = "bow_threshold"
_EVENT_TYPE_MULTI_TOKEN = "multi_token"


def _safe_path_part(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "empty"
    safe_chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        elif ch.isspace():
            safe_chars.append("_")
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("._") or "empty"


def resolve_output_dir(args, rare_event) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.abspath(str(args.output_path))
    event_type = str(args.event_type)
    if event_type == "token":
        output_dir = os.path.join(
            base_output_dir,
            event_type,
            f"prefix_length_{args.k}",
            f"token_{_safe_path_part(rare_event.token)}",
            "kl_qp_dense",
            f"ess_target_{_safe_path_part(args.ess_target)}",
            timestamp,
        )
    elif event_type == _EVENT_TYPE_MULTI_TOKEN:
        # A shared output path allows multiple indicators to reuse one proposal.
        output_dir = base_output_dir
    elif event_type == _EVENT_TYPE_BOW_THRESHOLD:
        output_dir = os.path.join(
            base_output_dir,
            event_type,
            f"prefix_length_{args.k}",
            f"token_{_safe_path_part(rare_event.token)}",
            f"surrogate_{_safe_path_part(rare_event.surrogate)}",
            "kl_qp_dense",
            f"ess_target_{_safe_path_part(args.ess_target)}",
            timestamp,
        )
    else:
        raise ValueError(f"Unsupported event_type for output dir: {event_type!r}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def multi_token_eval_dir(output_dir: str, rare_event) -> str:
    """Evaluation directory keyed by indicator mode and tokens."""
    mode = _safe_path_part(getattr(rare_event, "mode", "and"))
    tokens = _safe_path_part(getattr(rare_event, "token", "event"))
    # Shorten long path components and hash the full token string.
    _MAX_TOKEN_SEG = 80
    if len(tokens) > _MAX_TOKEN_SEG:
        digest = hashlib.sha1(tokens.encode("utf-8")).hexdigest()[:8]
        tokens = f"{tokens[:_MAX_TOKEN_SEG]}_{digest}"
    eval_dir = os.path.join(
        os.path.abspath(output_dir), "eval", f"mode_{mode}", f"tokens_{tokens}"
    )
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def write_run_params_txt(args, rare_event, output_dir: str, output_pickle_path: str) -> None:
    output_dir = os.path.abspath(output_dir)
    params_txt_path = os.path.join(output_dir, "run_params.txt")
    args_dict = vars(args)

    lines: list[str] = []
    lines.append("# Full run parameters")
    lines.append("")
    for key in sorted(args_dict.keys()):
        lines.append(f"{key}: {args_dict[key]}")

    lines.append("")
    lines.append("# Resolved rare-event fields")
    lines.append(f"resolved_event_token: {rare_event.token!r}")
    lines.append(f"resolved_event_surrogate: {rare_event.surrogate!r}")
    if hasattr(rare_event, "mode"):
        lines.append(f"resolved_event_mode: {rare_event.mode!r}")
    if hasattr(rare_event, "tokens"):
        lines.append(f"resolved_event_tokens: {list(rare_event.tokens)!r}")
    if hasattr(rare_event, "surrogate_mode"):
        lines.append(f"resolved_event_surrogate_mode: {rare_event.surrogate_mode!r}")
    if hasattr(rare_event, "surrogate_tokens"):
        lines.append(
            f"resolved_event_surrogate_tokens: {list(rare_event.surrogate_tokens)!r}"
        )
    lines.append(f"resolved_event_gt_prob: {float(rare_event.gt_prob):.12f}")
    if hasattr(rare_event, "threshold"):
        lines.append(f"resolved_event_threshold: {float(rare_event.threshold):.12f}")
    if hasattr(rare_event, "weights_npz"):
        lines.append(f"resolved_event_weights_npz: {rare_event.weights_npz!r}")

    lines.append("")
    lines.append("# Output locations")
    lines.append(f"output_path_parent: {os.path.abspath(str(args.output_path))}")
    lines.append(f"output_pickle_path: {output_pickle_path}")
    lines.append(f"output_dir: {output_dir}")

    with open(params_txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
