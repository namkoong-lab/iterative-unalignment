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
            f"alpha_{_safe_path_part(args.alpha_reg_moment)}",
            f"ess_target_{_safe_path_part(args.ess_target)}",
            timestamp,
        )
    elif event_type == _EVENT_TYPE_MULTI_TOKEN:
        # RUN_ID-controlled sharing: the proposal (determined by the surrogate)
        # and its trained_model/ live directly in the per-run dir (output_path),
        # with no timestamp and no indicator/surrogate keying. Same output_path
        # => same trained_model, so eval-only reuse (--reuse_trained_model) finds
        # it; the caller groups same-surrogate runs under one output_path.
        # Per-indicator eval results go in subdirs via multi_token_eval_dir().
        output_dir = base_output_dir
    elif event_type == _EVENT_TYPE_BOW_THRESHOLD:
        output_dir = os.path.join(
            base_output_dir,
            event_type,
            f"prefix_length_{args.k}",
            f"token_{_safe_path_part(rare_event.token)}",
            f"surrogate_{_safe_path_part(rare_event.surrogate)}",
            f"alpha_{_safe_path_part(args.alpha_reg_moment)}",
            f"ess_target_{_safe_path_part(args.ess_target)}",
            timestamp,
        )
    else:
        raise ValueError(f"Unsupported event_type for output dir: {event_type!r}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def multi_token_eval_dir(output_dir: str, rare_event) -> str:
    """Per-indicator eval subdir under a multi_token run dir.

    Keyed by the indicator (mode + tokens) so that several indicators reusing one
    trained proposal (all sharing output_dir) write to distinct eval folders
    instead of overwriting each other.
    """
    mode = _safe_path_part(getattr(rare_event, "mode", "and"))
    tokens = _safe_path_part(getattr(rare_event, "token", "event"))
    # Many-token indicators (e.g. 100-token OR) make the joined token string blow
    # past the filesystem's per-component name limit (255 bytes). Bound the
    # segment and append a short hash of the FULL token string so distinct
    # indicators that share a truncated prefix still land in distinct folders.
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
    lines.append(
        "plots: "
        "losses.png (includes event batch % on right axis), lambda_alpha.png, "
        "ess.png, is_estimator.png"
    )

    with open(params_txt_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
