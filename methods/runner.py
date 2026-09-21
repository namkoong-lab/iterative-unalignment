#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

from event_config import parse_event_config_json
from runner_lib import (
    get_token_gt_probs,
    parse_token_gt_pairs_file,
)
from train_lib import is_token_event, supported_event_types


def _parse_tokens(raw_tokens: list[str]) -> list[str]:
    tokens: list[str] = []
    for raw in raw_tokens:
        for token in raw.split(","):
            if token:
                tokens.append(token)
    if not tokens:
        raise ValueError("At least one non-empty token is required when --tokens is provided.")
    return tokens


def _parse_float_values(raw_values: list[str], arg_name: str) -> list[float]:
    values: list[float] = []
    for raw in raw_values:
        for token in raw.split(","):
            cleaned = token.strip()
            if not cleaned:
                continue
            try:
                values.append(float(cleaned))
            except ValueError as exc:
                raise ValueError(f"Invalid float value for {arg_name}: {cleaned!r}") from exc
    if not values:
        raise ValueError(f"At least one valid float is required when {arg_name} is provided.")
    return values


def _build_parser() -> argparse.ArgumentParser:
    event_types = supported_event_types()
    parser = argparse.ArgumentParser(
        description=(
            "General runner that forwards training args to train.py and, for token events, "
            "can run one training job per token."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Token runs: two ways to supply tokens (use one, not both):\n"
            "  (1) --tokens [...]  Optional --token_gt_file: looks up gt_prob in a "
            "reference table (runner_lib.get_token_gt_probs). Without it, gt_prob=0.\n"
            "  (2) --token_pairs_file PATH: each line is JSON token, ' : ', float P; "
            "no GT table lookup (runner_lib.parse_token_gt_pairs_file).\n"
            "For non-token event_type, omit both; a single job uses event_config_json as-is."
        ),
    )
    parser.add_argument("context", type=str, help="Prompt/context text")
    parser.add_argument("k", type=int, help="Number of generated tokens per rollout")
    parser.add_argument(
        "--event_type",
        type=str,
        default=event_types[0],
        choices=event_types,
        help="Rare event type. Token list expansion is enabled for token events only.",
    )
    parser.add_argument(
        "--event_config_json",
        type=str,
        required=True,
        help="JSON object string for event config; for token events, --tokens overrides config.token per run.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        default=None,
        help="Token list for token events (GT from table unless --token_pairs_file is used).",
    )
    parser.add_argument(
        "--token_pairs_file",
        type=str,
        default=None,
        help=(
            "Token events only: .txt with one pair per line (JSON token, ' : ', float P). "
            "Mutually exclusive with --tokens. Ignores --token_gt_file."
        ),
    )
    parser.add_argument(
        "--ess_target",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional ESS target list to iterate over. "
            "Accepts space-separated and/or comma-separated float values."
        ),
    )
    parser.add_argument(
        "--init_lambda",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional lambda-zero list to iterate over. "
            "Accepts space-separated and/or comma-separated float values."
        ),
    )
    parser.add_argument(
        "--lambda_floor",
        type=float,
        default=0.1,
        help="Fixed lambda-floor value used for all runs.",
    )
    parser.add_argument(
        "--token_gt_file",
        type=str,
        default=None,
        help=(
            "With --tokens only: optional full GT .txt for lookup (Prompt / L / Per-token table). "
            "If omitted, gt_prob defaults to 0; supply a reference value for accuracy comparisons. "
            "Not used with --token_pairs_file."
        ),
    )
    return parser


def _run_single_train(
    train_script: Path,
    context: str,
    k: int,
    event_type: str,
    event_config: dict,
    passthrough_args: list[str],
    override_args: list[str],
) -> None:
    command = [
        sys.executable,
        str(train_script),
        context,
        str(k),
        "--event_type",
        event_type,
        "--event_config_json",
        json.dumps(event_config),
        *passthrough_args,
        *override_args,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = _build_parser()
    args, passthrough_args = parser.parse_known_args()
    base_event_config = parse_event_config_json(args.event_config_json)
    ess_target_values = (
        _parse_float_values(args.ess_target, "--ess_target")
        if args.ess_target is not None
        else [None]
    )
    init_lambda_values = (
        _parse_float_values(args.init_lambda, "--init_lambda")
        if args.init_lambda is not None
        else [None]
    )
    lambda_floor = float(args.lambda_floor)

    train_script = Path(__file__).with_name("train.py")
    event_type = str(args.event_type)

    if args.token_pairs_file is not None and args.tokens is not None:
        raise ValueError("Use either --token_pairs_file or --tokens, not both.")
    if args.token_pairs_file is not None and args.token_gt_file is not None:
        # Pair files already include reference probabilities.
        raise ValueError("--token_gt_file is not used with --token_pairs_file; remove one of them.")

    run_event_configs: list[dict] = []
    pairs_file = args.token_pairs_file
    if pairs_file is not None:
        if not is_token_event(event_type):
            raise ValueError(
                f"--token_pairs_file is only supported for token events. Got event_type={event_type!r}."
            )
        pairs = parse_token_gt_pairs_file(pairs_file)
        for token, gt_prob in pairs:
            run_event_config = dict(base_event_config)
            run_event_config["token"] = token
            run_event_config["gt_prob"] = gt_prob
            run_event_configs.append(run_event_config)
    elif args.tokens is not None:
        tokens = _parse_tokens(args.tokens)
        if not is_token_event(event_type):
            raise ValueError(
                f"--tokens is only supported for token events. Got event_type={event_type!r}."
            )

        if args.token_gt_file:
            token_gt_probs = get_token_gt_probs(
                tokens=tokens,
                prefix=args.context,
                autoregressive_length=args.k,
                gt_file_path=args.token_gt_file,
            )
        else:
            token_gt_probs = [0.0] * len(tokens)
        for token, gt_prob in zip(tokens, token_gt_probs):
            run_event_config = dict(base_event_config)
            run_event_config["token"] = token
            run_event_config["gt_prob"] = gt_prob
            run_event_configs.append(run_event_config)
    else:
        run_event_configs = [base_event_config]

    for run_event_config in run_event_configs:
        token = run_event_config.get("token")
        gt_prob = run_event_config.get("gt_prob")

        for ess_target in ess_target_values:
            for init_lambda in init_lambda_values:
                override_args: list[str] = []
                if ess_target is not None:
                    override_args.extend(["--ess_target", str(ess_target)])
                if init_lambda is not None:
                    override_args.extend(["--init_lambda", str(init_lambda)])
                override_args.extend(["--lambda_floor", str(lambda_floor)])

                run_desc = f"event_config={run_event_config}"
                if token is not None and gt_prob is not None:
                    run_desc = f"token={token!r} gt_prob={float(gt_prob):.12f}"
                    if pairs_file is not None:
                        run_desc += f" (pairs_file={pairs_file})"
                print(
                    "[runner] Starting run for "
                    f"{run_desc}, "
                    f"ess_target={ess_target}, init_lambda={init_lambda}, lambda_floor={lambda_floor}"
                )
                _run_single_train(
                    train_script=train_script,
                    context=args.context,
                    k=args.k,
                    event_type=event_type,
                    event_config=run_event_config,
                    passthrough_args=passthrough_args,
                    override_args=override_args,
                )


if __name__ == "__main__":
    main()
