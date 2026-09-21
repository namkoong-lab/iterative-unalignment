#!/usr/bin/env python3
"""Estimate event probabilities with annealed TPS and MBAR."""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

from estimators.dorman import TPSConfig, estimate
from event_config import parse_event_config_json
from train_lib import build_rare_event, supported_event_types


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context")
    parser.add_argument("k", type=int)
    parser.add_argument("--event_type", choices=supported_event_types(), default="token")
    parser.add_argument("--event_config_json", required=True)
    parser.add_argument("--model_id", default="gpt2")
    parser.add_argument("--tokenizer_id", help="Optional separate tokenizer, e.g. for TinyStories")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable KV caching for suffix generation")
    parser.add_argument("--score_mode", choices=("auto", "event_score", "base_surrogate"), default="auto")
    defaults = TPSConfig()
    parser.add_argument("--bias_schedules_json", default=json.dumps(defaults.schedules),
                        help="Annealing schedules for p_b proportional to p exp(b*score); positive b favors high scores")
    parser.add_argument("--chains", type=int, default=defaults.chains)
    parser.add_argument("--steps_per_bias", type=int, default=defaults.steps_per_bias)
    parser.add_argument("--direct_samples", type=int, default=defaults.direct_samples)
    parser.add_argument("--direct_batch_size", type=int, default=defaults.direct_batch_size)
    parser.add_argument("--burnin_fraction", type=float, default=0.1)
    parser.add_argument("--gr_threshold", type=float, default=1.1)
    parser.add_argument("--filter_gr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=1, help="Number of independent complete TPS+MBAR runs")
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--output_path", type=Path, default=Path("runs/dorman"))
    parser.add_argument("--save_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=200,
                        help="Print progress every this many TPS steps or direct-sampling batches")
    args = parser.parse_args(argv)
    if args.k <= 0 or args.repeats <= 0 or args.seed < 0 or args.log_every <= 0:
        parser.error("k/repeats/log_every must be positive and seed must be nonnegative")
    return args


def json_safe(value):
    """Convert nonfinite values to JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    args = parse_args(argv)
    event_config = parse_event_config_json(args.event_config_json)
    schedules = json.loads(args.bias_schedules_json)
    config = TPSConfig(
        schedules=tuple(tuple(float(b) for b in schedule) for schedule in schedules),
        chains=args.chains, steps_per_bias=args.steps_per_bias,
        direct_samples=args.direct_samples, direct_batch_size=args.direct_batch_size,
        burnin_fraction=args.burnin_fraction,
        gr_threshold=args.gr_threshold, filter_gr=args.filter_gr,
    )
    import pymbar
    import torch
    from estimators.lm_backend import LanguageModelBackend
    from model_loading import _load_tokenizer, _resolve_device, load_reference_model
    from proposals._base import encode_prefix

    device = _resolve_device(args.device)
    tokenizer = _load_tokenizer(args.tokenizer_id or args.model_id)
    event = build_rare_event(args.event_type, event_config, tokenizer)
    if args.use_chat_template:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.context}], tokenize=True,
            add_generation_prompt=True, return_tensors="pt",
        ).to(device)
    else:
        prefix = encode_prefix(tokenizer, args.context, device)
    model = load_reference_model(args, device=device)
    backend = LanguageModelBackend(model, prefix, event, args.k, args.score_mode, args.use_cache)
    metadata = {
        "estimator_type": "DORMAN_TPS_MBAR", "model_id": args.model_id,
        "tokenizer_id": args.tokenizer_id or args.model_id,
        "context": args.context, "k": args.k, "event_type": args.event_type,
        "event_config": event_config, "gt_prob": float(event.gt_prob),
        "score_mode": backend.score_mode, "use_cache": args.use_cache,
        "use_chat_template": args.use_chat_template,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "config": asdict(config), "seed": args.seed, "repeats": args.repeats,
        "pymbar_version": pymbar.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:16]
    directory = args.output_path / f"{args.event_type}_{identity}"
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "estimates.json"
    payload = {"schema_version": 1, "metadata": metadata, "replications": []}

    def save():
        temporary = directory / "estimates.json.tmp"
        temporary.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
        temporary.replace(result_path)

    save()

    print(f"Dorman TPS + MBAR: output={directory}", flush=True)
    seeds = np.random.SeedSequence(args.seed).spawn(args.repeats)
    for index, sequence in enumerate(seeds):
        seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        torch.manual_seed(seed)
        print(f"Replication {index + 1}/{args.repeats}, seed={seed}", flush=True)
        try:
            result, traces = estimate(
                backend, config, seed, progress=lambda status: print(status, flush=True), log_every=args.log_every,
            )
        except BaseException as exc:
            payload["replications"].append({
                "replication": index, "seed": seed, "estimate": None,
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            })
            save()
            raise
        result["replication"] = index
        if args.save_traces:
            trace_path = directory / f"traces_{index:04d}.npz"
            np.savez_compressed(trace_path, **traces)
            result["traces_file"] = trace_path.name
        payload["replications"].append(result)
        save()
        print(f"Replication {index + 1}/{args.repeats}: "
              f"estimate={result['estimate']} status={result['status']} "
              f"cost={result['total_seconds']:.3f}s tokens={result['generated_tokens']}", flush=True)
    print(f"Saved {result_path}")


if __name__ == "__main__":
    main()
