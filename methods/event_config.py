"""JSON config parsing shared by train.py and runner.py."""

import json


def parse_event_config_json(config_json: str) -> dict:
    if not config_json.strip():
        raise ValueError("event_config_json cannot be empty.")
    try:
        parsed = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for event_config_json: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("event_config_json must decode to a JSON object/dict.")
    return parsed
