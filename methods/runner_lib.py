import json
from pathlib import Path
from typing import Sequence

# Optional token/probability pair files use this separator. Pass --token_gt_file to runner.py.
TOKEN_GT_PAIRS_LINE_SEP = " : "

# Prefixes in optional full-vocabulary GT tables read by get_token_gt_probs.
GT_FILE_PROMPT_PREFIX = "Prompt:"
GT_FILE_SEQ_LENGTH_PREFIX = "L (seq_length):"
GT_FILE_TABLE_HEADER_PREFIX = "Per-token probabilities"


def get_token_gt_probs(
    tokens: Sequence[str],
    prefix: str,
    autoregressive_length: int,
    gt_file_path: str | Path,
) -> list[float]:
    """
    Return GT probabilities aligned with `tokens` order.

    Raises:
        ValueError: if prompt/sequence length does not match GT metadata.
        ValueError: if any token is not found in the GT file.
    """
    if not tokens:
        raise ValueError("tokens must be a non-empty sequence.")

    gt_path = Path(gt_file_path)
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    missing = set(tokens)
    token_to_gt: dict[str, float] = {}
    in_table = False
    gt_prompt: str | None = None
    gt_seq_length: int | None = None

    with gt_path.open("r", encoding="utf-8", errors="replace") as gt_file:
        for raw_line in gt_file:
            line = raw_line.rstrip("\n")
            if not in_table:
                if line.startswith(GT_FILE_PROMPT_PREFIX):
                    gt_prompt = line.split(":", 1)[1].strip()
                elif line.startswith(GT_FILE_SEQ_LENGTH_PREFIX):
                    gt_seq_length = int(line.split(":", 1)[1].strip())
                if line.startswith(GT_FILE_TABLE_HEADER_PREFIX):
                    if gt_prompt is None or gt_seq_length is None:
                        raise ValueError(
                            f"GT file missing prompt/seq_length metadata: {gt_path}"
                        )
                    if gt_prompt != prefix:
                        raise ValueError(
                            "GT file prompt mismatch: "
                            f"expected {gt_prompt!r}, got {prefix!r}"
                        )
                    if gt_seq_length != autoregressive_length:
                        raise ValueError(
                            "GT file seq_length mismatch: "
                            f"expected {gt_seq_length}, got {autoregressive_length}"
                        )
                    in_table = True
                continue

            if not line:
                continue

            parts = line.split("\t", 3)
            if len(parts) < 4:
                continue

            _, _, gt_prob_raw, token = parts
            if token not in missing:
                continue

            try:
                token_to_gt[token] = float(gt_prob_raw)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GT probability for token {token!r}: {gt_prob_raw!r}"
                ) from exc

            missing.remove(token)
            if not missing:
                break

    if missing:
        missing_list = ", ".join(repr(token) for token in sorted(missing))
        raise ValueError(
            f"Token(s) not found in GT file {gt_path}: {missing_list}"
        )

    return [token_to_gt[token] for token in tokens]


def parse_token_gt_pairs_file(gt_pairs_path: str | Path) -> list[tuple[str, float]]:
    """
    Read a simple two-column text file: JSON string for the token, then TOKEN_GT_PAIRS_LINE_SEP,
    then float ground-truth probability. Blank lines and lines starting with # are skipped.

    Leading spaces inside the token are preserved via JSON (e.g. ``" word"``).
    """
    path = Path(gt_pairs_path)
    if not path.is_file():
        raise FileNotFoundError(f"Token pairs file not found: {path}")

    sep = TOKEN_GT_PAIRS_LINE_SEP
    out: list[tuple[str, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, raw_line in enumerate(handle, 1):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            idx = raw_line.rfind(sep)
            if idx < 0:
                raise ValueError(
                    f"{path}:{lineno}: expected {sep!r} between JSON token and probability"
                )
            left = raw_line[:idx].strip()
            right = raw_line[idx + len(sep) :].strip()
            try:
                token = json.loads(left)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON token field: {left!r}") from exc
            if not isinstance(token, str):
                raise ValueError(
                    f"{path}:{lineno}: JSON token must be a string, got {type(token).__name__}"
                )
            try:
                prob = float(right)
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid probability: {right!r}") from exc
            out.append((token, prob))

    if not out:
        raise ValueError(f"No token/probability pairs parsed from {path}")
    return out
