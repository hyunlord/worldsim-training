from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assistant_lengths(rows: list[dict], task: str) -> list[int]:
    lengths: list[int] = []
    for row in rows:
        if row.get("task") != task:
            continue
        messages = row.get("messages", [])
        assistant_messages = [message for message in messages if message.get("role") == "assistant"]
        if assistant_messages:
            lengths.append(len(str(assistant_messages[-1].get("content", ""))))
    return lengths


def analyze(train_file: str) -> None:
    path = Path(train_file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    try:
        from training.lib.structured_generation import TASK_MAX_NEW_TOKENS
    except ImportError:
        TASK_MAX_NEW_TOKENS = {}

    tasks = sorted({str(row.get("task", "?")) for row in rows})

    print(f"{'Task':<6} {'Count':>6}  {'Chars(min/max/mean/p95)':>28}  {'Est.Tok(max/p95)':>18}  {'max_new_tokens':>15}  {'Headroom':>8}")
    print("-" * 100)

    for task in tasks:
        lengths = sorted(_assistant_lengths(rows, task))
        if not lengths:
            continue

        count = len(lengths)
        min_len = lengths[0]
        max_len = lengths[-1]
        mean_len = sum(lengths) / count
        p95_len = lengths[min(count - 1, int(count * 0.95))]

        tok_max = max_len / 3.5
        tok_p95 = p95_len / 3.5
        max_new_tokens = TASK_MAX_NEW_TOKENS.get(task, "?")
        headroom = f"{max_new_tokens / tok_max:.1f}x" if isinstance(max_new_tokens, (int, float)) and tok_max > 0 else "?"

        chars_str = f"{min_len:>4} / {max_len:>4} / {mean_len:>5.0f} / {p95_len:>4}"
        token_str = f"{tok_max:>6.0f} / {tok_p95:>5.0f}"

        print(
            f"{task:<6} {count:>6}  {chars_str:>28}  {token_str:>18}  {str(max_new_tokens):>15}  {headroom:>8}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} <train_file.jsonl>", file=sys.stderr)
        raise SystemExit(1)
    analyze(sys.argv[1])
