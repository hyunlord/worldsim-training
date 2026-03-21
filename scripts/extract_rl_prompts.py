#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import read_jsonl
from scripts.reward_functions import parse_situation_from_prompt, parse_tci_from_prompt


def extract_rl_prompts(
    *,
    input_path: Path,
    output_path: Path,
    tasks: set[str] | None = None,
) -> int:
    rows = read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            task = str(row.get("task", ""))
            if tasks and task not in tasks:
                continue
            system = messages[0].get("content", "")
            user_prompt = messages[1].get("content", "")
            payload = {
                "prompt": user_prompt,
                "system": system,
                "task": task,
                "tci": parse_tci_from_prompt(user_prompt),
                "situation_id": parse_situation_from_prompt(user_prompt),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract RL prompts from training-format WorldSim data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_filter = {task.strip().upper() for task in args.tasks.split(",") if task.strip()} or None
    count = extract_rl_prompts(
        input_path=Path(args.input),
        output_path=Path(args.output),
        tasks=task_filter,
    )
    print(json.dumps({"written": count, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
