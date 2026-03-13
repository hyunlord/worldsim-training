#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import read_jsonl, resolve_path, write_jsonl


CURRICULUM_STAGES: dict[int, set[str]] = {
    1: {"E", "F", "I", "J"},
    2: {"A", "B", "K", "L"},
    3: {"C", "G"},
    4: {"H", "M", "N", "NEG", "GEN"},
}


def curriculum_order(rows: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    ordered: list[dict] = []
    consumed_ids: set[int] = set()

    for stage_num in sorted(CURRICULUM_STAGES):
        stage_tasks = CURRICULUM_STAGES[stage_num]
        by_task: dict[str, list[dict]] = defaultdict(list)
        for index, row in enumerate(rows):
            if row.get("task") in stage_tasks:
                by_task[str(row["task"])].append((index, row))
        for task in by_task:
            rng.shuffle(by_task[task])

        while by_task:
            exhausted: list[str] = []
            for task in sorted(by_task):
                if not by_task[task]:
                    exhausted.append(task)
                    continue
                original_index, row = by_task[task].pop(0)
                consumed_ids.add(original_index)
                ordered.append(row)
                if not by_task[task]:
                    exhausted.append(task)
            for task in exhausted:
                by_task.pop(task, None)

    ordered.extend(row for index, row in enumerate(rows) if index not in consumed_ids)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorder a training dataset into a 4-stage curriculum.")
    parser.add_argument("input_path")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    input_path = resolve_path(repo_root, args.input_path)
    output_path = resolve_path(repo_root, args.output) if args.output else input_path.with_name(f"{input_path.stem}_curriculum.jsonl")
    rows = read_jsonl(input_path)
    ordered = curriculum_order(rows, seed=args.seed)
    write_jsonl(output_path, ordered)
    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "rows": len(ordered),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
