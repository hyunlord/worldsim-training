#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, read_jsonl, resolve_path, write_jsonl
from scripts.curriculum_order_v3 import curriculum_order_v3

KOREAN_V2_TASKS = {"A", "B", "C", "G"}


def _canonical_output(value: object) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _content_hash(row: dict) -> str:
    payload = {
        "task": row.get("task", ""),
        "output": _canonical_output(row.get("output", "")),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _tag_rows(rows: list[dict], source_name: str) -> list[dict]:
    return [
        {
            **row,
            "_source": source_name,
            "merge_source_batch": row.get("merge_source_batch") or source_name,
        }
        for row in rows
    ]


def _filter_tasks(rows: list[dict], allowed_tasks: set[str]) -> list[dict]:
    return [row for row in rows if str(row.get("task", "")) in allowed_tasks]


def _stratified_split(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task", "unknown"))].append(row)

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for task in sorted(by_task):
        task_rows = list(by_task[task])
        rng.shuffle(task_rows)
        if len(task_rows) == 1:
            train_rows.extend(task_rows)
            continue
        dev_count = max(1, int(len(task_rows) * dev_ratio))
        if dev_count >= len(task_rows):
            dev_count = len(task_rows) - 1
        dev_rows.extend(task_rows[:dev_count])
        train_rows.extend(task_rows[dev_count:])
    return train_rows, dev_rows


def _dedupe_rows(train_rows: list[dict], dev_rows: list[dict]) -> tuple[list[dict], list[dict], int]:
    seen_hashes: set[str] = set()
    deduped_train: list[dict] = []
    deduped_dev: list[dict] = []
    removed = 0

    for row in train_rows:
        content_hash = _content_hash(row)
        if content_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(content_hash)
        deduped_train.append(row)

    for row in dev_rows:
        content_hash = _content_hash(row)
        if content_hash in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(content_hash)
        deduped_dev.append(row)

    return deduped_train, deduped_dev, removed


def assemble_v3_dataset(
    repo_root: Path,
    *,
    dev_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Path | None = None,
    dataset_id: str = "worldsim-v3-mix",
    v2_train_path: Path | None = None,
    v2_dev_path: Path | None = None,
    logic_path: Path | None = None,
    new_tasks_path: Path | None = None,
    negative_path: Path | None = None,
    general_path: Path | None = None,
) -> AttrDict:
    v2_train_source = v2_train_path or repo_root / "data/final/worldsim-v2-mix/train.jsonl"
    v2_dev_source = v2_dev_path or repo_root / "data/final/worldsim-v2-mix/dev.jsonl"
    logic_source = logic_path or repo_root / "data/validated/batch_v3_01_english_logic/passed.jsonl"
    new_tasks_source = new_tasks_path or repo_root / "data/validated/batch_v3_02_new_tasks/passed.jsonl"
    negative_source = negative_path or repo_root / "data/samples/negative_examples.jsonl"
    general_source = general_path or repo_root / "data/samples/general_korean.jsonl"

    v2_train = _tag_rows(_filter_tasks(read_jsonl(v2_train_source), KOREAN_V2_TASKS), "v2_train")
    v2_dev = _tag_rows(_filter_tasks(read_jsonl(v2_dev_source), KOREAN_V2_TASKS), "v2_dev")
    logic_rows = _tag_rows(read_jsonl(logic_source), "batch_v3_01")
    new_task_rows = _tag_rows(read_jsonl(new_tasks_source), "batch_v3_02")
    negative_rows = _tag_rows(read_jsonl(negative_source), "negative")
    general_rows = _tag_rows(read_jsonl(general_source), "general")

    logic_train, logic_dev = _stratified_split(logic_rows, dev_ratio, seed)
    new_train, new_dev = _stratified_split(new_task_rows, dev_ratio, seed)

    train_rows = [*v2_train, *logic_train, *new_train, *negative_rows, *general_rows]
    dev_rows = [*v2_dev, *logic_dev, *new_dev]
    deduped_train, deduped_dev, duplicates_removed = _dedupe_rows(train_rows, dev_rows)
    curriculum_train = curriculum_order_v3(deduped_train, seed=seed)

    out_dir = output_dir or (repo_root / "data" / "final" / dataset_id)
    ensure_directory(out_dir)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    curriculum_path = out_dir / "train_curriculum.jsonl"
    manifest_path = out_dir / "merge_manifest.json"

    write_jsonl(train_path, deduped_train)
    write_jsonl(dev_path, deduped_dev)
    write_jsonl(curriculum_path, curriculum_train)

    manifest = {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "dev_ratio": dev_ratio,
        "seed": seed,
        "sources": {
            "v2_train_filtered": len(v2_train),
            "v2_dev_filtered": len(v2_dev),
            "logic_total": len(logic_rows),
            "new_tasks_total": len(new_task_rows),
            "negatives": len(negative_rows),
            "general": len(general_rows),
        },
        "deduplication": {"removed": duplicates_removed},
        "output": {
            "train": len(deduped_train),
            "dev": len(deduped_dev),
            "train_curriculum": len(curriculum_train),
            "total": len(deduped_train) + len(deduped_dev),
        },
        "task_counts": {
            "train": dict(sorted(Counter(str(row.get("task", "unknown")) for row in deduped_train).items())),
            "dev": dict(sorted(Counter(str(row.get("task", "unknown")) for row in deduped_dev).items())),
            "train_curriculum": dict(sorted(Counter(str(row.get("task", "unknown")) for row in curriculum_train).items())),
        },
        "source_counts": {
            "train": dict(sorted(Counter(str(row.get("_source", "unknown")) for row in deduped_train).items())),
            "dev": dict(sorted(Counter(str(row.get("_source", "unknown")) for row in deduped_dev).items())),
        },
        "paths": {
            "train": str(train_path),
            "dev": str(dev_path),
            "train_curriculum": str(curriculum_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return AttrDict(
        train_path=train_path,
        dev_path=dev_path,
        curriculum_path=curriculum_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the mixed WorldSim v3 dataset from v2/v3 sources.")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/final/worldsim-v3-mix")
    parser.add_argument("--dataset-id", default="worldsim-v3-mix")
    parser.add_argument("--v2-train", default="data/final/worldsim-v2-mix/train.jsonl")
    parser.add_argument("--v2-dev", default="data/final/worldsim-v2-mix/dev.jsonl")
    parser.add_argument("--logic-passed", default="data/validated/batch_v3_01_english_logic/passed.jsonl")
    parser.add_argument("--new-tasks-passed", default="data/validated/batch_v3_02_new_tasks/passed.jsonl")
    parser.add_argument("--negative", default="data/samples/negative_examples.jsonl")
    parser.add_argument("--general", default="data/samples/general_korean.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = assemble_v3_dataset(
        repo_root,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        output_dir=resolve_path(repo_root, args.output_dir),
        dataset_id=args.dataset_id,
        v2_train_path=resolve_path(repo_root, args.v2_train),
        v2_dev_path=resolve_path(repo_root, args.v2_dev),
        logic_path=resolve_path(repo_root, args.logic_passed),
        new_tasks_path=resolve_path(repo_root, args.new_tasks_passed),
        negative_path=resolve_path(repo_root, args.negative),
        general_path=resolve_path(repo_root, args.general),
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
