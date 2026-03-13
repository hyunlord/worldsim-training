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
    return hashlib.sha256(_canonical_output(row.get("output", "")).encode("utf-8")).hexdigest()


def _tag_rows(rows: list[dict], source_name: str) -> list[dict]:
    tagged: list[dict] = []
    for row in rows:
        tagged.append(
            {
                **row,
                "_source": source_name,
                "merge_source_batch": row.get("merge_source_batch") or source_name,
            }
        )
    return tagged


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
        dev_count = max(1, int(len(task_rows) * dev_ratio))
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


def assemble_v2_dataset(
    repo_root: Path,
    *,
    dev_ratio: float = 0.1,
    seed: int = 42,
    output_dir: Path | None = None,
    dataset_id: str = "worldsim-v2-mix",
) -> AttrDict:
    v1_train = _tag_rows(read_jsonl(repo_root / "data/final/worldsim-v31-mix-v1/train.jsonl"), "v1_train")
    v1_dev = _tag_rows(read_jsonl(repo_root / "data/final/worldsim-v31-mix-v1/dev.jsonl"), "v1_dev")
    batch1_rows = _tag_rows(read_jsonl(repo_root / "data/validated/batch_v2_01_tasks_in/passed.jsonl"), "batch_v2_01")
    batch2_rows = _tag_rows(read_jsonl(repo_root / "data/validated/batch_v2_02_task_g_fix/passed.jsonl"), "batch_v2_02")
    negative_rows = _tag_rows(read_jsonl(repo_root / "data/samples/negative_examples.jsonl"), "negative")
    general_rows = _tag_rows(read_jsonl(repo_root / "data/samples/general_korean.jsonl"), "general")

    batch1_train, batch1_dev = _stratified_split(batch1_rows, dev_ratio, seed)
    batch2_train, batch2_dev = _stratified_split(batch2_rows, dev_ratio, seed)

    train_rows = [*v1_train, *batch1_train, *batch2_train, *negative_rows, *general_rows]
    dev_rows = [*v1_dev, *batch1_dev, *batch2_dev]
    deduped_train, deduped_dev, duplicates_removed = _dedupe_rows(train_rows, dev_rows)

    out_dir = output_dir or (repo_root / "data" / "final" / dataset_id)
    ensure_directory(out_dir)
    train_path = out_dir / "train.jsonl"
    dev_path = out_dir / "dev.jsonl"
    manifest_path = out_dir / "merge_manifest.json"

    write_jsonl(train_path, deduped_train)
    write_jsonl(dev_path, deduped_dev)

    manifest = {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "dev_ratio": dev_ratio,
        "seed": seed,
        "sources": {
            "v1_train": len(v1_train),
            "v1_dev": len(v1_dev),
            "batch1_total": len(batch1_rows),
            "batch2_total": len(batch2_rows),
            "negatives": len(negative_rows),
            "general": len(general_rows),
        },
        "deduplication": {"removed": duplicates_removed},
        "output": {
            "train": len(deduped_train),
            "dev": len(deduped_dev),
            "total": len(deduped_train) + len(deduped_dev),
        },
        "task_counts": {
            "train": dict(sorted(Counter(str(row.get("task", "unknown")) for row in deduped_train).items())),
            "dev": dict(sorted(Counter(str(row.get("task", "unknown")) for row in deduped_dev).items())),
        },
        "source_counts": {
            "train": dict(sorted(Counter(str(row.get("_source", "unknown")) for row in deduped_train).items())),
            "dev": dict(sorted(Counter(str(row.get("_source", "unknown")) for row in deduped_dev).items())),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return AttrDict(
        train_path=train_path,
        dev_path=dev_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the mixed WorldSim v2 dataset from all available sources.")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/final/worldsim-v2-mix")
    parser.add_argument("--dataset-id", default="worldsim-v2-mix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = assemble_v2_dataset(
        repo_root,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        output_dir=resolve_path(repo_root, args.output_dir),
        dataset_id=args.dataset_id,
    )
    print(json.dumps(result.manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
