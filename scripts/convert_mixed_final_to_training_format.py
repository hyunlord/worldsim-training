#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, load_yaml, read_jsonl, resolve_path
from scripts.prepare_dataset import _row_to_training_example, _training_system_prompts, _validate_messages_row


SUPPORTED_TASKS = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"}


def _load_rows(path: Path, *, split: str) -> list[dict]:
    rows = read_jsonl(path)
    annotated: list[dict] = []
    for row in rows:
        annotated.append(
            {
                **row,
                "source_split": row.get("merge_source_split") or row.get("dataset_split") or split,
            }
        )
    return annotated


def _preserve_provenance(row: dict, converted: dict, *, split: str, dataset_id: str) -> dict:
    return {
        "task": converted.get("task") or row.get("task"),
        "layer": converted.get("layer") or row.get("layer"),
        "source_batch": row.get("merge_source_batch"),
        "source_split": row.get("merge_source_split") or row.get("dataset_split") or split,
        "source_dataset_id": row.get("merge_dataset_id") or dataset_id,
        "source_task": row.get("merge_source_task") or row.get("task"),
        "messages": converted["messages"],
    }


def _validate_converted_rows(rows: list[dict]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        _validate_messages_row(row)
        task = row.get("task")
        if task in SUPPORTED_TASKS:
            assistant_content = row["messages"][-1]["content"]
            if not assistant_content.strip():
                raise ValueError(f"Converted row for task {task} has empty assistant target")
            try:
                parsed = json.loads(assistant_content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Converted row for task {task} has malformed JSON assistant target") from exc
            if not isinstance(parsed, (dict, list)):
                raise ValueError(f"Converted row for task {task} has non-JSON structured assistant target")
        counts[str(task or "unknown")] += 1
    return dict(sorted(counts.items()))


def convert_mixed_final_to_training_format(
    *,
    repo_root: Path,
    input_train: Path,
    input_dev: Path,
    source_manifest: Path | None,
    output_dir: Path,
    dataset_id: str,
) -> AttrDict:
    settings = load_yaml(repo_root / "config" / "generation.yaml")
    system_prompts = _training_system_prompts(repo_root, settings)

    train_rows = _load_rows(input_train, split="train")
    dev_rows = _load_rows(input_dev, split="dev")

    converted_train = [
        _preserve_provenance(row, _row_to_training_example(row, system_prompts), split="train", dataset_id=dataset_id)
        for row in train_rows
    ]
    converted_dev = [
        _preserve_provenance(row, _row_to_training_example(row, system_prompts), split="dev", dataset_id=dataset_id)
        for row in dev_rows
    ]

    train_task_counts = _validate_converted_rows(converted_train)
    dev_task_counts = _validate_converted_rows(converted_dev)

    ensure_directory(output_dir)
    train_output = output_dir / "train_converted.jsonl"
    dev_output = output_dir / "dev_converted.jsonl"
    manifest_path = output_dir / "conversion_manifest.json"

    with train_output.open("w", encoding="utf-8") as handle:
        for row in converted_train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with dev_output.open("w", encoding="utf-8") as handle:
        for row in converted_dev:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_manifest_payload: dict[str, Any] | None = None
    if source_manifest and source_manifest.exists():
        source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))

    manifest_payload = {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "detected_training_format": "messages",
        "training_contract_source": str(repo_root / "scripts" / "prepare_dataset.py"),
        "source_files": {
            "input_train": str(input_train),
            "input_dev": str(input_dev),
            "source_manifest": str(source_manifest) if source_manifest else None,
        },
        "output_files": {
            "train_converted": str(train_output),
            "dev_converted": str(dev_output),
        },
        "counts": {
            "train_source": len(train_rows),
            "dev_source": len(dev_rows),
            "train_converted": len(converted_train),
            "dev_converted": len(converted_dev),
            "excluded_total": 0,
        },
        "task_counts": {
            "train": train_task_counts,
            "dev": dev_task_counts,
        },
        "provenance_fields": [
            "task",
            "layer",
            "source_batch",
            "source_split",
            "source_dataset_id",
            "source_task",
        ],
        "assumptions": [
            "The in-repo training contract is the messages schema enforced by scripts/prepare_dataset.py.",
            "All structured WorldSim tasks keep assistant targets as canonical JSON strings.",
            "No rows were excluded during conversion; any malformed source row would fail conversion explicitly.",
        ],
        "source_manifest_payload": source_manifest_payload,
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return AttrDict(
        train_output=train_output,
        dev_output=dev_output,
        manifest_path=manifest_path,
        train_count=len(converted_train),
        dev_count=len(converted_dev),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert finalized mixed WorldSim rows into the training-ready messages schema.")
    parser.add_argument("--input-train", required=True)
    parser.add_argument("--input-dev", required=True)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="worldsim-v31-mix-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = convert_mixed_final_to_training_format(
        repo_root=repo_root,
        input_train=resolve_path(repo_root, args.input_train),
        input_dev=resolve_path(repo_root, args.input_dev),
        source_manifest=resolve_path(repo_root, args.source_manifest) if args.source_manifest else None,
        output_dir=resolve_path(repo_root, args.output_dir),
        dataset_id=args.dataset_id,
    )
    print(
        json.dumps(
            {
                "train_output": str(result.train_output),
                "dev_output": str(result.dev_output),
                "manifest_path": str(result.manifest_path),
                "train_count": result.train_count,
                "dev_count": result.dev_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
