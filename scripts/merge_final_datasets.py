#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, read_jsonl, resolve_path, write_jsonl


ALLOWED_TASKS = {"A", "B", "C", "E", "F", "G", "H"}


@dataclass(slots=True)
class MergeSource:
    batch_id: str
    split: str
    rows: list[dict]
    manifest: dict[str, Any] | None
    source_path: Path
    manifest_path: Path | None


def _read_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def _loose_signature_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).lower()


def _parse_output_dict(row: dict) -> dict[str, Any] | None:
    output = row.get("output")
    if isinstance(output, dict):
        return dict(output)
    if not isinstance(output, str):
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _infer_task(row: dict) -> str | None:
    task = row.get("task")
    if isinstance(task, str) and task in ALLOWED_TASKS:
        return task

    payload = _parse_output_dict(row)
    if not isinstance(payload, dict):
        return None
    if "action_id" in payload:
        return "E"
    if "previous_emotion" in payload or "cause_ko" in payload:
        return "F"
    if "interpretation_ko" in payload:
        return "G"
    if "resource_modifiers" in payload or "agent_modifiers" in payload:
        return "H"
    if "speech_ko" in payload:
        return "C"
    if "dominant_trait" in payload:
        return "A"
    if "emotion_expressed" in payload:
        return "B"
    return None


def _stable_hash(parts: tuple[str, ...], *, seed: int) -> str:
    payload = "::".join(parts)
    return hashlib.sha256(f"{seed}:{payload}".encode("utf-8")).hexdigest()


def _annotate_source_row(row: dict, *, batch_id: str, split: str, line_number: int, task: str) -> dict:
    return {
        **row,
        "task": task,
        "merge_source_batch": batch_id,
        "merge_source_split": split,
        "merge_source_line": line_number,
        "merge_source_task": task,
    }


def _annotate_excluded(row: dict, *, reason: str, details: str | None = None) -> dict:
    annotated = {
        **row,
        "merge_exclusion_reason": reason,
    }
    if details:
        annotated["merge_exclusion_detail"] = details
    return annotated


def _task_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(str(row.get("task", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def _nested_counts(rows: list[dict], *, field: str) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[str(row.get("task", "unknown"))][str(row.get(field, "unknown"))] += 1
    return {task: dict(sorted(counter.items())) for task, counter in sorted(counts.items())}


def _prepare_sources(
    *,
    batch1_train: Path,
    batch1_dev: Path,
    batch1_manifest: Path | None,
    batch2_train: Path,
    batch2_dev: Path,
    batch2_manifest: Path | None,
) -> list[MergeSource]:
    return [
        MergeSource(
            batch_id="batch_v31_01_abc",
            split="train",
            rows=read_jsonl(batch1_train),
            manifest=_read_manifest(batch1_manifest),
            source_path=batch1_train,
            manifest_path=batch1_manifest,
        ),
        MergeSource(
            batch_id="batch_v31_01_abc",
            split="dev",
            rows=read_jsonl(batch1_dev),
            manifest=_read_manifest(batch1_manifest),
            source_path=batch1_dev,
            manifest_path=batch1_manifest,
        ),
        MergeSource(
            batch_id="batch_v31_02_gefhc",
            split="train",
            rows=read_jsonl(batch2_train),
            manifest=_read_manifest(batch2_manifest),
            source_path=batch2_train,
            manifest_path=batch2_manifest,
        ),
        MergeSource(
            batch_id="batch_v31_02_gefhc",
            split="dev",
            rows=read_jsonl(batch2_dev),
            manifest=_read_manifest(batch2_manifest),
            source_path=batch2_dev,
            manifest_path=batch2_manifest,
        ),
    ]


def merge_final_datasets(
    *,
    batch1_train: Path,
    batch1_dev: Path,
    batch1_manifest: Path | None,
    batch2_train: Path,
    batch2_dev: Path,
    batch2_manifest: Path | None,
    output_dir: Path,
    dataset_id: str,
    b_train_cap: int = 800,
    b_dev_cap: int = 96,
    seed: int = 42,
) -> AttrDict:
    if b_train_cap < 0 or b_dev_cap < 0:
        raise ValueError("B caps must be >= 0")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    ensure_directory(output_dir)
    sources = _prepare_sources(
        batch1_train=batch1_train,
        batch1_dev=batch1_dev,
        batch1_manifest=batch1_manifest,
        batch2_train=batch2_train,
        batch2_dev=batch2_dev,
        batch2_manifest=batch2_manifest,
    )

    candidates_by_split: dict[str, list[dict]] = {"dev": [], "train": []}
    excluded_rows: list[dict] = []
    source_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)

    for source in sources:
        source_counts[source.batch_id][source.split] = _task_counts(source.rows)
        for line_number, row in enumerate(source.rows, start=1):
            task = _infer_task(row)
            if task is None:
                excluded_rows.append(
                    _annotate_excluded(
                        _annotate_source_row(row, batch_id=source.batch_id, split=source.split, line_number=line_number, task="unknown"),
                        reason="ambiguous_task",
                    )
                )
                continue

            canonical_output = _canonical_output(row.get("output"))
            if not canonical_output:
                excluded_rows.append(
                    _annotate_excluded(
                        _annotate_source_row(row, batch_id=source.batch_id, split=source.split, line_number=line_number, task=task),
                        reason="malformed_source_row",
                        details="missing_output",
                    )
                )
                continue

            annotated = _annotate_source_row(row, batch_id=source.batch_id, split=source.split, line_number=line_number, task=task)
            annotated["_canonical_output"] = canonical_output
            annotated["_exact_signature"] = f"{task}::{canonical_output}"
            annotated["_loose_signature"] = f"{task}::{_loose_signature_text(canonical_output)}"
            annotated["_stable_rank"] = _stable_hash(
                (source.batch_id, source.split, str(line_number), canonical_output),
                seed=seed,
            )
            candidates_by_split[source.split].append(annotated)

    selected_by_split: dict[str, list[dict]] = {"dev": [], "train": []}
    for split in ("dev", "train"):
        b_cap = b_dev_cap if split == "dev" else b_train_cap
        b_candidates = [row for row in candidates_by_split[split] if row["task"] == "B"]
        non_b_candidates = [row for row in candidates_by_split[split] if row["task"] != "B"]
        kept_b_candidates = sorted(b_candidates, key=lambda row: row["_stable_rank"])[:b_cap]
        kept_b_signatures = {row["_stable_rank"] for row in kept_b_candidates}
        for row in sorted(b_candidates, key=lambda row: row["_stable_rank"]):
            if row["_stable_rank"] not in kept_b_signatures:
                excluded_rows.append(_annotate_excluded(row, reason="task_cap", details=f"B_{split}_cap"))
        selected_by_split[split] = non_b_candidates + kept_b_candidates

    included_rows: dict[str, list[dict]] = {"dev": [], "train": []}
    exact_seen: dict[str, dict[str, str]] = {}
    loose_seen: dict[str, dict[str, str]] = {}
    duplicate_exact = 0
    duplicate_near = 0

    for split in ("dev", "train"):
        ordered_rows = sorted(
            selected_by_split[split],
            key=lambda row: (row["merge_source_batch"], row["merge_source_line"]),
        )
        for row in ordered_rows:
            exact_signature = row["_exact_signature"]
            loose_signature = row["_loose_signature"]
            if exact_signature in exact_seen:
                duplicate_exact += 1
                seen = exact_seen[exact_signature]
                excluded_rows.append(
                    _annotate_excluded(
                        row,
                        reason="duplicate_content",
                        details=f"kept:{seen['batch']}:{seen['split']}",
                    )
                )
                continue
            if loose_signature in loose_seen:
                duplicate_near += 1
                seen = loose_seen[loose_signature]
                excluded_rows.append(
                    _annotate_excluded(
                        row,
                        reason="duplicate_near_content",
                        details=f"kept:{seen['batch']}:{seen['split']}",
                    )
                )
                continue

            included = {key: value for key, value in row.items() if not key.startswith("_")}
            included["dataset_split"] = split
            included["merge_dataset_id"] = dataset_id
            included_rows[split].append(included)
            exact_seen[exact_signature] = {"batch": row["merge_source_batch"], "split": split}
            loose_seen[loose_signature] = {"batch": row["merge_source_batch"], "split": split}

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    excluded_path = output_dir / "excluded.jsonl"
    manifest_path = output_dir / "merge_manifest.json"
    report_path = output_dir / "merge_report.json"

    write_jsonl(train_path, included_rows["train"])
    write_jsonl(dev_path, included_rows["dev"])
    write_jsonl(excluded_path, [{key: value for key, value in row.items() if not key.startswith("_")} for row in excluded_rows])

    included_task_counts = {
        task: {
            "train": sum(1 for row in included_rows["train"] if row.get("task") == task),
            "dev": sum(1 for row in included_rows["dev"] if row.get("task") == task),
            "total": sum(1 for row in included_rows["train"] + included_rows["dev"] if row.get("task") == task),
        }
        for task in sorted(ALLOWED_TASKS)
        if any(row.get("task") == task for row in included_rows["train"] + included_rows["dev"])
    }
    excluded_task_reason_counts = _nested_counts(excluded_rows, field="merge_exclusion_reason")
    excluded_reason_counts = Counter(str(row.get("merge_exclusion_reason", "unknown")) for row in excluded_rows)

    manifest_payload = {
        "dataset_id": dataset_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "input_sources": {
            source.batch_id: {
                "train_file": str(next(item.source_path for item in sources if item.batch_id == source.batch_id and item.split == "train")),
                "dev_file": str(next(item.source_path for item in sources if item.batch_id == source.batch_id and item.split == "dev")),
                "manifest_file": str(next((item.manifest_path for item in sources if item.batch_id == source.batch_id and item.manifest_path is not None), None)),
                "source_counts_by_split": source_counts[source.batch_id],
                "source_manifest": next((item.manifest for item in sources if item.batch_id == source.batch_id and item.manifest is not None), None),
            }
            for source in sources
            if source.split == "train"
        },
        "b_cap": {"train": b_train_cap, "dev": b_dev_cap},
        "included_counts_by_task": included_task_counts,
        "excluded_counts_by_task": excluded_task_reason_counts,
        "excluded_counts_by_reason": dict(sorted(excluded_reason_counts.items())),
        "duplicate_filtering": {
            "exact_duplicates": duplicate_exact,
            "near_duplicates": duplicate_near,
            "excluded_total": duplicate_exact + duplicate_near,
        },
        "final_counts": {
            "train": len(included_rows["train"]),
            "dev": len(included_rows["dev"]),
            "included_total": len(included_rows["train"]) + len(included_rows["dev"]),
            "excluded_total": len(excluded_rows),
        },
        "assumptions": [
            "Merged only finalized train/dev rows from batch finals.",
            "Dev rows were processed before train rows to prevent split leakage.",
            "Task B was capped with deterministic hash ordering.",
            "Duplicate filtering used task + canonical output exact match and punctuation-stripped loose match.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_payload = {
        "dataset_id": dataset_id,
        "included_tasks": _task_counts(included_rows["train"] + included_rows["dev"]),
        "excluded_reasons": dict(sorted(excluded_reason_counts.items())),
        "b_cap": manifest_payload["b_cap"],
        "duplicate_filtering": manifest_payload["duplicate_filtering"],
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return AttrDict(
        train_path=train_path,
        dev_path=dev_path,
        excluded_path=excluded_path,
        manifest_path=manifest_path,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge finalized batch datasets with conservative task caps and provenance.")
    parser.add_argument("--batch1-train", required=True)
    parser.add_argument("--batch1-dev", required=True)
    parser.add_argument("--batch1-manifest", required=False)
    parser.add_argument("--batch2-train", required=True)
    parser.add_argument("--batch2-dev", required=True)
    parser.add_argument("--batch2-manifest", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", default="worldsim-v31-mix-v1")
    parser.add_argument("--b-train-cap", type=int, default=800)
    parser.add_argument("--b-dev-cap", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = merge_final_datasets(
        batch1_train=resolve_path(repo_root, args.batch1_train),
        batch1_dev=resolve_path(repo_root, args.batch1_dev),
        batch1_manifest=resolve_path(repo_root, args.batch1_manifest) if args.batch1_manifest else None,
        batch2_train=resolve_path(repo_root, args.batch2_train),
        batch2_dev=resolve_path(repo_root, args.batch2_dev),
        batch2_manifest=resolve_path(repo_root, args.batch2_manifest) if args.batch2_manifest else None,
        output_dir=resolve_path(repo_root, args.output_dir),
        dataset_id=args.dataset_id,
        b_train_cap=args.b_train_cap,
        b_dev_cap=args.b_dev_cap,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "train_path": str(result.train_path),
                "dev_path": str(result.dev_path),
                "excluded_path": str(result.excluded_path),
                "manifest_path": str(result.manifest_path),
                "report_path": str(result.report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
