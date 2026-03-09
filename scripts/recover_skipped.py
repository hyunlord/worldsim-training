#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, read_jsonl, resolve_path, write_jsonl
from scripts.lib.postprocess import CANONICAL_TASKS, classify_record, enrich_record, load_postprocess_policy


def recover_skipped(repo_root: Path, *, skipped_file: Path, output_dir: Path) -> AttrDict:
    ensure_directory(output_dir)
    policy = load_postprocess_policy(repo_root / "config")
    recovered: list[dict] = []
    review_rows: list[dict] = []
    unrecoverable: list[dict] = []
    counts_by_task: Counter[str] = Counter()
    counts_by_skip_reason: Counter[str] = Counter()
    counts_by_disposition: Counter[str] = Counter()
    recovery_actions: Counter[str] = Counter()

    for row in read_jsonl(skipped_file):
        task = str(row.get("task", ""))
        if task not in CANONICAL_TASKS:
            continue
        counts_by_task[task] += 1
        counts_by_skip_reason[row.get("skip_reason") or row.get("reason") or "unknown"] += 1
        result = classify_record(row, policy)
        enriched = enrich_record(row, result)
        for action in enriched["postprocess"]["recovery_actions"]:
            recovery_actions[action] += 1
        if result.disposition in {"passed", "recoverable"}:
            recovered.append(enriched)
            counts_by_disposition["recovered"] += 1
        elif result.disposition == "review":
            review_rows.append(enriched)
            counts_by_disposition["needs_review"] += 1
        else:
            unrecoverable.append(enriched)
            counts_by_disposition["unrecoverable"] += 1

    recovered_path = output_dir / "recovered.jsonl"
    review_path = output_dir / "needs_review.jsonl"
    unrecoverable_path = output_dir / "unrecoverable.jsonl"
    report_path = output_dir / "recovery_report.json"
    write_jsonl(recovered_path, recovered)
    write_jsonl(review_path, review_rows)
    write_jsonl(unrecoverable_path, unrecoverable)
    report = {
        "counts_by_task": dict(counts_by_task),
        "counts_by_original_skip_reason": dict(counts_by_skip_reason),
        "counts_by_final_disposition": dict(counts_by_disposition),
        "examples_of_common_recoverable_patterns": dict(recovery_actions.most_common(10)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return AttrDict(
        recovered_path=recovered_path,
        needs_review_path=review_path,
        unrecoverable_path=unrecoverable_path,
        report_path=report_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover usable A/B/C rows from skipped.jsonl.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--skipped-file", required=True)
    parser.add_argument("--output-dir", default="data/validated/recovery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = recover_skipped(
        repo_root=repo_root,
        skipped_file=resolve_path(repo_root, args.skipped_file),
        output_dir=resolve_path(repo_root, args.output_dir),
    )
    print(
        json.dumps(
            {
                "recovered_path": str(result.recovered_path),
                "needs_review_path": str(result.needs_review_path),
                "unrecoverable_path": str(result.unrecoverable_path),
                "report_path": str(result.report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
