#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, ensure_within_directory, load_yaml, read_jsonl, resolve_path, write_jsonl


def _validate_dataset_name(dataset_name: str) -> str:
    if not dataset_name or dataset_name in {".", ".."}:
        raise ValueError("dataset_name must be a simple non-empty directory name")
    if Path(dataset_name).is_absolute() or Path(dataset_name).name != dataset_name or re.search(r"[\\/]", dataset_name):
        raise ValueError("dataset_name must not contain path separators")
    return dataset_name


def _resolve_default_paths(repo_root: Path, settings: dict, dataset_name: str) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    dataset_name = _validate_dataset_name(dataset_name)
    paths = settings.get("paths", {})
    validated_dir = resolve_path(repo_root, paths.get("validated_dir", "data/validated"))
    final_dir = resolve_path(repo_root, paths.get("final_dir", "data/final"))
    return (
        validated_dir / "postprocess" / "passed.jsonl",
        validated_dir / "recovery" / "recovered.jsonl",
        validated_dir / "review_samples" / "approved.jsonl",
        validated_dir / "postprocess" / "review.jsonl",
        validated_dir / "postprocess" / "failed.jsonl",
        validated_dir / "recovery" / "needs_review.jsonl",
        validated_dir / "recovery" / "unrecoverable.jsonl",
        final_dir,
        final_dir / dataset_name,
    )


def _resolve_output_dir(final_dir: Path, output_dir: Path) -> Path:
    ensure_directory(final_dir)
    return ensure_within_directory(final_dir, output_dir, label="final_dir output_dir")


def _read_required_jsonl(path: Path, *, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")
    return read_jsonl(path)


def _read_optional_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


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


def _dedupe_key(row: dict) -> tuple[str, str]:
    return str(row.get("task", "unknown")), _canonical_output(row.get("output"))


def _disposition(row: dict) -> str:
    postprocess = row.get("postprocess")
    if isinstance(postprocess, dict):
        value = postprocess.get("disposition")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _annotate_included(row: dict, *, source_name: str) -> dict:
    return {
        **row,
        "dataset_source": source_name,
    }


def _annotate_excluded(row: dict, *, source_name: str, reason: str, duplicate_of_source: str | None = None) -> dict:
    annotated = {
        **row,
        "dataset_source": source_name,
        "source_disposition": _disposition(row),
        "exclusion_reason": reason,
    }
    if duplicate_of_source:
        annotated["duplicate_of_source"] = duplicate_of_source
    return annotated


def _split_rows(rows: list[dict], *, dev_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 <= dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if len(rows) <= 1 or dev_fraction <= 0:
        return [{**row, "dataset_split": "train"} for row in rows], []

    dev_count = min(len(rows) - 1, max(1, int(len(rows) * dev_fraction)))
    if dev_count <= 0:
        return [{**row, "dataset_split": "train"} for row in rows], []

    indexed_rows = list(enumerate(rows))
    random.Random(seed).shuffle(indexed_rows)
    dev_indices = {index for index, _ in indexed_rows[:dev_count]}

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    for index, row in enumerate(rows):
        target = dev_rows if index in dev_indices else train_rows
        split = "dev" if index in dev_indices else "train"
        target.append({**row, "dataset_split": split})
    return train_rows, dev_rows


def _count_by_field(rows: list[dict], field: str) -> dict[str, int]:
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def _infer_version(rows: list[dict], *, keys: tuple[str, ...], fallback: str) -> str:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        postprocess = row.get("postprocess")
        if isinstance(postprocess, dict):
            for key in keys:
                value = postprocess.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        normalization = row.get("normalization")
        if isinstance(normalization, dict):
            for key in keys:
                value = normalization.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        validation = row.get("validation")
        if isinstance(validation, dict):
            for key in keys:
                value = validation.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return fallback


def assemble_final_dataset(
    repo_root: Path | None = None,
    *,
    passed_file: Path | None = None,
    recovered_file: Path | None = None,
    approved_review_file: Path | None = None,
    review_file: Path | None = None,
    failed_file: Path | None = None,
    needs_review_file: Path | None = None,
    unrecoverable_file: Path | None = None,
    output_dir: Path | None = None,
    dataset_name: str = "worldsim-final",
    dev_fraction: float = 0.1,
    seed: int = 42,
):
    if seed < 0:
        raise ValueError("seed must be >= 0")

    settings: dict = {}
    if repo_root is not None:
        settings = load_yaml(repo_root / "config" / "generation.yaml")
        default_paths = _resolve_default_paths(repo_root, settings, dataset_name)
        passed_file = passed_file or default_paths[0]
        recovered_file = recovered_file or default_paths[1]
        approved_review_file = approved_review_file or default_paths[2]
        review_file = review_file or default_paths[3]
        failed_file = failed_file or default_paths[4]
        needs_review_file = needs_review_file or default_paths[5]
        unrecoverable_file = unrecoverable_file or default_paths[6]
        output_dir = _resolve_output_dir(default_paths[7], output_dir or default_paths[8])
    elif None in (passed_file, output_dir):
        raise ValueError("Either repo_root or passed_file and output_dir must be provided")
    else:
        output_dir = Path(output_dir).resolve()

    passed_rows = _read_required_jsonl(Path(passed_file), label="postprocess passed")
    approved_review_rows = _read_optional_jsonl(Path(approved_review_file) if approved_review_file else None)
    recovered_rows = _read_optional_jsonl(Path(recovered_file) if recovered_file else None)
    review_rows = _read_optional_jsonl(Path(review_file) if review_file else None)
    failed_rows = _read_optional_jsonl(Path(failed_file) if failed_file else None)
    needs_review_rows = _read_optional_jsonl(Path(needs_review_file) if needs_review_file else None)
    unrecoverable_rows = _read_optional_jsonl(Path(unrecoverable_file) if unrecoverable_file else None)

    included_rows: list[dict] = []
    excluded_rows: list[dict] = []
    seen_keys: dict[tuple[str, str], str] = {}

    for source_name, rows in (
        ("passed", passed_rows),
        ("approved_review", approved_review_rows),
        ("recovered", recovered_rows),
    ):
        for row in rows:
            dedupe_key = _dedupe_key(row)
            if not dedupe_key[1]:
                excluded_rows.append(_annotate_excluded(row, source_name=source_name, reason="missing_output"))
                continue
            duplicate_of_source = seen_keys.get(dedupe_key)
            if duplicate_of_source is not None:
                excluded_rows.append(
                    _annotate_excluded(
                        row,
                        source_name=source_name,
                        reason="duplicate_content",
                        duplicate_of_source=duplicate_of_source,
                    )
                )
                continue
            seen_keys[dedupe_key] = source_name
            included_rows.append(_annotate_included(row, source_name=source_name))

    for row in review_rows:
        excluded_rows.append(_annotate_excluded(row, source_name="review", reason="review_not_approved"))
    for row in needs_review_rows:
        excluded_rows.append(_annotate_excluded(row, source_name="needs_review", reason="needs_review_not_approved"))
    for row in failed_rows:
        excluded_rows.append(_annotate_excluded(row, source_name="failed", reason="failed_source"))
    for row in unrecoverable_rows:
        excluded_rows.append(_annotate_excluded(row, source_name="unrecoverable", reason="unrecoverable_source"))

    train_rows, dev_rows = _split_rows(included_rows, dev_fraction=dev_fraction, seed=seed)

    ensure_directory(output_dir)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    excluded_path = output_dir / "excluded.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"

    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    write_jsonl(excluded_path, excluded_rows)

    all_source_rows = [
        *passed_rows,
        *approved_review_rows,
        *recovered_rows,
        *review_rows,
        *failed_rows,
        *needs_review_rows,
        *unrecoverable_rows,
    ]
    included_source_counts = Counter(row["dataset_source"] for row in included_rows)
    excluded_reason_counts = Counter(row["exclusion_reason"] for row in excluded_rows)
    manifest_payload = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "dev_fraction": dev_fraction,
        "output_files": {
            "train": str(train_path),
            "dev": str(dev_path),
            "excluded": str(excluded_path),
        },
        "source_files": {
            "passed_file": str(passed_file),
            "approved_review_file": str(approved_review_file) if approved_review_file else None,
            "recovered_file": str(recovered_file) if recovered_file else None,
            "review_file": str(review_file) if review_file else None,
            "failed_file": str(failed_file) if failed_file else None,
            "needs_review_file": str(needs_review_file) if needs_review_file else None,
            "unrecoverable_file": str(unrecoverable_file) if unrecoverable_file else None,
        },
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
            "included_total": len(included_rows),
            "excluded_total": len(excluded_rows),
        },
        "input_counts": {
            "passed": len(passed_rows),
            "approved_review": len(approved_review_rows),
            "recovered": len(recovered_rows),
            "review": len(review_rows),
            "failed": len(failed_rows),
            "needs_review": len(needs_review_rows),
            "unrecoverable": len(unrecoverable_rows),
        },
        "included_counts_by_source": dict(sorted(included_source_counts.items())),
        "excluded_counts_by_reason": dict(sorted(excluded_reason_counts.items())),
        "included_task_counts": _count_by_field(included_rows, "task"),
        "excluded_task_counts": _count_by_field(excluded_rows, "task"),
        "included_recovered_rows": included_source_counts.get("recovered", 0),
        "normalization_version": _infer_version(all_source_rows, keys=("normalization_version",), fallback="unknown"),
        "validator_version": _infer_version(all_source_rows, keys=("validator_version",), fallback="unknown"),
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return AttrDict(
        train_path=train_path,
        dev_path=dev_path,
        excluded_path=excluded_path,
        manifest_path=manifest_path,
        counts=manifest_payload["counts"],
        included_recovered_rows=manifest_payload["included_recovered_rows"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a conservative final dataset bundle from postprocess outputs.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--passed-file", default=None)
    parser.add_argument("--recovered-file", default=None)
    parser.add_argument("--approved-review-file", default=None)
    parser.add_argument("--review-file", default=None)
    parser.add_argument("--failed-file", default=None)
    parser.add_argument("--needs-review-file", default=None)
    parser.add_argument("--unrecoverable-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-name", default="worldsim-final")
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_dir = resolve_path(repo_root, args.config_dir)
    load_yaml(config_dir / "generation.yaml")

    try:
        result = assemble_final_dataset(
            repo_root=repo_root,
            passed_file=resolve_path(repo_root, args.passed_file) if args.passed_file else None,
            recovered_file=resolve_path(repo_root, args.recovered_file) if args.recovered_file else None,
            approved_review_file=resolve_path(repo_root, args.approved_review_file) if args.approved_review_file else None,
            review_file=resolve_path(repo_root, args.review_file) if args.review_file else None,
            failed_file=resolve_path(repo_root, args.failed_file) if args.failed_file else None,
            needs_review_file=resolve_path(repo_root, args.needs_review_file) if args.needs_review_file else None,
            unrecoverable_file=resolve_path(repo_root, args.unrecoverable_file) if args.unrecoverable_file else None,
            output_dir=resolve_path(repo_root, args.output_dir) if args.output_dir else None,
            dataset_name=args.dataset_name,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "train_path": str(result.train_path),
                "dev_path": str(result.dev_path),
                "excluded_path": str(result.excluded_path),
                "manifest_path": str(result.manifest_path),
                "counts": result.counts,
                "included_recovered_rows": result.included_recovered_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
