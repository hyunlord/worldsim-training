#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_directory, load_yaml, resolve_path
from scripts.lib.postprocess import snapshot_metadata


@dataclass(slots=True)
class SnapshotResult:
    snapshot_dir: Path
    raw_snapshot: Path
    skipped_snapshot: Path
    metadata_path: Path


def _default_snapshot_dir(repo_root: Path) -> Path:
    settings = load_yaml(repo_root / "config" / "generation.yaml")
    manifest_dir = resolve_path(repo_root, settings.get("paths", {}).get("manifest_dir", "artifacts/manifests"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return manifest_dir / "snapshots" / timestamp


def _copy_if_present(source: Path | None, destination: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    ensure_directory(destination.parent)
    shutil.copy2(source, destination)
    return destination


def create_snapshot(
    repo_root: Path,
    *,
    raw_file: Path,
    skipped_file: Path,
    passed_file: Path | None = None,
    failed_file: Path | None = None,
    report_file: Path | None = None,
    output_dir: Path | None = None,
) -> SnapshotResult:
    output_dir = output_dir or _default_snapshot_dir(repo_root)
    ensure_directory(output_dir)
    raw_snapshot = _copy_if_present(raw_file, output_dir / "generated.jsonl")
    skipped_snapshot = _copy_if_present(skipped_file, output_dir / "skipped.jsonl")
    passed_snapshot = _copy_if_present(passed_file, output_dir / "passed.jsonl")
    failed_snapshot = _copy_if_present(failed_file, output_dir / "failed.jsonl")
    report_snapshot = _copy_if_present(report_file, output_dir / "report.json")
    metadata_path = output_dir / "dataset_snapshot.json"
    metadata = snapshot_metadata(
        repo_root,
        source_files={
            "raw_generated_file": str(raw_file),
            "skipped_file": str(skipped_file),
            "passed_file": str(passed_file) if passed_file else None,
            "failed_file": str(failed_file) if failed_file else None,
            "report_file": str(report_file) if report_file else None,
        },
        snapshot_files={
            "raw_generated_file": str(raw_snapshot) if raw_snapshot else None,
            "skipped_file": str(skipped_snapshot) if skipped_snapshot else None,
            "passed_file": str(passed_snapshot) if passed_snapshot else None,
            "failed_file": str(failed_snapshot) if failed_snapshot else None,
            "report_file": str(report_snapshot) if report_snapshot else None,
        },
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return SnapshotResult(
        snapshot_dir=output_dir,
        raw_snapshot=raw_snapshot or output_dir / "generated.jsonl",
        skipped_snapshot=skipped_snapshot or output_dir / "skipped.jsonl",
        metadata_path=metadata_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible snapshot of generated WorldSim files.")
    parser.add_argument("--raw-file", required=True)
    parser.add_argument("--skipped-file", required=True)
    parser.add_argument("--passed-file", default=None)
    parser.add_argument("--failed-file", default=None)
    parser.add_argument("--report-file", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    result = create_snapshot(
        repo_root=repo_root,
        raw_file=resolve_path(repo_root, args.raw_file),
        skipped_file=resolve_path(repo_root, args.skipped_file),
        passed_file=resolve_path(repo_root, args.passed_file) if args.passed_file else None,
        failed_file=resolve_path(repo_root, args.failed_file) if args.failed_file else None,
        report_file=resolve_path(repo_root, args.report_file) if args.report_file else None,
        output_dir=resolve_path(repo_root, args.output_dir) if args.output_dir else None,
    )
    print(
        json.dumps(
            {
                "snapshot_dir": str(result.snapshot_dir),
                "raw_snapshot": str(result.raw_snapshot),
                "skipped_snapshot": str(result.skipped_snapshot),
                "metadata_path": str(result.metadata_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
