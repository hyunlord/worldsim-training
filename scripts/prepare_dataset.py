#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, load_yaml, read_jsonl, resolve_path, write_jsonl, write_yaml


def _legacy_counts(validated: int, negative: int, general: int) -> dict[str, int]:
    total = validated + negative + general
    return {
        "validated_passed": validated,
        "negative_samples": negative,
        "general_samples": general,
        "total_examples": total,
        "validated": validated,
        "negative": negative,
        "general": general,
        "total": total,
    }


def _tag_rows(rows: list[dict], source_split: str) -> list[dict]:
    return [{**row, "source_split": source_split} for row in rows]


def _resolve_repo_paths(repo_root: Path, settings: dict, dataset_name: str) -> tuple[Path, Path, Path, Path, Path]:
    paths = settings.get("paths", {})
    passed_file = resolve_path(repo_root, Path(paths.get("validated_dir", "data/validated")) / "passed.jsonl")
    negative_key = paths.get("negative_samples_file") or paths.get("negative_samples") or "data/samples/negative_examples.jsonl"
    general_key = paths.get("general_samples_file") or paths.get("general_samples") or "data/samples/general_korean.jsonl"
    output_file = resolve_path(repo_root, Path(paths.get("final_dir", "data/final")) / f"{dataset_name}.jsonl")
    manifest_dir = paths.get("manifest_dir") or paths.get("manifests_dir") or "artifacts/manifests"
    manifest_file = resolve_path(repo_root, Path(manifest_dir) / f"{dataset_name}_manifest.yaml")
    return passed_file, resolve_path(repo_root, negative_key), resolve_path(repo_root, general_key), output_file, manifest_file


def prepare_dataset(
    repo_root: Path | None = None,
    *,
    passed_file: Path | None = None,
    negative_samples_file: Path | None = None,
    general_samples_file: Path | None = None,
    output_file: Path | None = None,
    manifest_file: Path | None = None,
    dataset_name: str = "worldsim-training",
):
    dataset_mix = {"include_negative_samples": True, "include_general_samples": True}
    if repo_root is not None:
        settings = load_yaml(repo_root / "config" / "generation.yaml")
        dataset_mix.update(settings.get("dataset_mix", {}))
        if any(value is None for value in (passed_file, negative_samples_file, general_samples_file, output_file, manifest_file)):
            passed_file, negative_samples_file, general_samples_file, output_file, manifest_file = _resolve_repo_paths(repo_root, settings, dataset_name)
    elif None in (passed_file, negative_samples_file, general_samples_file, output_file, manifest_file):
        raise ValueError("Either repo_root or all file paths must be provided")

    validated_rows = _tag_rows(read_jsonl(passed_file), "validated")
    negative_rows = _tag_rows(read_jsonl(negative_samples_file), "negative") if dataset_mix["include_negative_samples"] else []
    general_rows = _tag_rows(read_jsonl(general_samples_file), "general") if dataset_mix["include_general_samples"] else []
    combined = [*validated_rows, *negative_rows, *general_rows]
    write_jsonl(output_file, combined)

    manifest_payload = {
        "dataset_name": dataset_name,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_path": str(output_file),
        "counts": {
            "validated": len(validated_rows),
            "negative": len(negative_rows),
            "general": len(general_rows),
            "total": len(combined),
        },
        "sources": {
            "passed_file": str(passed_file),
            "negative_samples_file": str(negative_samples_file),
            "general_samples_file": str(general_samples_file),
        },
    }
    write_yaml(manifest_file, manifest_payload)

    return AttrDict(
        dataset_path=output_file,
        manifest_path=manifest_file,
        counts=_legacy_counts(len(validated_rows), len(negative_rows), len(general_rows)),
        dataset_name=dataset_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final training dataset bundle.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--passed-file", default=None)
    parser.add_argument("--negative-samples", default=None)
    parser.add_argument("--general-samples", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--dataset-name", default="worldsim-training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_dir = resolve_path(repo_root, args.config_dir)
    settings = load_yaml(config_dir / "generation.yaml")
    passed_file, negative_samples_file, general_samples_file, output_file, manifest_file = _resolve_repo_paths(
        repo_root,
        settings,
        args.dataset_name,
    )

    result = prepare_dataset(
        repo_root=repo_root,
        passed_file=resolve_path(repo_root, args.passed_file) if args.passed_file else passed_file,
        negative_samples_file=resolve_path(repo_root, args.negative_samples) if args.negative_samples else negative_samples_file,
        general_samples_file=resolve_path(repo_root, args.general_samples) if args.general_samples else general_samples_file,
        output_file=resolve_path(repo_root, args.output_file) if args.output_file else output_file,
        manifest_file=resolve_path(repo_root, args.manifest_file) if args.manifest_file else manifest_file,
        dataset_name=args.dataset_name,
    )
    print(
        json.dumps(
            {
                "dataset_path": str(result.dataset_path),
                "manifest_path": str(result.manifest_path),
                "counts": result.counts,
                "dataset_name": result.dataset_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
