#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, ensure_within_directory, load_yaml, read_jsonl, resolve_path, write_jsonl


DEFAULT_TARGETS = {
    "A": 20,
    "B": 50,
    "C": 30,
    "E": 30,
    "F": 30,
    "G": 50,
    "H": 20,
    "RECOVERED": 30,
}

DIVERSITY_FIELDS = {
    "A": ("personality_id", "world_id", "dominant_trait", "temperament_expressed", "length_bucket", "disposition"),
    "B": ("personality_id", "world_id", "situation_id", "emotion_expressed", "length_bucket", "disposition"),
    "C": ("personality_id", "world_id", "speaker_role", "emotion_expressed", "register", "disposition"),
    "E": ("personality_id", "world_id", "situation_id", "personality_reasoning", "action_id", "disposition"),
    "F": ("personality_id", "world_id", "situation_id", "emotion", "previous_emotion", "disposition"),
    "G": ("personality_id", "oracle_id", "temperament_id", "action_tendency", "register", "disposition"),
    "H": ("worldbuilding_id", "expected_world_type", "name", "length_bucket", "disposition"),
    "RECOVERED": ("task", "personality_id", "world_id", "situation_id", "emotion_expressed", "register", "disposition"),
}


def _resolve_default_paths(repo_root: Path, settings: dict) -> tuple[Path, Path, Path, Path]:
    paths = settings.get("paths", {})
    validated_dir = resolve_path(repo_root, paths.get("validated_dir", "data/validated"))
    postprocess_dir = validated_dir / "postprocess"
    recovery_dir = validated_dir / "recovery"
    output_dir = validated_dir / "review_samples"
    return validated_dir, postprocess_dir, recovery_dir, output_dir


def _resolve_output_dir(validated_dir: Path, output_dir: Path) -> Path:
    ensure_directory(validated_dir)
    return ensure_within_directory(validated_dir, output_dir, label="validated_dir output_dir")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _parsed_output(row: dict) -> dict | list | None:
    output = row.get("output")
    if isinstance(output, (dict, list)):
        return output
    if not isinstance(output, str):
        return None
    text = output.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _lookup_output_field(parsed_output: dict | list | None, field: str) -> str | None:
    if not isinstance(parsed_output, dict):
        return None
    value = parsed_output.get(field)
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _primary_text(parsed_output: dict | list | None) -> str:
    for field in ("text_ko", "speech_ko", "hint_ko", "cause_ko", "interpretation_ko", "description_en", "name"):
        value = _lookup_output_field(parsed_output, field)
        if value:
            return value
    return ""


def _length_bucket(parsed_output: dict | list | None) -> str:
    length = len(_primary_text(parsed_output))
    if length == 0:
        return "missing"
    if length < 20:
        return "short"
    if length < 60:
        return "medium"
    return "long"


def _disposition(row: dict) -> str:
    postprocess = row.get("postprocess")
    if isinstance(postprocess, dict):
        value = postprocess.get("disposition")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _field_value(row: dict, field: str) -> str | None:
    parsed_output = _parsed_output(row)
    if field == "length_bucket":
        return _length_bucket(parsed_output)
    if field == "disposition":
        return _disposition(row)
    if field in row and isinstance(row[field], str):
        value = row[field].strip()
        return value or None
    return _lookup_output_field(parsed_output, field)


def _selection_priority(row: dict) -> int:
    return {
        "review": 3,
        "recoverable": 2,
        "approved_review": 1,
        "passed": 0,
    }.get(_disposition(row), 0)


def _stable_rank(row: dict, seed: int, bucket: str) -> str:
    payload = _canonical_json(row)
    return hashlib.sha256(f"{seed}:{bucket}:{payload}".encode("utf-8")).hexdigest()


def _novel_dimensions(row: dict, fields: tuple[str, ...], seen: dict[str, set[str]]) -> list[tuple[str, str]]:
    discoveries: list[tuple[str, str]] = []
    for field in fields:
        value = _field_value(row, field)
        if value is None or value in seen[field]:
            continue
        discoveries.append((field, value))
    return discoveries


def _sample_reason(bucket: str, novel_dimensions: list[tuple[str, str]], row: dict) -> str:
    label = bucket.lower()
    if novel_dimensions:
        summary = ", ".join(f"{field}={value}" for field, value in novel_dimensions[:3])
        return f"{label} diversity coverage: {summary}"
    return f"{label} stable fill after diversity coverage ({_disposition(row)})"


def _select_diverse_rows(rows: list[dict], *, bucket: str, target: int, seed: int) -> list[dict]:
    if target < 0:
        raise ValueError(f"{bucket} target must be >= 0")
    if target == 0 or not rows:
        return []

    fields = DIVERSITY_FIELDS[bucket]
    pending = sorted(rows, key=lambda row: _stable_rank(row, seed, bucket))
    seen = {field: set() for field in fields}
    selected: list[dict] = []

    while pending and len(selected) < target:
        best_index = 0
        best_score: tuple[int, int] | None = None
        best_dimensions: list[tuple[str, str]] = []

        for index, row in enumerate(pending):
            novel_dimensions = _novel_dimensions(row, fields, seen)
            score = (len(novel_dimensions), _selection_priority(row))
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
                best_dimensions = novel_dimensions

        chosen = pending.pop(best_index)
        for field, value in best_dimensions:
            seen[field].add(value)
        selected.append(
            {
                **chosen,
                "sample_group": bucket.lower(),
                "sample_rank": len(selected) + 1,
                "sample_reason": _sample_reason(bucket, best_dimensions, chosen),
            }
        )
    return selected


def _read_required_jsonl(path: Path, *, label: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")
    return read_jsonl(path)


def _read_optional_jsonl(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []


def sample_for_review(
    repo_root: Path | None = None,
    *,
    postprocess_dir: Path | None = None,
    recovery_dir: Path | None = None,
    output_dir: Path | None = None,
    target_a: int = DEFAULT_TARGETS["A"],
    target_b: int = DEFAULT_TARGETS["B"],
    target_c: int = DEFAULT_TARGETS["C"],
    target_e: int = DEFAULT_TARGETS["E"],
    target_f: int = DEFAULT_TARGETS["F"],
    target_g: int = DEFAULT_TARGETS["G"],
    target_h: int = DEFAULT_TARGETS["H"],
    target_recovered: int = DEFAULT_TARGETS["RECOVERED"],
    seed: int = 42,
):
    if seed < 0:
        raise ValueError("seed must be >= 0")

    settings: dict = {}
    if repo_root is not None:
        settings = load_yaml(repo_root / "config" / "generation.yaml")
        validated_dir, default_postprocess_dir, default_recovery_dir, default_output_dir = _resolve_default_paths(repo_root, settings)
        postprocess_dir = postprocess_dir or default_postprocess_dir
        recovery_dir = recovery_dir or default_recovery_dir
        output_dir = _resolve_output_dir(validated_dir, output_dir or default_output_dir)
    elif None in (postprocess_dir, recovery_dir, output_dir):
        raise ValueError("Either repo_root or postprocess_dir, recovery_dir, and output_dir must be provided")
    else:
        output_dir = Path(output_dir).resolve()

    passed_path = Path(postprocess_dir) / "passed.jsonl"
    recoverable_path = Path(postprocess_dir) / "recoverable.jsonl"
    review_path = Path(postprocess_dir) / "review.jsonl"
    recovered_path = Path(recovery_dir) / "recovered.jsonl"
    needs_review_path = Path(recovery_dir) / "needs_review.jsonl"

    postprocess_rows = [
        *_read_required_jsonl(passed_path, label="postprocess passed"),
        *_read_optional_jsonl(recoverable_path),
        *_read_optional_jsonl(review_path),
    ]
    recovered_rows = [
        *_read_optional_jsonl(recovered_path),
        *_read_optional_jsonl(needs_review_path),
    ]

    task_targets = {
        "A": target_a,
        "B": target_b,
        "C": target_c,
        "E": target_e,
        "F": target_f,
        "G": target_g,
        "H": target_h,
    }
    review_rows_by_task: dict[str, list[dict]] = {}
    for task, target in task_targets.items():
        review_rows_by_task[task] = _select_diverse_rows(
            [row for row in postprocess_rows if row.get("task") == task],
            bucket=task,
            target=target,
            seed=seed,
        )
    review_recovered = _select_diverse_rows(recovered_rows, bucket="RECOVERED", target=target_recovered, seed=seed)

    ensure_directory(output_dir)
    output_paths: dict[str, Path] = {}
    for task in task_targets:
        output_paths[task] = output_dir / f"review_task_{task.lower()}.jsonl"
    recovered_output_path = output_dir / "review_recovered.jsonl"
    manifest_path = output_dir / "review_manifest.json"

    for task, rows in review_rows_by_task.items():
        write_jsonl(output_paths[task], rows)
    write_jsonl(recovered_output_path, review_recovered)

    manifest_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "targets": {
            **task_targets,
            "recovered": target_recovered,
        },
        "selected_counts": {
            **{task: len(rows) for task, rows in review_rows_by_task.items()},
            "recovered": len(review_recovered),
        },
        "source_files": {
            "postprocess_passed_file": str(passed_path),
            "postprocess_recoverable_file": str(recoverable_path),
            "postprocess_review_file": str(review_path),
            "recovered_file": str(recovered_path),
            "needs_review_file": str(needs_review_path),
        },
        "output_files": {
            **{f"review_task_{task.lower()}": str(path) for task, path in output_paths.items()},
            "review_recovered": str(recovered_output_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = AttrDict(
        review_recovered=recovered_output_path,
        manifest_path=manifest_path,
        counts=manifest_payload["selected_counts"],
        review_paths={task: output_paths[task] for task in task_targets},
    )
    for task, path in output_paths.items():
        result[f"review_task_{task.lower()}"] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample deterministic review sets from postprocess outputs.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--postprocess-dir", default=None)
    parser.add_argument("--recovery-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--target-a", type=int, default=DEFAULT_TARGETS["A"])
    parser.add_argument("--target-b", type=int, default=DEFAULT_TARGETS["B"])
    parser.add_argument("--target-c", type=int, default=DEFAULT_TARGETS["C"])
    parser.add_argument("--target-e", type=int, default=DEFAULT_TARGETS["E"])
    parser.add_argument("--target-f", type=int, default=DEFAULT_TARGETS["F"])
    parser.add_argument("--target-g", type=int, default=DEFAULT_TARGETS["G"])
    parser.add_argument("--target-h", type=int, default=DEFAULT_TARGETS["H"])
    parser.add_argument("--target-recovered", type=int, default=DEFAULT_TARGETS["RECOVERED"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_dir = resolve_path(repo_root, args.config_dir)
    load_yaml(config_dir / "generation.yaml")

    try:
        result = sample_for_review(
            repo_root=repo_root,
            postprocess_dir=resolve_path(repo_root, args.postprocess_dir) if args.postprocess_dir else None,
            recovery_dir=resolve_path(repo_root, args.recovery_dir) if args.recovery_dir else None,
            output_dir=resolve_path(repo_root, args.output_dir) if args.output_dir else None,
            target_a=args.target_a,
            target_b=args.target_b,
            target_c=args.target_c,
            target_e=args.target_e,
            target_f=args.target_f,
            target_g=args.target_g,
            target_h=args.target_h,
            target_recovered=args.target_recovered,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                **{f"review_task_{task.lower()}": str(path) for task, path in result.review_paths.items()},
                "review_recovered": str(result.review_recovered),
                "manifest_path": str(result.manifest_path),
                "counts": result.counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
