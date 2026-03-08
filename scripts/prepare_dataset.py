#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_within_directory, load_yaml, read_jsonl, resolve_path, write_jsonl, write_yaml


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


def _training_system_prompts(repo_root: Path | None, settings: dict) -> dict[str, str]:
    defaults = {
        "L3": "너는 석기시대 서사 도우미다. JSON으로만 답하라.",
        "L4": "너는 석기시대 서사 도우미다. JSON으로만 답하라. 한국어와 영어를 함께 써라.",
        "NEG": "너는 학습 샘플 감시자다. 제시된 답안이 버릴 예시인지 retain 또는 reject 한 단어로만 답하라.",
        "GEN": "너는 한국어 문장 도우미다. 자연스러운 일반 한국어 한 문장으로 답하라.",
    }
    if repo_root is None:
        return defaults

    training_prompts = settings.get("prompts", {}).get("training", {})
    layer3_path = training_prompts.get("layer3_system")
    layer4_path = training_prompts.get("layer4_system")
    negative_path = training_prompts.get("negative_system")
    general_path = training_prompts.get("general_system")
    if layer3_path:
        defaults["L3"] = resolve_path(repo_root, layer3_path).read_text(encoding="utf-8").strip()
    if layer4_path:
        defaults["L4"] = resolve_path(repo_root, layer4_path).read_text(encoding="utf-8").strip()
    if negative_path:
        defaults["NEG"] = resolve_path(repo_root, negative_path).read_text(encoding="utf-8").strip()
    if general_path:
        defaults["GEN"] = resolve_path(repo_root, general_path).read_text(encoding="utf-8").strip()
    return defaults


def _validate_messages_row(row: dict) -> dict:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Unsupported dataset row for task {row.get('task', 'unknown')}: invalid messages payload")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Unsupported dataset row for task {row.get('task', 'unknown')}: invalid messages payload")
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ValueError(f"Unsupported dataset row for task {row.get('task', 'unknown')}: invalid messages payload")
    return row


def _assistant_content(output: object, *, task: str) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, separators=(",", ":"))
    raise ValueError(f"Unsupported dataset row for task {task}: output must be a string or JSON value")


def _row_to_training_example(row: dict, system_prompts: dict[str, str]) -> dict:
    if "messages" in row:
        return _validate_messages_row(row)

    task = row.get("task")
    prompt = row.get("prompt")
    output = row.get("output")
    if task in {"A", "B", "C", "D", "E", "F"} and prompt and output:
        layer = row.get("layer", "L3" if task in {"E", "F"} else "L4")
        assistant_content = _assistant_content(output, task=task)
        return {
            "task": task,
            "layer": layer,
            "source_split": row.get("source_split"),
            "messages": [
                {"role": "system", "content": system_prompts[layer]},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content},
            ],
        }
    if task in {"A", "B", "C", "D", "E", "F"}:
        raise ValueError(f"Unsupported dataset row for task {task}: missing prompt/output")
    if task == "NEG" and output:
        sample_output = _assistant_content(output, task=task)
        user_content = "[TASK] NEG"
        if row.get("reason"):
            user_content += f"\n[REASON] {row['reason']}"
        user_content += f"\n[SAMPLE] {sample_output}"
        return {
            "task": task,
            "label": row.get("label", "reject"),
            "reason": row.get("reason"),
            "source_split": row.get("source_split"),
            "messages": [
                {"role": "system", "content": system_prompts["NEG"]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": row.get("label", "reject")},
            ],
        }
    if task == "NEG":
        raise ValueError("Unsupported dataset row for task NEG: missing output")
    if task == "GEN" and output:
        assistant_content = _assistant_content(output, task=task)
        return {
            "task": task,
            "label": row.get("label", "retain"),
            "source_split": row.get("source_split"),
            "messages": [
                {"role": "system", "content": system_prompts["GEN"]},
                {"role": "user", "content": row.get("prompt", "[TASK] GEN\n[STYLE] 자연스러운 일반 한국어 한 문장을 써라.")},
                {"role": "assistant", "content": assistant_content},
            ],
        }
    if task == "GEN":
        raise ValueError("Unsupported dataset row for task GEN: missing output")
    raise ValueError(f"Unsupported dataset row for task {task or 'unknown'}")


def _validate_dataset_name(dataset_name: str) -> str:
    if not dataset_name or dataset_name in {".", ".."}:
        raise ValueError("dataset_name must be a simple non-empty file stem")
    if Path(dataset_name).is_absolute() or Path(dataset_name).name != dataset_name or re.search(r"[\\/]", dataset_name):
        raise ValueError("dataset_name must not contain path separators")
    return dataset_name


def _resolve_repo_paths(repo_root: Path, settings: dict, dataset_name: str) -> tuple[Path, Path, Path, Path, Path]:
    dataset_name = _validate_dataset_name(dataset_name)
    paths = settings.get("paths", {})
    validated_dir = resolve_path(repo_root, paths.get("validated_dir", "data/validated"))
    final_dir = resolve_path(repo_root, paths.get("final_dir", "data/final"))
    manifest_dir = resolve_path(repo_root, paths.get("manifest_dir") or paths.get("manifests_dir") or "artifacts/manifests")
    passed_file = validated_dir / "passed.jsonl"
    negative_key = paths.get("negative_samples_file") or paths.get("negative_samples") or "data/samples/negative_examples.jsonl"
    general_key = paths.get("general_samples_file") or paths.get("general_samples") or "data/samples/general_korean.jsonl"
    output_file = (final_dir / f"{dataset_name}.jsonl").resolve()
    manifest_file = (manifest_dir / f"{dataset_name}_manifest.yaml").resolve()
    if output_file.parent != final_dir.resolve():
        raise ValueError("dataset_name escapes final_dir")
    if manifest_file.parent != manifest_dir.resolve():
        raise ValueError("dataset_name escapes manifest_dir")
    return passed_file, resolve_path(repo_root, negative_key), resolve_path(repo_root, general_key), output_file, manifest_file


def _require_file(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required {label} file does not exist: {path}")


def _resolve_dataset_outputs(repo_root: Path, settings: dict, output_file: Path, manifest_file: Path) -> tuple[Path, Path]:
    paths = settings.get("paths", {})
    final_dir = resolve_path(repo_root, paths.get("final_dir", "data/final"))
    manifest_dir = resolve_path(repo_root, paths.get("manifest_dir") or paths.get("manifests_dir") or "artifacts/manifests")
    return (
        ensure_within_directory(final_dir, output_file, label="final_dir output_file"),
        ensure_within_directory(manifest_dir, manifest_file, label="manifest_dir manifest_file"),
    )


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
    settings: dict = {}
    if repo_root is not None:
        settings = load_yaml(repo_root / "config" / "generation.yaml")
        dataset_mix.update(settings.get("dataset_mix", {}))
        if any(value is None for value in (passed_file, negative_samples_file, general_samples_file, output_file, manifest_file)):
            defaults = _resolve_repo_paths(repo_root, settings, dataset_name)
            passed_file = passed_file or defaults[0]
            negative_samples_file = negative_samples_file or defaults[1]
            general_samples_file = general_samples_file or defaults[2]
            output_file = output_file or defaults[3]
            manifest_file = manifest_file or defaults[4]
    elif None in (passed_file, negative_samples_file, general_samples_file, output_file, manifest_file):
        raise ValueError("Either repo_root or all file paths must be provided")

    _require_file(passed_file, label="validated")
    if dataset_mix["include_negative_samples"]:
        _require_file(negative_samples_file, label="negative samples")
    if dataset_mix["include_general_samples"]:
        _require_file(general_samples_file, label="general samples")
    if repo_root is not None:
        output_file, manifest_file = _resolve_dataset_outputs(repo_root, settings, output_file, manifest_file)

    validated_rows = _tag_rows(read_jsonl(passed_file), "validated")
    negative_rows = _tag_rows(read_jsonl(negative_samples_file), "negative") if dataset_mix["include_negative_samples"] else []
    general_rows = _tag_rows(read_jsonl(general_samples_file), "general") if dataset_mix["include_general_samples"] else []
    system_prompts = _training_system_prompts(repo_root, settings)
    combined = [
        *[_row_to_training_example(row, system_prompts) for row in validated_rows],
        *[_row_to_training_example(row, system_prompts) for row in negative_rows],
        *[_row_to_training_example(row, system_prompts) for row in general_rows],
    ]
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

    try:
        result = prepare_dataset(
            repo_root=repo_root,
            passed_file=resolve_path(repo_root, args.passed_file) if args.passed_file else passed_file,
            negative_samples_file=resolve_path(repo_root, args.negative_samples) if args.negative_samples else negative_samples_file,
            general_samples_file=resolve_path(repo_root, args.general_samples) if args.general_samples else general_samples_file,
            output_file=resolve_path(repo_root, args.output_file) if args.output_file else output_file,
            manifest_file=resolve_path(repo_root, args.manifest_file) if args.manifest_file else manifest_file,
            dataset_name=args.dataset_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
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
