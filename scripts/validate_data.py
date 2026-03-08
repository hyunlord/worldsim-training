#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, load_yaml, read_jsonl, resolve_path, write_jsonl


def load_validation_rules(config_dir: Path) -> dict:
    settings = load_yaml(config_dir / "generation.yaml")
    if "validation" in settings:
        return settings["validation"]
    return settings.get("defaults", {}).get("validation", {})


def count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text.strip()) if part.strip()])


def find_forbidden_words(text: str, words: list[str]) -> list[str]:
    return [word for word in words if word in text]


def matches_register(text: str, expected_register: str, register_endings: dict) -> bool:
    patterns = register_endings.get(expected_register, [])
    if not patterns:
        return True
    stripped = text.strip()
    return any(re.search(pattern, stripped) for pattern in patterns)


def find_meta_patterns(text: str, meta_patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in meta_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found


def is_repetitive(text: str, threshold: float = 0.5) -> bool:
    words = text.split()
    if len(words) < 4:
        return False
    for index in range(len(words) - 2):
        if words[index] == words[index + 1] == words[index + 2]:
            return True
    bigrams = [f"{words[index]} {words[index + 1]}" for index in range(len(words) - 1)]
    if not bigrams:
        return False
    ratio = Counter(bigrams).most_common(1)[0][1] / len(bigrams)
    return ratio > threshold


def validate_record(record: dict, rules: dict) -> list[str]:
    text = record.get("output", "")
    task = record.get("task", "")
    register = record.get("register", "haera")
    limits = rules.get("task_limits", {}).get(task, {"min_chars": 1, "max_chars": 999, "sentences": None})
    violations: list[str] = []

    if not text or len(text.strip()) < 3:
        return ["empty"]

    char_count = len(text)
    if char_count < limits.get("min_chars", 1):
        violations.append("too_short")
    if char_count > limits.get("max_chars", 999):
        violations.append("too_long")

    expected_sentences = limits.get("sentences")
    if expected_sentences is not None and count_sentences(text) != expected_sentences:
        violations.append("sentence_count_mismatch")

    forbidden = find_forbidden_words(text, rules.get("forbidden_words", []))
    if forbidden:
        violations.append(f"forbidden({','.join(forbidden)})")

    register_endings = rules.get("register_endings", {})
    if register_endings and not matches_register(text, register, register_endings):
        violations.append("register_mismatch")

    meta = find_meta_patterns(text, rules.get("meta_patterns", []))
    if meta:
        violations.append(f"meta({','.join(meta)})")

    if "{" in text or "}" in text or "```" in text:
        violations.append("json_leak")
    if re.search(r"[A-Za-z]{3,}", text):
        violations.append("english_leak")
    if is_repetitive(text):
        violations.append("repetition")

    return violations


def _paths_for_repo(repo_root: Path) -> tuple[Path, Path]:
    settings = load_yaml(repo_root / "config" / "generation.yaml")
    raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
    validated_dir = resolve_path(repo_root, settings.get("paths", {}).get("validated_dir", "data/validated"))
    return raw_dir, validated_dir


def validate_file(
    input_path: Path | None = None,
    *,
    validated_dir: Path | None = None,
    rules: dict | None = None,
    repo_root: Path | None = None,
) -> dict:
    if repo_root is not None and (validated_dir is None or rules is None or input_path is None):
        raw_dir, default_validated_dir = _paths_for_repo(repo_root)
        validated_dir = validated_dir or default_validated_dir
        rules = rules or load_validation_rules(repo_root / "config")
        input_path = input_path or sorted(raw_dir.glob("*.jsonl"))[-1]

    if input_path is None or validated_dir is None or rules is None:
        raise ValueError("input_path, validated_dir, and rules are required unless repo_root is provided")

    passed: list[dict] = []
    failed: list[dict] = []
    breakdown: Counter[str] = Counter()

    for record in read_jsonl(input_path):
        violations = validate_record(record, rules)
        enriched = {**record, "violations": violations}
        if violations:
            failed.append(enriched)
            for violation in violations:
                breakdown[violation.split("(")[0]] += 1
        else:
            passed.append(enriched)

    ensure_directory(validated_dir)
    write_jsonl(validated_dir / "passed.jsonl", passed)
    write_jsonl(validated_dir / "failed.jsonl", failed)

    summary = {
        "input_file": str(input_path),
        "passed": len(passed),
        "failed": len(failed),
        "total": len(passed) + len(failed),
        "violations": dict(breakdown),
    }
    (validated_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_dataset(repo_root: Path, input_path: Path | None = None):
    _, validated_dir = _paths_for_repo(repo_root)
    summary = validate_file(input_path=input_path, validated_dir=validated_dir, rules=load_validation_rules(repo_root / "config"), repo_root=repo_root)
    return AttrDict(
        passed_path=validated_dir / "passed.jsonl",
        failed_path=validated_dir / "failed.jsonl",
        pass_count=summary["passed"],
        fail_count=summary["failed"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated WorldSim raw data.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    config_dir = resolve_path(repo_root, args.config_dir)
    settings = load_yaml(config_dir / "generation.yaml")

    input_value = args.input or sorted(resolve_path(repo_root, settings["paths"]["raw_dir"]).glob("*.jsonl"))[-1]
    input_path = resolve_path(repo_root, input_value) if isinstance(input_value, str) else input_value
    output_value = args.output_dir or settings["paths"]["validated_dir"]
    output_dir = resolve_path(repo_root, output_value)

    summary = validate_file(input_path=input_path, validated_dir=output_dir, rules=load_validation_rules(config_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
