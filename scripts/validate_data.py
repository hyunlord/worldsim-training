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

AUTO_REPLACEMENTS = {
    "마을": "무리가 사는 곳",
    "식량": "먹거리",
    "전투": "피 흘리는 싸움",
    "자연": "온 누리",
    "건축": "집짓기",
    "족장": "우두머리",
    "구출": "빼내오기",
    "맹수": "날랜 짐승",
    "동맹": "손잡기",
    "공격": "덤비기",
    "방어": "막아서기",
    "이동": "옮겨가기",
    "위험": "아슬아슬한 것",
    "동물": "짐승",
    "무기": "날붙이",
    "기술": "솜씨",
}
LAYER3_TASKS = {"E", "F"}
DEFAULT_LAYER3_EMOTIONS = {"joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"}


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


def _has_final_consonant(char: str) -> bool:
    if not char or not ("가" <= char <= "힣"):
        return False
    return (ord(char) - ord("가")) % 28 != 0


def _normalize_particles(text: str) -> str:
    pairs = {
        ("을", "를"): lambda final: "을" if final else "를",
        ("이", "가"): lambda final: "이" if final else "가",
        ("은", "는"): lambda final: "은" if final else "는",
        ("과", "와"): lambda final: "과" if final else "와",
    }
    repaired = text
    for (first, second), selector in pairs.items():
        pattern = re.compile(rf"([가-힣]+)({first}|{second})(?=$|[^가-힣])")

        def repl(match: re.Match[str]) -> str:
            word = match.group(1)
            particle = selector(_has_final_consonant(word[-1]))
            return f"{word}{particle}"

        repaired = pattern.sub(repl, repaired)
    return repaired


def auto_repair(text: str, replacements: dict[str, str] | None = None) -> tuple[str, int]:
    repaired = text
    count = 0
    replacements = replacements or AUTO_REPLACEMENTS
    for forbidden, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if forbidden in repaired:
            repaired = repaired.replace(forbidden, replacement)
            count += 1
    return _normalize_particles(repaired), count


def validate_layer3_json(
    text: str,
    task: str,
    *,
    action_options: list[str] | None = None,
    valid_emotions: set[str] | None = None,
    text_limits: dict | None = None,
) -> list[str]:
    violations: list[str] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ["json_parse_error"]
    if not isinstance(data, dict):
        return ["json_root_not_object"]

    if task == "E":
        action_id = data.get("action_id")
        if action_id is None:
            violations.append("missing_action_id")
        elif isinstance(action_id, bool) or not isinstance(action_id, int) or action_id < 0 or action_id > 9 or (action_options is not None and action_id >= len(action_options)):
            violations.append("invalid_action_id")

        confidence = data.get("confidence")
        if confidence is None:
            violations.append("missing_confidence")
        elif isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            violations.append("invalid_confidence")

        hint = data.get("hint")
        if hint is None or len(str(hint).strip()) < 5:
            violations.append("missing_hint")
        elif not isinstance(hint, str):
            violations.append("invalid_hint_type")
        else:
            min_chars = int((text_limits or {}).get("min_chars", 0))
            max_chars = int((text_limits or {}).get("max_chars", 9999))
            if len(hint.strip()) < min_chars:
                violations.append("too_short")
            if len(hint.strip()) > max_chars:
                violations.append("too_long")

    elif task == "F":
        emotion = data.get("emotion")
        valid_emotions = valid_emotions or DEFAULT_LAYER3_EMOTIONS
        if emotion not in valid_emotions:
            violations.append("invalid_emotion")

        intensity = data.get("intensity")
        if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= float(intensity) <= 1:
            violations.append("invalid_intensity")

        cause = data.get("cause")
        if cause is None or len(str(cause).strip()) < 5:
            violations.append("missing_cause")
        elif not isinstance(cause, str):
            violations.append("invalid_cause_type")
        else:
            min_chars = int((text_limits or {}).get("min_chars", 0))
            max_chars = int((text_limits or {}).get("max_chars", 9999))
            if len(cause.strip()) < min_chars:
                violations.append("too_short")
            if len(cause.strip()) > max_chars:
                violations.append("too_long")

    return violations


def repair_layer3_json(text: str, task: str, *, replacements: dict[str, str] | None = None) -> tuple[str, int]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, 0
    if not isinstance(payload, dict):
        return text, 0

    repair_count = 0
    if task == "E" and isinstance(payload.get("hint"), str):
        payload["hint"], repair_count = auto_repair(payload["hint"], replacements=replacements)
    elif task == "F" and isinstance(payload.get("cause"), str):
        payload["cause"], repair_count = auto_repair(payload["cause"], replacements=replacements)

    return json.dumps(payload, ensure_ascii=False), repair_count


def validate_record(record: dict, rules: dict) -> tuple[str, list[str], int]:
    text = record.get("output", "")
    task = record.get("task", "")
    repaired_text = text
    repair_count = 0
    replacements = rules.get("auto_replacements", AUTO_REPLACEMENTS)
    layer3_rules = rules.get("layer3_json", {})

    if task in LAYER3_TASKS:
        repaired_text, repair_count = repair_layer3_json(text, task, replacements=replacements)
        task_key = "task_e" if task == "E" else "task_f"
        task_rule = layer3_rules.get(task_key, {})
        valid_emotions = set(task_rule.get("valid_emotions", rules.get("layer3_emotions", DEFAULT_LAYER3_EMOTIONS)))
        text_limits = rules.get("task_limits", {}).get(task, {})
        violations = validate_layer3_json(
            repaired_text,
            task,
            action_options=record.get("action_options"),
            valid_emotions=valid_emotions,
            text_limits=text_limits,
        )
        if violations:
            return repaired_text, violations, repair_count

        payload = json.loads(repaired_text)
        text_fields = [payload.get("hint", "")] if task == "E" else [payload.get("cause", "")]
        forbidden: list[str] = []
        meta: list[str] = []
        for field_text in text_fields:
            if not isinstance(field_text, str):
                continue
            forbidden.extend(find_forbidden_words(field_text, rules.get("forbidden_words", [])))
            meta.extend(find_meta_patterns(field_text, rules.get("meta_patterns", [])))
        if forbidden:
            return repaired_text, [f"forbidden({','.join(sorted(set(forbidden)))})"], repair_count
        if meta:
            return repaired_text, [f"meta({','.join(sorted(set(meta)))})"], repair_count
        return repaired_text, [], repair_count

    register = record.get("register", "haera")
    limits = rules.get("task_limits", {}).get(task, {"min_chars": 1, "max_chars": 999, "sentences": None})
    violations: list[str] = []

    repaired_text, repair_count = auto_repair(text, replacements=replacements)
    if not repaired_text or len(repaired_text.strip()) < 3:
        return repaired_text, ["empty"], repair_count

    char_count = len(repaired_text)
    if char_count < limits.get("min_chars", 1):
        violations.append("too_short")
    if char_count > limits.get("max_chars", 999):
        violations.append("too_long")

    expected_sentences = limits.get("sentences")
    if expected_sentences is not None and count_sentences(repaired_text) != expected_sentences:
        violations.append("sentence_count_mismatch")

    forbidden = find_forbidden_words(repaired_text, rules.get("forbidden_words", []))
    if forbidden:
        violations.append(f"forbidden({','.join(forbidden)})")

    register_endings = rules.get("register_endings", {})
    if register_endings and not matches_register(repaired_text, register, register_endings):
        violations.append("register_mismatch")

    meta = find_meta_patterns(repaired_text, rules.get("meta_patterns", []))
    if meta:
        violations.append(f"meta({','.join(meta)})")

    if "{" in repaired_text or "}" in repaired_text or "```" in repaired_text:
        violations.append("json_leak")
    if re.search(r"[A-Za-z]{3,}", repaired_text):
        violations.append("english_leak")
    if is_repetitive(repaired_text):
        violations.append("repetition")

    return repaired_text, violations, repair_count


def _paths_for_repo(repo_root: Path) -> tuple[Path, Path]:
    settings = load_yaml(repo_root / "config" / "generation.yaml")
    raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
    validated_dir = resolve_path(repo_root, settings.get("paths", {}).get("validated_dir", "data/validated"))
    return raw_dir, validated_dir


def latest_raw_file(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No raw JSONL files found in {raw_dir}")
    return candidates[-1]


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
        input_path = input_path or latest_raw_file(raw_dir)

    if input_path is None or validated_dir is None or rules is None:
        raise ValueError("input_path, validated_dir, and rules are required unless repo_root is provided")
    if not input_path.exists():
        raise FileNotFoundError(f"Validation input file does not exist: {input_path}")

    passed: list[dict] = []
    failed: list[dict] = []
    breakdown: Counter[str] = Counter()

    for record in read_jsonl(input_path):
        repaired_output, violations, repair_count = validate_record(record, rules)
        enriched = {**record, "output": repaired_output, "violations": violations}
        if repair_count:
            enriched["repair_count"] = repair_count
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

    input_value = args.input or latest_raw_file(resolve_path(repo_root, settings["paths"]["raw_dir"]))
    input_path = resolve_path(repo_root, input_value) if isinstance(input_value, str) else input_value
    output_value = args.output_dir or settings["paths"]["validated_dir"]
    output_dir = resolve_path(repo_root, output_value)

    try:
        summary = validate_file(input_path=input_path, validated_dir=output_dir, rules=load_validation_rules(config_dir))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
