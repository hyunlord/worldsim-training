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

from scripts.common import AttrDict, ensure_directory, ensure_within_directory, load_yaml, read_jsonl, resolve_path, write_jsonl

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
DEFAULT_EMOTIONS = {"joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"}
DEFAULT_TRAIT_AXES = {
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
}
DEFAULT_REASONING_AXES = {f"high_{axis}" for axis in DEFAULT_TRAIT_AXES}
DEFAULT_SPEAKER_ROLES = {"elder", "hunter", "shaman", "warrior", "healer", "gatherer", "craftsman", "chief", "scout", "observer"}
DEFAULT_TRANSITION_TYPES = {"gradual", "sudden", "sustained"}
TASK_REQUIRED_FIELDS = {
    "A": ["text_ko", "text_en", "register", "dominant_trait"],
    "B": ["text_ko", "text_en", "register", "emotion_expressed", "intensity", "mimetics"],
    "C": ["speech_ko", "speech_en", "register", "emotion_expressed", "speaker_role"],
    "D": ["text_ko", "text_en", "event_type"],
    "E": ["action_id", "confidence", "hint_ko", "hint_en", "personality_reasoning"],
    "F": ["emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type"],
}
TASK_KO_FIELDS = {
    "A": ["text_ko"],
    "B": ["text_ko"],
    "C": ["speech_ko"],
    "D": ["text_ko"],
    "E": ["hint_ko"],
    "F": ["cause_ko"],
}
TASK_EN_FIELDS = {
    "A": ["text_en"],
    "B": ["text_en"],
    "C": ["speech_en"],
    "D": ["text_en"],
    "E": ["hint_en"],
    "F": ["cause_en"],
}
TASK_PRIMARY_KO_FIELD = {
    "A": "text_ko",
    "B": "text_ko",
    "C": "speech_ko",
    "D": "text_ko",
    "E": "hint_ko",
    "F": "cause_ko",
}


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


def _compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _enum_values(rules: dict, key: str, default: set[str]) -> set[str]:
    values = rules.get(key)
    if not values:
        return set(default)
    return {str(value) for value in values}


def _register_values() -> set[str]:
    return {"haera", "hao", "hae"}


def _task_limits(rules: dict, task: str) -> dict:
    return rules.get("task_limits", {}).get(task, {"min_chars": 1, "max_chars": 999, "sentences": None})


def _primary_ko_field(task: str) -> str | None:
    return TASK_PRIMARY_KO_FIELD.get(task)


def _parse_output(output: object) -> tuple[dict | None, list[str]]:
    if isinstance(output, dict):
        return output, []
    if not isinstance(output, str):
        return None, ["not_json"]
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None, ["not_json"]
    if not isinstance(payload, dict):
        return None, ["json_root_not_object"]
    return payload, []


def _contextual_allowed_values(record: dict, rules: dict) -> dict[str, set[str] | None]:
    emotions = _enum_values(rules, "emotions", DEFAULT_EMOTIONS)
    allowed: dict[str, set[str] | None] = {
        "register": {str(record["register"])} if record.get("register") else _register_values(),
        "dominant_trait": {record["dominant_trait"]} if record.get("dominant_trait") else _enum_values(rules, "trait_axes", DEFAULT_TRAIT_AXES),
        "emotion_expressed": {record["emotion_id"]} if record.get("emotion_id") else emotions,
        "emotion": emotions,
        "speaker_role": {record["speaker_role"]} if record.get("speaker_role") else _enum_values(rules, "speaker_roles", DEFAULT_SPEAKER_ROLES),
        "event_type": {record["situation_id"]} if record.get("situation_id") else None,
        "personality_reasoning": {record["personality_reasoning"]} if record.get("personality_reasoning") else _enum_values(rules, "reasoning_axes", DEFAULT_REASONING_AXES),
        "previous_emotion": {record["current_emotion_id"]} if record.get("current_emotion_id") else emotions,
        "transition_type": _enum_values(rules, "transition_types", DEFAULT_TRANSITION_TYPES),
    }
    return allowed


def repair_and_validate_json_output(record: dict, rules: dict) -> tuple[str, list[str], int]:
    task = record.get("task", "")
    if task not in TASK_REQUIRED_FIELDS:
        return str(record.get("output", "")), ["unknown_task"], 0

    payload, parse_violations = _parse_output(record.get("output", ""))
    if payload is None:
        return str(record.get("output", "")), parse_violations, 0

    repair_count = 0
    for field in TASK_KO_FIELDS.get(task, []):
        value = payload.get(field)
        if isinstance(value, str):
            repaired, field_repairs = auto_repair(value, rules.get("auto_replacements", AUTO_REPLACEMENTS))
            payload[field] = repaired
            repair_count += field_repairs

    violations: list[str] = []
    required_fields = TASK_REQUIRED_FIELDS[task]
    for field in required_fields:
        if field not in payload:
            violations.append(f"missing_{field}")

    primary_ko_field = _primary_ko_field(task)
    limits = _task_limits(rules, task)
    allowed = _contextual_allowed_values(record, rules)

    for field in TASK_KO_FIELDS.get(task, []):
        value = payload.get(field)
        if field in required_fields and (not isinstance(value, str) or len(value.strip()) < 3):
            if field not in payload:
                continue
            violations.append(f"missing_{field}" if isinstance(value, str) else f"invalid_{field}_type")
            continue
        if not isinstance(value, str):
            continue
        forbidden = find_forbidden_words(value, rules.get("forbidden_words", []))
        if forbidden:
            violations.append(f"forbidden_in_{field}")
        meta = find_meta_patterns(value, rules.get("meta_patterns", []))
        if meta:
            violations.append(f"meta_in_{field}")
        if field == primary_ko_field:
            if len(value.strip()) < int(limits.get("min_chars", 1)):
                violations.append("too_short")
            if len(value.strip()) > int(limits.get("max_chars", 999)):
                violations.append("too_long")
            expected_sentences = limits.get("sentences")
            if expected_sentences is not None and count_sentences(value) != expected_sentences:
                violations.append("sentence_count_mismatch")
            if is_repetitive(value):
                violations.append("repetition_ko")
            register = record.get("register") or payload.get("register")
            if register and not matches_register(value, str(register), rules.get("register_endings", {})):
                violations.append("register_mismatch")

    for field in TASK_EN_FIELDS.get(task, []):
        value = payload.get(field)
        if not isinstance(value, str) or len(value.strip()) < 3:
            violations.append(f"missing_{field}" if field not in payload or isinstance(value, str) else f"invalid_{field}_type")

    if "register" in payload and str(payload["register"]) not in allowed["register"]:
        violations.append("invalid_register")

    if "dominant_trait" in payload and payload["dominant_trait"] not in (allowed["dominant_trait"] or set()):
        violations.append("invalid_dominant_trait")

    emotion_field = "emotion_expressed" if "emotion_expressed" in payload else "emotion" if "emotion" in payload else None
    if emotion_field:
        if payload[emotion_field] not in (allowed[emotion_field] or set()):
            violations.append("invalid_emotion")

    if "speaker_role" in payload and payload["speaker_role"] not in (allowed["speaker_role"] or set()):
        violations.append("invalid_speaker_role")

    if "event_type" in payload and allowed["event_type"] is not None and payload["event_type"] not in allowed["event_type"]:
        violations.append("invalid_event_type")

    if "personality_reasoning" in payload and payload["personality_reasoning"] not in (allowed["personality_reasoning"] or set()):
        violations.append("invalid_personality_reasoning")

    if "previous_emotion" in payload and payload["previous_emotion"] not in (allowed["previous_emotion"] or set()):
        violations.append("invalid_previous_emotion")

    if "transition_type" in payload and payload["transition_type"] not in (allowed["transition_type"] or set()):
        violations.append("invalid_transition_type")

    if "mimetics" in payload:
        if not isinstance(payload["mimetics"], list) or any(not isinstance(item, str) or not item.strip() for item in payload["mimetics"]):
            violations.append("invalid_mimetics")

    for numeric_field in ("confidence", "intensity"):
        if numeric_field in payload:
            value = payload[numeric_field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                violations.append("invalid_numeric_range")

    if "action_id" in payload:
        value = payload["action_id"]
        action_options = record.get("action_options")
        max_action = 9 if action_options is None else len(action_options) - 1
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= max_action:
            violations.append("invalid_action_id")

    return _compact_json(payload), sorted(set(violations)), repair_count


def validate_json_output(record: dict, rules: dict) -> list[str]:
    _, violations, _ = repair_and_validate_json_output(record, rules)
    return violations


def _paths_for_repo(repo_root: Path) -> tuple[Path, Path]:
    settings = load_yaml(repo_root / "config" / "generation.yaml")
    raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
    validated_dir = resolve_path(repo_root, settings.get("paths", {}).get("validated_dir", "data/validated"))
    return raw_dir, validated_dir


def _resolve_validated_output_dir(repo_root: Path, settings: dict, output_dir: str | Path | None = None) -> Path:
    validated_dir = ensure_directory(resolve_path(repo_root, settings.get("paths", {}).get("validated_dir", "data/validated")))
    if output_dir is None:
        return validated_dir
    candidate = resolve_path(repo_root, output_dir)
    return ensure_within_directory(validated_dir, candidate, label="validated_dir output_dir")


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
        settings = load_yaml(repo_root / "config" / "generation.yaml")
        raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
        default_validated_dir = _resolve_validated_output_dir(repo_root, settings)
        validated_dir = _resolve_validated_output_dir(repo_root, settings, validated_dir or default_validated_dir)
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
        repaired_output, violations, repair_count = repair_and_validate_json_output(record, rules)
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
    output_dir = _resolve_validated_output_dir(repo_root, settings, args.output_dir)

    try:
        summary = validate_file(input_path=input_path, validated_dir=output_dir, rules=load_validation_rules(config_dir))
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
