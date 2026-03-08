from pathlib import Path

import json
import pytest

from scripts.validate_data import auto_repair, load_validation_rules, validate_dataset, validate_file, validate_layer3_json


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_validate_file_splits_pass_fail_and_writes_report(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
validation:
  forbidden_words: ["식량"]
  meta_patterns: ["WorldSim"]
  task_limits:
    A: {min_chars: 5, max_chars: 40, sentences: 1}
    B: {min_chars: 5, max_chars: 60, sentences: 2}
    C: {min_chars: 5, max_chars: 30, sentences: 1}
    D: {min_chars: 5, max_chars: 30, sentences: 1}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_file,
        [
            {"task": "A", "register": "haera", "output": "풀숲을 살피며 조심스레 걸음을 옮겼다."},
            {"task": "D", "register": "haera", "output": "식량을 찾았다."},
        ],
    )

    rules = load_validation_rules(config_dir)
    summary = validate_file(raw_file, validated_dir=validated_dir, rules=rules)

    assert summary["total"] == 2
    assert summary["passed"] == 2
    assert summary["failed"] == 0
    assert (validated_dir / "passed.jsonl").exists()
    assert (validated_dir / "failed.jsonl").exists()
    assert (validated_dir / "report.json").exists()


def test_auto_repair_replaces_forbidden_words() -> None:
    repaired, count = auto_repair("마을에서 식량을 지켰다")

    assert repaired == "무리가 사는 곳에서 먹거리를 지켰다"
    assert count == 2


def test_validate_layer3_json_checks_task_e_shape() -> None:
    violations = validate_layer3_json(
        json.dumps({"action_id": 6, "confidence": 1.2, "hint": ""}, ensure_ascii=False),
        "E",
        action_options=["도망", "숨기", "맞서기"],
    )

    assert "invalid_action_id" in violations
    assert "invalid_confidence" in violations
    assert "missing_hint" in violations


def test_validate_layer3_json_checks_task_f_shape() -> None:
    violations = validate_layer3_json(
        json.dumps({"emotion": "calm", "intensity": -0.1, "cause": ""}, ensure_ascii=False),
        "F",
    )

    assert "invalid_emotion" in violations
    assert "invalid_intensity" in violations
    assert "missing_cause" in violations


def test_validate_file_repairs_forbidden_words_and_passes_layer3_json(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
validation:
  forbidden_words: ["마을", "식량"]
  layer3_emotions: ["joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"]
  task_limits:
    A: {min_chars: 5, max_chars: 40, sentences: 1}
    E: {min_chars: 1, max_chars: 200, sentences: null}
    F: {min_chars: 1, max_chars: 200, sentences: null}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_file,
        [
            {
                "task": "A",
                "register": "haera",
                "output": "마을에서 식량을 지켰다.",
            },
            {
                "task": "E",
                "output": "{\"action_id\": 0, \"confidence\": 0.9, \"hint\": \"마을 곁에서 식량을 품었다\"}",
                "action_options": ["도망", "숨기", "맞서기"],
            },
            {
                "task": "F",
                "output": "{\"emotion\": \"fear\", \"intensity\": 0.85, \"cause\": \"마을에 짐승이 들이닥쳤다\"}",
            },
        ],
    )

    rules = load_validation_rules(config_dir)
    summary = validate_file(raw_file, validated_dir=validated_dir, rules=rules)

    assert summary["passed"] == 3
    assert summary["failed"] == 0

    passed_rows = [json.loads(line) for line in (validated_dir / "passed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert passed_rows[0]["output"] == "무리가 사는 곳에서 먹거리를 지켰다."
    assert "무리가 사는 곳" in passed_rows[1]["output"]
    assert passed_rows[1]["repair_count"] == 2
    assert passed_rows[2]["repair_count"] == 1


def test_validate_layer3_json_rejects_non_object_roots_and_non_string_text_fields() -> None:
    root_violations = validate_layer3_json("[]", "E", action_options=["도망"])
    hint_violations = validate_layer3_json(
        json.dumps({"action_id": 0, "confidence": 0.9, "hint": {"bad": "value"}}, ensure_ascii=False),
        "E",
        action_options=["도망"],
    )
    cause_violations = validate_layer3_json(
        json.dumps({"emotion": "fear", "intensity": 0.7, "cause": ["bad"]}, ensure_ascii=False),
        "F",
    )

    assert "json_root_not_object" in root_violations
    assert "invalid_hint_type" in hint_violations
    assert "invalid_cause_type" in cause_violations


def test_validate_layer3_json_rejects_boolean_numeric_fields() -> None:
    violations = validate_layer3_json(
        json.dumps({"action_id": True, "confidence": False, "hint": "곧바로 숨었다"}, ensure_ascii=False),
        "E",
        action_options=["도망", "숨기"],
    )

    assert "invalid_action_id" in violations
    assert "invalid_confidence" in violations


def test_validate_file_marks_unexpected_layer3_payloads_as_failures_instead_of_crashing(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
validation:
  forbidden_words: ["식량"]
  task_limits:
    E: {min_chars: 5, max_chars: 20, sentences: null}
    F: {min_chars: 5, max_chars: 20, sentences: null}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_file,
        [
            {"task": "E", "output": "[]", "action_options": ["도망"]},
            {"task": "F", "output": "{\"emotion\": \"fear\", \"intensity\": 0.7, \"cause\": {\"bad\": \"value\"}}"},
        ],
    )

    summary = validate_file(raw_file, validated_dir=validated_dir, rules=load_validation_rules(config_dir))
    failed_rows = [json.loads(line) for line in (validated_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["passed"] == 0
    assert summary["failed"] == 2
    assert failed_rows[0]["violations"] == ["json_root_not_object"]
    assert failed_rows[1]["violations"] == ["invalid_cause_type"]


def test_validate_file_applies_task_limits_to_layer3_text_fields(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
validation:
  task_limits:
    E: {min_chars: 5, max_chars: 10, sentences: null}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_file,
        [
            {
                "task": "E",
                "output": "{\"action_id\": 0, \"confidence\": 0.9, \"hint\": \"숨을 죽이고 오래 웅크렸다\"}",
                "action_options": ["도망", "숨기"],
            }
        ],
    )

    summary = validate_file(raw_file, validated_dir=validated_dir, rules=load_validation_rules(config_dir))
    failed_rows = [json.loads(line) for line in (validated_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["failed"] == 1
    assert failed_rows[0]["violations"] == ["too_long"]


def test_validate_dataset_raises_clear_error_when_raw_dir_is_empty(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "config" / "generation.yaml").write_text(
        """
paths:
  raw_dir: data/raw
  validated_dir: data/validated
validation:
  forbidden_words: []
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="No raw JSONL files found"):
        validate_dataset(tmp_path)


def test_validate_file_raises_clear_error_when_input_file_is_missing(tmp_path: Path) -> None:
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text("validation:\n  forbidden_words: []\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Validation input file does not exist"):
        validate_file(tmp_path / "data" / "raw" / "missing.jsonl", validated_dir=validated_dir, rules=load_validation_rules(config_dir))
