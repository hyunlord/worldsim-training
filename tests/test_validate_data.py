import json
from pathlib import Path

import pytest
import yaml

from scripts.validate_data import _resolve_validated_output_dir, auto_repair, latest_raw_file, load_validation_rules, validate_dataset, validate_file, validate_json_output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def bilingual_rules_yaml() -> str:
    return """
validation:
  forbidden_words: ["마을", "식량"]
  meta_patterns: ["WorldSim"]
  trait_axes: ["honesty_humility", "emotionality", "extraversion", "agreeableness", "conscientiousness", "openness"]
  reasoning_axes: ["high_honesty_humility", "high_emotionality", "high_extraversion", "high_agreeableness", "high_conscientiousness", "high_openness"]
  speaker_roles: ["elder", "hunter", "shaman", "warrior", "healer", "gatherer", "craftsman", "chief", "scout", "observer"]
  transition_types: ["gradual", "sudden", "sustained"]
  register_endings:
    haera: ['다[.\\s]?$', '는다[.\\s]?$', '았다[.\\s]?$', '었다[.\\s]?$']
    hao: ['오[.\\s!]?$', '소[.\\s!]?$', '시오[.\\s!]?$']
    hae: ['해[.\\s!]?$', '야[.\\s!]?$', '지[.\\s!]?$', '어[.\\s!]?$']
  task_limits:
    A: {min_chars: 20, max_chars: 40, sentences: 1}
    B: {min_chars: 30, max_chars: 60, sentences: 2}
    C: {min_chars: 15, max_chars: 30, sentences: 1}
    D: {min_chars: 10, max_chars: 25, sentences: 1}
    E: {min_chars: 10, max_chars: 30, sentences: 1}
    F: {min_chars: 10, max_chars: 25, sentences: 1}
""".strip()


def test_validate_file_repairs_korean_json_fields_and_writes_report(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(bilingual_rules_yaml(), encoding="utf-8")
    write_jsonl(
        raw_file,
        [
            {
                "task": "A",
                "register": "haera",
                "dominant_trait": "conscientiousness",
                "output": compact_json(
                    {
                        "text_ko": "마을 곁을 살피며 먹거리를 챙겼다.",
                        "text_en": "Watched the camp and gathered food.",
                        "register": "haera",
                        "dominant_trait": "conscientiousness",
                        "temperament_expressed": "choleric",
                    }
                ),
            },
            {
                "task": "D",
                "situation_id": "food_found",
                "output": compact_json(
                    {
                        "text_ko": "돌이가 식량을 찾았다.",
                        "text_en": "Dol-i found food.",
                        "event_type": "food_found",
                    }
                ),
            },
        ],
    )

    rules = load_validation_rules(config_dir)
    summary = validate_file(raw_file, validated_dir=validated_dir, rules=rules)

    assert summary["passed"] == 2
    passed_rows = [json.loads(line) for line in (validated_dir / "passed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    repaired_a = json.loads(passed_rows[0]["output"])
    repaired_d = json.loads(passed_rows[1]["output"])
    assert repaired_a["text_ko"] == "무리가 사는 곳 곁을 살피며 먹거리를 챙겼다."
    assert repaired_d["text_ko"] == "돌이가 먹거리를 찾았다."
    assert repaired_a["temperament_expressed"] == "choleric"
    assert passed_rows[0]["repair_count"] == 1
    assert (validated_dir / "report.json").exists()


def test_auto_repair_replaces_forbidden_words() -> None:
    repaired, count = auto_repair("마을에서 식량을 지켰다")

    assert repaired == "무리가 사는 곳에서 먹거리를 지켰다"
    assert count == 2


def test_validate_json_output_checks_required_fields_and_enums() -> None:
    violations = validate_json_output(
        {
            "task": "C",
            "register": "hao",
            "speaker_role": "chief",
            "emotion_id": "anger",
            "output": compact_json(
                {
                    "speech_ko": "당장 나서시오!",
                    "register": "hao",
                    "emotion_expressed": "rage",
                    "speaker_role": "",
                }
            ),
        },
        load_validation_rules_for_inline(),
    )

    assert "missing_speech_en" in violations
    assert "invalid_emotion" in violations
    assert "invalid_speaker_role" in violations


def test_validate_json_output_rejects_registers_that_do_not_match_requested_task_register() -> None:
    violations = validate_json_output(
        {
            "task": "C",
            "register": "haera",
            "speaker_role": "chief",
            "emotion_id": "anger",
            "output": compact_json(
                {
                    "speech_ko": "지금 바로 앞으로 나오시오",
                    "speech_en": "Step forward right now.",
                    "register": "hao",
                    "emotion_expressed": "anger",
                    "speaker_role": "chief",
                }
            ),
        },
        load_validation_rules_for_inline(),
    )

    assert "invalid_register" in violations
    assert "register_mismatch" in violations


def test_validate_json_output_rejects_non_object_roots_and_bad_types() -> None:
    rules = load_validation_rules_for_inline()
    root_violations = validate_json_output({"task": "E", "action_options": ["도망"], "output": "[]"}, rules)
    type_violations = validate_json_output(
        {
            "task": "B",
            "emotion_id": "fear",
            "register": "haera",
            "output": compact_json(
                    {
                        "text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.",
                        "text_en": "The bushes shook. The whole body trembled.",
                        "register": "haera",
                        "emotion_expressed": "fear",
                        "intensity": True,
                    "mimetics": "오들오들",
                }
            ),
        },
        rules,
    )

    assert root_violations == ["json_root_not_object"]
    assert "invalid_numeric_range" in type_violations
    assert "invalid_mimetics" in type_violations


def test_validate_file_marks_invalid_bilingual_rows_as_failures(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(bilingual_rules_yaml(), encoding="utf-8")
    write_jsonl(
        raw_file,
        [
            {
                "task": "B",
                "register": "haera",
                "emotion_id": "fear",
                "output": compact_json(
                    {
                        "text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.",
                        "text_en": "",
                        "register": "hao",
                        "emotion_expressed": "fear",
                        "intensity": 0.9,
                        "mimetics": ["오들오들"],
                    }
                ),
            },
            {
                "task": "F",
                "current_emotion_id": "trust",
                "output": compact_json(
                    {
                        "emotion": "fear",
                        "intensity": 0.9,
                        "cause_ko": "날랜 짐승이 바로 눈앞에 나타났다",
                        "cause_en": "A fierce beast appeared.",
                        "previous_emotion": "joy",
                        "transition_type": "fast",
                    }
                ),
            },
        ],
    )

    summary = validate_file(raw_file, validated_dir=validated_dir, rules=load_validation_rules(config_dir))
    failed_rows = [json.loads(line) for line in (validated_dir / "failed.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert summary["failed"] == 2
    assert "missing_text_en" in failed_rows[0]["violations"]
    assert "invalid_register" in failed_rows[0]["violations"]
    assert "invalid_previous_emotion" in failed_rows[1]["violations"]
    assert "invalid_transition_type" in failed_rows[1]["violations"]


def test_validate_json_output_supports_v31_task_g_and_h_contracts() -> None:
    rules = load_validation_rules_for_inline()
    rules["temperament_ids"] = ["choleric", "melancholic"]
    rules["temperament_biases"] = ["action_oriented", "cautious_conservative"]
    rules["oracle_action_tendencies"] = ["mobilize", "defend", "wait", "retreat", "celebrate", "mourn"]
    rules["oracle_misinterpretations"] = [
        "overconfident_literal",
        "cautious_reversal",
        "optimistic_expansion",
        "passive_deferral",
        "symbolic_abstraction",
    ]

    valid_g = validate_json_output(
        {
            "task": "G",
            "register": "hao",
            "temperament_id": "choleric",
            "output": compact_json(
                {
                    "interpretation_ko": "산을 넘어가야 살 길이 열린다오.",
                    "interpretation_en": "We must cross the mountain to live.",
                    "action_tendency": "mobilize",
                    "confidence": 0.9,
                    "register": "hao",
                    "misinterpretation_type": "overconfident_literal",
                    "temperament_bias": "choleric_action_oriented",
                }
            ),
        },
        rules,
    )
    invalid_h = validate_json_output(
        {
            "task": "H",
            "output": compact_json(
                {
                    "name": "dungeonEconomy",
                    "description_en": "short",
                    "resource_modifiers": [{"target": "dungeon_loot", "multiplier": 9.0}],
                    "special_zones": "bad",
                    "special_resources": [],
                    "agent_modifiers": [],
                }
            ),
        },
        rules,
    )

    assert valid_g == []
    assert "invalid_name_format" in invalid_h
    assert "short_description" in invalid_h
    assert "missing_or_invalid_special_zones" in invalid_h
    assert "invalid_multiplier_range" in invalid_h


def test_validate_dataset_raises_clear_error_when_raw_dir_is_empty(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "config" / "generation.yaml").write_text(
        """
paths:
  raw_dir: data/raw
  validated_dir: data/validated
"""
        + "\n"
        + bilingual_rules_yaml(),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="No raw JSONL files found"):
        validate_dataset(tmp_path)


def test_latest_raw_file_uses_newest_mtime_not_lexicographic_name(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    older = raw_dir / "review_tmp_failure.jsonl"
    newer = raw_dir / "generated_20260308T123000000000Z.jsonl"
    older.write_text("{}\n", encoding="utf-8")
    newer.write_text("{}\n", encoding="utf-8")

    selected = latest_raw_file(raw_dir)

    assert selected == newer


def test_validate_file_raises_clear_error_when_input_file_is_missing(tmp_path: Path) -> None:
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(bilingual_rules_yaml(), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Validation input file does not exist"):
        validate_file(tmp_path / "data" / "raw" / "missing.jsonl", validated_dir=validated_dir, rules=load_validation_rules(config_dir))


def test_resolve_validated_output_dir_rejects_paths_outside_validated_dir(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
"""
        + "\n"
        + bilingual_rules_yaml(),
        encoding="utf-8",
    )
    settings = load_validation_rules(config_dir)
    full_settings = {"paths": {"validated_dir": "data/validated"}, "validation": settings}

    with pytest.raises(ValueError, match="validated_dir"):
        _resolve_validated_output_dir(tmp_path, full_settings, tmp_path / "escape")


def test_validate_file_rejects_validated_dir_outside_repo_when_repo_root_is_provided(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    raw_dir = tmp_path / "data" / "raw"
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "generation.yaml").write_text(
        """
paths:
  raw_dir: data/raw
  validated_dir: data/validated
"""
        + "\n"
        + bilingual_rules_yaml(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_dir / "sample.jsonl",
        [
            {
                "task": "A",
                "register": "haera",
                "dominant_trait": "conscientiousness",
                "output": compact_json(
                    {
                        "text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.",
                        "text_en": "Fearless and always sees things through.",
                        "register": "haera",
                        "dominant_trait": "conscientiousness",
                    }
                ),
            }
        ],
    )

    with pytest.raises(ValueError, match="validated_dir"):
        validate_file(input_path=raw_dir / "sample.jsonl", validated_dir=tmp_path / "escape", repo_root=tmp_path)


def load_validation_rules_for_inline() -> dict:
    return yaml.safe_load(bilingual_rules_yaml())["validation"]
