from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyze_sample_classifies_valid_json() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "E",
        "generated_assistant": json.dumps(
            {
                "action_id": 0,
                "confidence": 0.9,
                "hint_ko": "곧 달아났다",
                "hint_en": "fled quickly",
                "personality_reasoning": "high_HA",
            },
            ensure_ascii=False,
        )
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "ok"


def test_analyze_sample_classifies_fenced_json() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "E",
        "generated_assistant": "```json\n{\"action_id\":0,\"confidence\":0.9,\"hint_ko\":\"곧 달아났다\",\"hint_en\":\"fled\",\"personality_reasoning\":\"high_HA\"}\n```",
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "fenced_json"


def test_analyze_sample_classifies_truncation() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "G",
        "generated_assistant": "{\"interpretation_ko\":\"나는 이 말을 해석하고 곧 나서야 한다고 생각하오\"",
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "truncation"


def test_analyze_sample_classifies_trailing_text() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "A",
        "generated_assistant": "{\"text_ko\":\"조심조심 걷는다\",\"text_en\":\"Walks carefully\",\"register\":\"haera\",\"dominant_trait\":\"harm_avoidance\",\"temperament_expressed\":\"melancholic\"} trailing",
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "trailing_text"


def test_analyze_sample_classifies_enum_drift() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "G",
        "generated_assistant": json.dumps(
            {
                "interpretation_ko": "나는 이 말을 해석하고 곧 나서야 한다고 생각하오",
                "interpretation_en": "I interpret this as a call to act now.",
                "action_tendency": "Defend",
                "confidence": 0.9,
                "register": "hao",
                "misinterpretation_type": "Overconfident Literal",
                "temperament_bias": "action oriented",
            },
            ensure_ascii=False,
        ),
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "enum_drift"
    assert analysis["enum_drift"][0]["field_name"] == "action_tendency"


def test_analyze_sample_classifies_language_drift() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "G",
        "generated_assistant": json.dumps(
            {
                "interpretation_ko": "This means we should advance immediately.",
                "interpretation_en": "This means we should advance immediately.",
                "action_tendency": "mobilize",
                "confidence": 0.9,
                "register": "hao",
                "misinterpretation_type": "overconfident_literal",
                "temperament_bias": "action oriented",
            },
            ensure_ascii=False,
        ),
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "language_drift"


def test_analyze_sample_classifies_semantic_low_quality() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "G",
        "generated_assistant": json.dumps(
            {
                "interpretation_ko": "나는 곧 나서야 하오",
                "interpretation_en": "I should act soon.",
                "action_tendency": "mobilize",
                "confidence": 0.9,
                "register": "hao",
                "misinterpretation_type": "passive_deferral",
                "temperament_bias": "action oriented",
            },
            ensure_ascii=False,
        ),
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "semantic_low_quality"


def test_analyze_sample_classifies_prompt_leakage() -> None:
    from tools.generation_analyzer import analyze_sample

    sample = {
        "task": "C",
        "generated_assistant": json.dumps(
            {
                "speech_ko": "I'm here to help you find your way.",
                "speech_en": "I am here to assist you in finding your path.",
                "register": "haera",
                "emotion_expressed": "trust",
                "speaker_role": "shaman",
                "temperament_tone": "snake_case phrase",
            },
            ensure_ascii=False,
        ),
    }

    analysis = analyze_sample(sample)
    assert analysis["primary_category"] == "prompt_leakage"
    assert analysis["prompt_leakage"]["pattern"] == "placeholder_literal"


def test_summarize_samples_reports_counts() -> None:
    from tools.generation_analyzer import summarize_samples

    samples = [
        {"task": "E", "generated_assistant": "{\"action_id\":0,\"confidence\":0.9,\"hint_ko\":\"곧 달아났다\",\"hint_en\":\"fled\",\"personality_reasoning\":\"high_HA\"}"},
        {"task": "G", "generated_assistant": "{\"interpretation_ko\":\"나는 이 말을 해석하고 곧 나서야 한다고 생각하오\",\"interpretation_en\":\"I interpret this as a call to act now.\",\"action_tendency\":\"Defend\",\"confidence\":0.9,\"register\":\"hao\",\"misinterpretation_type\":\"Overconfident Literal\",\"temperament_bias\":\"action oriented\"}"},
    ]

    summary = summarize_samples(samples)
    assert summary["total_samples"] == 2
    assert summary["counts_by_failure_category"]["ok"] == 1
    assert summary["counts_by_failure_category"]["enum_drift"] == 1


def test_generate_report_includes_structured_success_metrics() -> None:
    from tools.generation_analyzer import generate_report

    samples = [
        {
            "task": "A",
            "generated_assistant": '{"text_ko":"조심조심 걷는다","text_en":"Walks carefully","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}',
            "raw_generated_assistant": '{"text_ko":"조심조심 걷는다","text_en":"Walks carefully","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic","schema_explanation":"do not copy"}',
            "structured_attempt_count": 2,
            "structured_repair_applied": True,
            "structured_decoding": {"enabled": False, "supported": False, "used_mode": "none"},
        },
        {
            "task": "G",
            "generated_assistant": '{"interpretation_ko":"이 말은 지금은 물러서라고 판단한다.","interpretation_en":"This means it is time to withdraw.","action_tendency":"retreat","confidence":0.7,"register":"hao","misinterpretation_type":"cautious_reversal","temperament_bias":"melancholic caution"}',
            "raw_generated_assistant": '{"interpretation_ko":"이 말은 지금은 물러서라고 판단한다.","interpretation_en":"This means it is time to withdraw.","action_tendency":"retreat","confidence":0.7,"register":"hao","misinterpretation_type":"cautious_reversal","temperament_bias":"melancholic caution"}',
            "structured_attempt_count": 1,
            "structured_repair_applied": False,
            "structured_decoding": {"enabled": False, "supported": False, "used_mode": "none"},
        },
    ]

    report = generate_report(samples)

    assert report["extra_key_count"] == 1
    assert report["json_parse_failure_rate"] == 0.0
    assert report["retry_rate"] == 0.5
    assert report["structured_success_rate"] == 0.5
    assert report["extra_key_rate"] == 0.5
    assert report["enum_drift_rate"] == 0.0
    assert report["repair_applied_rate"] == 0.5
    assert report["constrained_decoding_used_rate"] == 0.0


def test_cli_writes_analysis_report(tmp_path: Path) -> None:
    sample_path = tmp_path / "sample_generations.jsonl"
    report_path = tmp_path / "analysis_report.json"
    write_jsonl(
        sample_path,
        [
            {
                "task": "E",
                "generated_assistant": "{\"action_id\":0,\"confidence\":0.9,\"hint_ko\":\"곧 달아났다\",\"hint_en\":\"fled\",\"personality_reasoning\":\"high_HA\"}",
            },
            {
                "task": "G",
                "generated_assistant": "{\"interpretation_ko\":\"This means we should advance immediately.\",\"interpretation_en\":\"This means we should advance immediately.\",\"action_tendency\":\"mobilize\",\"confidence\":0.9,\"register\":\"hao\",\"misinterpretation_type\":\"overconfident_literal\",\"temperament_bias\":\"action oriented\"}",
            },
        ],
    )

    result = subprocess.run(
        [sys.executable, "tools/generation_analyzer.py", str(sample_path), "--output", str(report_path), "--pretty"],
        cwd="/Users/rexxa/github/worldsim-training",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_samples"] == 2
    assert report["language_drift_count"] == 1


def test_recommend_next_action_prefers_structure_failures() -> None:
    from tools.generation_analyzer import recommend_next_action

    recommendation = recommend_next_action(
        {
            "malformed_json_count": 1,
            "truncation_count": 0,
            "enum_drift_count": 2,
            "language_drift_count": 0,
            "semantic_low_quality_count": 0,
            "semantic_drift_count": 0,
        }
    )

    assert recommendation["status"] == "structure_failure"
    assert "generation-time fix" in recommendation["recommended_next_action"]


def test_recommend_next_action_flags_prompt_leakage() -> None:
    from tools.generation_analyzer import recommend_next_action

    recommendation = recommend_next_action(
        {
            "malformed_json_count": 0,
            "truncation_count": 0,
            "enum_drift_count": 0,
            "prompt_leakage_count": 2,
            "language_drift_count": 0,
            "semantic_low_quality_count": 0,
            "semantic_drift_count": 0,
        }
    )

    assert recommendation["status"] == "prompt_leakage_issue"
    assert "prompt leakage" in recommendation["recommended_next_action"]
