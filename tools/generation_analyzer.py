from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import read_jsonl
from training.lib.qlora_smoke import (
    VALID_ACTION_TENDENCIES,
    VALID_EMOTIONS,
    VALID_MISINTERPRETATION_TYPES,
    VALID_REGISTERS,
    VALID_TRANSITION_TYPES,
    _enum_drift_issues,
    _normalize_enum_candidate,
    _parse_json_payload,
    analyze_sample_generation,
    strip_json_fence,
    validate_g_semantics,
)


FAILURE_CATEGORIES = {
    "ok",
    "malformed_json",
    "truncation",
    "fenced_json",
    "trailing_text",
    "enum_drift",
    "language_drift",
    "semantic_low_quality",
    "semantic_drift",
}


def load_samples(path: str | Path) -> list[dict]:
    return read_jsonl(Path(path))


def validate_json_text(text: str) -> dict[str, Any]:
    raw_text = str(text).strip()
    stripped_text = strip_json_fence(raw_text)
    raw_payload, raw_error = _parse_json_payload(raw_text)
    stripped_payload, stripped_error = _parse_json_payload(stripped_text)
    raw_parseable = raw_payload is not None
    stripped_parseable = stripped_payload is not None
    fenced_json = stripped_text != raw_text
    recoverable_fenced_json = fenced_json and not raw_parseable and stripped_parseable
    malformed_json = not stripped_parseable

    failure_detail = classify_json_failure(text)
    return {
        "raw_text": raw_text,
        "stripped_text": stripped_text,
        "raw_parseable_json": raw_parseable,
        "fence_stripped_parseable_json": stripped_parseable,
        "fenced_json": fenced_json,
        "recoverable_fenced_json": recoverable_fenced_json,
        "malformed_json": malformed_json,
        "raw_json_parse_error": raw_error,
        "stripped_json_parse_error": stripped_error,
        "payload": raw_payload if raw_payload is not None else stripped_payload,
        "json_failure": failure_detail,
    }


def classify_json_failure(text: str) -> dict[str, Any]:
    raw_text = str(text).strip()
    stripped_text = strip_json_fence(raw_text)
    raw_payload, raw_error = _parse_json_payload(raw_text)
    stripped_payload, stripped_error = _parse_json_payload(stripped_text)
    raw_parseable = raw_payload is not None
    stripped_parseable = stripped_payload is not None
    fenced_json = stripped_text != raw_text

    category = "ok"
    details: dict[str, Any] = {
        "raw_parseable_json": raw_parseable,
        "fence_stripped_parseable_json": stripped_parseable,
        "fenced_json": fenced_json,
        "raw_json_parse_error": raw_error,
        "stripped_json_parse_error": stripped_error,
    }

    if fenced_json and stripped_parseable and not raw_parseable:
        category = "fenced_json"
    elif not stripped_text:
        category = "malformed_json"
        details["pattern"] = "empty_output"
    elif stripped_text[0] not in "{[":
        category = "malformed_json"
        details["pattern"] = "non_json_prefix"
    else:
        decoder = json.JSONDecoder()
        try:
            _, end_index = decoder.raw_decode(stripped_text)
            trailing = stripped_text[end_index:].strip()
            if trailing:
                category = "trailing_text"
                details["pattern"] = "extra_text_after_valid_json"
                details["trailing_text"] = trailing
        except json.JSONDecodeError as exc:
            details["json_error_msg"] = exc.msg
            details["json_error_pos"] = exc.pos
            unbalanced_object = stripped_text.count("{") > stripped_text.count("}")
            unbalanced_array = stripped_text.count("[") > stripped_text.count("]")
            missing_terminal = not stripped_text.endswith(("}", "]"))
            if unbalanced_object or unbalanced_array or missing_terminal:
                category = "truncation"
                details["pattern"] = "partial_object"
            elif 'Expecting \',\' delimiter' in exc.msg:
                category = "malformed_json"
                details["pattern"] = "invalid_comma_placement"
            elif "Unterminated string" in exc.msg:
                category = "malformed_json"
                details["pattern"] = "broken_quotes"
            else:
                category = "malformed_json"
                details["pattern"] = "invalid_syntax"

    details["category"] = category
    return details


def _expected_enum_values(task: str, field_name: str) -> list[str]:
    if field_name in {"emotion_expressed", "emotion", "previous_emotion"}:
        return sorted(VALID_EMOTIONS)
    if field_name == "register":
        return sorted(VALID_REGISTERS)
    if field_name == "transition_type":
        return sorted(VALID_TRANSITION_TYPES)
    if field_name == "action_tendency":
        return sorted(VALID_ACTION_TENDENCIES)
    if field_name == "misinterpretation_type":
        return sorted(VALID_MISINTERPRETATION_TYPES)
    return []


def _drift_type(actual_value: str, expected_values: Sequence[str]) -> str:
    if any("\uac00" <= char <= "\ud7a3" for char in actual_value):
        return "language_mismatch"

    normalized = _normalize_enum_candidate(actual_value)
    expected_set = set(expected_values)
    if normalized in expected_set:
        stripped = actual_value.strip()
        if stripped.lower() == normalized:
            return "casing_only"
        return "spacing_or_hyphen"

    if any(token in actual_value.lower() for token in ("fear", "anger", "trust", "defend", "wait", "mobilize", "retreat", "celebrate", "mourn")):
        return "semantic_mismatch"
    if len(actual_value.split()) > 1:
        return "synonym_or_paraphrase"
    return "unknown_value"


def check_enum_drift(sample_or_data: Mapping[str, Any]) -> list[dict[str, Any]]:
    task = str(sample_or_data.get("task", "unknown"))
    if "generated_assistant" in sample_or_data:
        payload = validate_json_text(str(sample_or_data.get("generated_assistant", ""))).get("payload")
    else:
        payload = sample_or_data

    issues = _enum_drift_issues(task, payload)
    results: list[dict[str, Any]] = []
    for field_name, actual_value in issues:
        expected_values = _expected_enum_values(task, field_name)
        results.append(
            {
                "field_name": field_name,
                "expected_enum_set": expected_values,
                "actual_value": actual_value,
                "drift_type": _drift_type(actual_value, expected_values),
            }
        )
    return results


def analyze_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    task = str(sample.get("task", "unknown"))
    json_analysis = validate_json_text(str(sample.get("generated_assistant", "")))
    enum_drift = check_enum_drift(sample)
    base = analyze_sample_generation(sample)
    semantic = validate_g_semantics(sample) if task == "G" else None

    primary_category = "ok"
    if json_analysis["json_failure"]["category"] == "fenced_json":
        primary_category = "fenced_json"
    elif json_analysis["json_failure"]["category"] == "truncation":
        primary_category = "truncation"
    elif json_analysis["json_failure"]["category"] == "trailing_text":
        primary_category = "trailing_text"
    elif json_analysis["malformed_json"]:
        primary_category = "malformed_json"
    elif enum_drift:
        primary_category = "enum_drift"
    elif semantic and semantic["semantic_status"] == "LANGUAGE_DRIFT":
        primary_category = "language_drift"
    elif semantic and semantic["semantic_status"] == "SEMANTIC_DRIFT":
        primary_category = "semantic_drift"
    elif semantic and semantic["semantic_status"] == "LOW_QUALITY":
        primary_category = "semantic_low_quality"

    return {
        "task": task,
        "primary_category": primary_category,
        "json_analysis": json_analysis,
        "enum_drift": enum_drift,
        "semantic": semantic,
        "base_analysis": base,
        "generated_assistant": str(sample.get("generated_assistant", "")),
        "raw_generated_assistant": sample.get("raw_generated_assistant"),
    }


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    analyses = [analyze_sample(sample) for sample in samples]
    counts = Counter(analysis["primary_category"] for analysis in analyses)
    affected_tasks: dict[str, Counter[str]] = defaultdict(Counter)
    for analysis in analyses:
        if analysis["primary_category"] != "ok":
            affected_tasks[analysis["task"]][analysis["primary_category"]] += 1

    overall_status = "structurally_usable"
    if any(counts.get(category, 0) > 0 for category in ("malformed_json", "truncation", "fenced_json", "trailing_text")):
        overall_status = "structure_failure"
    elif counts.get("enum_drift", 0) > 0:
        overall_status = "enum_instability"
    elif any(counts.get(category, 0) > 0 for category in ("language_drift", "semantic_low_quality", "semantic_drift")):
        overall_status = "semantic_quality_issue"

    return {
        "total_samples": len(samples),
        "counts_by_failure_category": dict(sorted(counts.items())),
        "malformed_json_count": counts.get("malformed_json", 0),
        "truncation_count": counts.get("truncation", 0),
        "fenced_json_count": counts.get("fenced_json", 0),
        "trailing_text_count": counts.get("trailing_text", 0),
        "enum_drift_count": counts.get("enum_drift", 0),
        "language_drift_count": counts.get("language_drift", 0),
        "semantic_low_quality_count": counts.get("semantic_low_quality", 0),
        "semantic_drift_count": counts.get("semantic_drift", 0),
        "affected_tasks_summary": {
            task: dict(sorted(category_counts.items()))
            for task, category_counts in sorted(affected_tasks.items())
        },
        "overall_status": overall_status,
        "analyses": analyses,
    }


def generate_report(samples: Sequence[Mapping[str, Any]], *, examples_per_category: int = 3) -> dict[str, Any]:
    summary = summarize_samples(samples)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for analysis in summary["analyses"]:
        category = analysis["primary_category"]
        if category == "ok" or len(examples[category]) >= examples_per_category:
            continue
        examples[category].append(
            {
                "task": analysis["task"],
                "generated_assistant": analysis["generated_assistant"],
                "json_failure": analysis["json_analysis"]["json_failure"],
                "enum_drift": analysis["enum_drift"],
                "semantic": analysis["semantic"],
            }
        )

    return {
        "total_samples": summary["total_samples"],
        "counts_by_failure_category": summary["counts_by_failure_category"],
        "malformed_json_count": summary["malformed_json_count"],
        "truncation_count": summary["truncation_count"],
        "fenced_json_count": summary["fenced_json_count"],
        "trailing_text_count": summary["trailing_text_count"],
        "enum_drift_count": summary["enum_drift_count"],
        "language_drift_count": summary["language_drift_count"],
        "semantic_low_quality_count": summary["semantic_low_quality_count"],
        "semantic_drift_count": summary["semantic_drift_count"],
        "affected_tasks_summary": summary["affected_tasks_summary"],
        "overall_status": summary["overall_status"],
        "example_failures": dict(sorted(examples.items())),
    }


def recommend_next_action(report: Mapping[str, Any]) -> dict[str, str]:
    malformed_json_count = int(report.get("malformed_json_count", 0) or 0)
    truncation_count = int(report.get("truncation_count", 0) or 0)
    enum_drift_count = int(report.get("enum_drift_count", 0) or 0)
    language_drift_count = int(report.get("language_drift_count", 0) or 0)
    semantic_low_quality_count = int(report.get("semantic_low_quality_count", 0) or 0)
    semantic_drift_count = int(report.get("semantic_drift_count", 0) or 0)

    if malformed_json_count > 0 or truncation_count > 0:
        return {
            "status": "structure_failure",
            "recommended_next_action": "Apply a generation-time fix before a longer smoke run.",
        }
    if enum_drift_count > 0:
        return {
            "status": "enum_instability",
            "recommended_next_action": "Stabilize task-specific enum generation before the next smoke run.",
        }
    if language_drift_count > 0 or semantic_low_quality_count > 0 or semantic_drift_count > 0:
        return {
            "status": "semantic_quality_issue",
            "recommended_next_action": "Investigate semantic quality before escalating training duration.",
        }
    return {
        "status": "structurally_usable",
        "recommended_next_action": "Proceed to the next longer smoke or a more realistic training step.",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze WorldSim sample_generations.jsonl artifacts.")
    parser.add_argument("sample_path", help="Path to sample_generations.jsonl")
    parser.add_argument("--output", help="Where to write analysis_report.json")
    parser.add_argument("--examples-per-category", type=int, default=3, help="Maximum failure examples per category")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print summary JSON to stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    sample_path = Path(args.sample_path)
    output_path = Path(args.output) if args.output else sample_path.with_name("analysis_report.json")

    samples = load_samples(sample_path)
    report = generate_report(samples, examples_per_category=args.examples_per_category)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None),
        encoding="utf-8",
    )

    stdout_payload = {
        "total_samples": report["total_samples"],
        "overall_status": report["overall_status"],
        "counts_by_failure_category": report["counts_by_failure_category"],
        "recommended_next_action": recommend_next_action(report),
        "output": str(output_path),
    }
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
