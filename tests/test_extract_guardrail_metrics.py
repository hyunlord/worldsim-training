from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_extract_metrics_reads_valid_directory(tmp_path: Path) -> None:
    module = load_module("extract_guardrail_metrics", Path.cwd() / "scripts/extract_guardrail_metrics.py")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "structured_metrics": {
                    "structured_success_rate": 0.92,
                    "per_task": {"A": {"total": 2, "success": 2, "failure": 0}},
                    "repairs_by_type": {"fence_strip": 1},
                }
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "analysis_report.json").write_text(
        json.dumps(
            {
                "overall_status": "pass",
                "structured_success_rate": 0.92,
                "malformed_json_count": 1,
                "extra_key_count": 2,
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        output_dir / "sample_generations.jsonl",
        [
            {
                "structured_repair_actions": [
                    {"kind": "filter_extra_keys", "removed_keys": ["schema_explanation", "emoji"]},
                    {"kind": "fence_strip"},
                ]
            }
        ],
    )

    payload = module.extract_metrics(str(output_dir))

    assert payload["structured_metrics"]["structured_success_rate"] == 0.92
    assert payload["analysis_report_summary"]["overall_status"] == "pass"
    assert payload["repair_breakdown"]["filter_extra_keys"] == 1
    assert payload["repair_breakdown"]["fence_strip"] == 1
    assert payload["top_removed_keys"] == {"emoji": 1, "schema_explanation": 1}
    assert payload["total_samples"] == 1


def test_extract_metrics_handles_missing_metrics_json(tmp_path: Path) -> None:
    module = load_module("extract_guardrail_metrics", Path.cwd() / "scripts/extract_guardrail_metrics.py")
    output_dir = tmp_path / "run"
    output_dir.mkdir()

    payload = module.extract_metrics(str(output_dir))

    assert payload["structured_metrics"] == {}
    assert payload["analysis_report_summary"]["overall_status"] is None
    assert payload["repair_breakdown"] == {}


def test_extract_metrics_handles_missing_analysis_report(tmp_path: Path) -> None:
    module = load_module("extract_guardrail_metrics", Path.cwd() / "scripts/extract_guardrail_metrics.py")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "metrics.json").write_text(json.dumps({"structured_metrics": {"structured_success_rate": 1.0}}), encoding="utf-8")

    payload = module.extract_metrics(str(output_dir))

    assert payload["structured_metrics"]["structured_success_rate"] == 1.0
    assert payload["analysis_report_summary"]["overall_status"] is None


def test_extract_metrics_handles_empty_samples(tmp_path: Path) -> None:
    module = load_module("extract_guardrail_metrics", Path.cwd() / "scripts/extract_guardrail_metrics.py")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "sample_generations.jsonl").write_text("", encoding="utf-8")

    payload = module.extract_metrics(str(output_dir))

    assert payload["total_samples"] == 0
    assert payload["top_removed_keys"] == {}


def test_print_report_does_not_crash(tmp_path: Path, capsys) -> None:
    module = load_module("extract_guardrail_metrics", Path.cwd() / "scripts/extract_guardrail_metrics.py")
    data = {
        "structured_metrics": {
            "structured_success_rate": 0.92,
            "json_parse_failure_rate": 0.04,
            "repair_applied_rate": 0.30,
            "extra_key_rate": 0.10,
            "total_attempts": 50,
            "total_successes": 46,
            "total_failures": 4,
            "first_attempt_success": 40,
            "required_retry": 6,
            "max_retries_exhausted": 1,
            "per_task": {"A": {"total": 10, "success": 9, "failure": 1}},
            "repairs_by_type": {"fence_strip": 3},
        },
        "analysis_report_summary": {"overall_status": "pass", "structured_success_rate": 0.92},
        "top_removed_keys": {"schema_explanation": 5},
    }

    module.print_report(data)
    captured = capsys.readouterr()

    assert "GUARDRAIL VERIFICATION REPORT" in captured.out
    assert "VERDICT: PARTIAL" in captured.out
