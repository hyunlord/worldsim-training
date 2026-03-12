from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def extract_metrics(output_dir: str) -> dict[str, Any]:
    base = Path(output_dir)

    metrics_path = base / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    structured = metrics.get("structured_metrics", {})

    report_path = base / "analysis_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    sample_path = base / "sample_generations.jsonl"
    samples: list[dict[str, Any]] = []
    if sample_path.exists():
        with sample_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

    repair_breakdown: dict[str, int] = {}
    keys_removed_all: dict[str, int] = {}
    for sample in samples:
        for action in sample.get("structured_repair_actions") or []:
            if isinstance(action, str):
                repair_breakdown[action] = repair_breakdown.get(action, 0) + 1
                continue
            if not isinstance(action, dict):
                continue
            kind = str(action.get("kind", "unknown"))
            repair_breakdown[kind] = repair_breakdown.get(kind, 0) + 1
            if kind == "filter_extra_keys":
                for key in action.get("removed_keys", []):
                    key_name = str(key)
                    keys_removed_all[key_name] = keys_removed_all.get(key_name, 0) + 1

    return {
        "structured_metrics": structured,
        "analysis_report_summary": {
            "overall_status": report.get("overall_status"),
            "structured_success_rate": report.get("structured_success_rate"),
            "malformed_json_count": report.get("malformed_json_count"),
            "fenced_json_count": report.get("fenced_json_count"),
            "enum_drift_total": report.get("enum_drift_total"),
            "extra_keys_count": report.get("extra_key_count", report.get("extra_keys_count")),
            "semantic_low_quality_count": report.get("semantic_low_quality_count"),
        },
        "repair_breakdown": dict(sorted(repair_breakdown.items(), key=lambda item: (-item[1], item[0]))),
        "top_removed_keys": dict(sorted(keys_removed_all.items(), key=lambda item: (-item[1], item[0]))[:10]),
        "total_samples": len(samples),
    }


def print_report(data: dict[str, Any]) -> None:
    sm = data.get("structured_metrics", {})
    ar = data.get("analysis_report_summary", {})

    print("=" * 60)
    print("GUARDRAIL VERIFICATION REPORT")
    print("=" * 60)

    print("\n--- Structured Metrics (from BatchMetrics) ---")
    for key in [
        "structured_success_rate",
        "json_parse_failure_rate",
        "repair_applied_rate",
        "extra_key_rate",
        "total_attempts",
        "total_successes",
        "total_failures",
        "first_attempt_success",
        "required_retry",
        "max_retries_exhausted",
    ]:
        val = sm.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        print(f"  {key}: {val}")

    print("\n--- Per-Task Breakdown ---")
    for task_id, task_data in sorted(sm.get("per_task", {}).items()):
        total = task_data.get("total", 0)
        success = task_data.get("success", 0)
        rate = f"{success / total:.2%}" if total > 0 else "N/A"
        print(f"  Task {task_id}: {success}/{total} ({rate})")

    print("\n--- Repairs Applied ---")
    for repair_type, count in sorted(sm.get("repairs_by_type", {}).items(), key=lambda item: (-item[1], item[0])):
        print(f"  {repair_type}: {count}")

    print("\n--- Top Removed Keys (Schema Leakage) ---")
    for key, count in data.get("top_removed_keys", {}).items():
        print(f"  {key}: {count}")

    print("\n--- Analysis Report Summary ---")
    for key, val in ar.items():
        print(f"  {key}: {val}")

    print("\n--- Decision ---")
    ssr = sm.get("structured_success_rate", 0)
    if isinstance(ssr, (int, float)):
        if ssr >= 0.95:
            print("  VERDICT: PASS — guardrails achieved 95%+ structured success")
        elif ssr >= 0.85:
            print("  VERDICT: PARTIAL — prompt hardening next (85-95%)")
        else:
            print("  VERDICT: INSUFFICIENT — constrained decoding needed (<85%)")
    else:
        print("  VERDICT: UNKNOWN — could not parse structured_success_rate")

    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(f"Usage: {Path(sys.argv[0]).name} <output_dir>", file=sys.stderr)
        return 1

    data = extract_metrics(args[0])
    print_report(data)
    json_out = Path(args[0]) / "guardrail_verification_report.json"
    json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nJSON report saved to: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
