from __future__ import annotations

import json

from training.lib.structured_metrics import BatchMetrics, GenerationAttemptMetrics


def test_batch_metrics_records_single_success() -> None:
    metrics = BatchMetrics()
    metrics.record(
        GenerationAttemptMetrics(
            task_id="A",
            attempt_number=1,
            raw_length=100,
            json_parse_success=True,
            schema_validation_success=True,
            overall_success=True,
        )
    )

    assert metrics.total_attempts == 1
    assert metrics.total_successes == 1
    assert metrics.structured_success_rate == 1.0


def test_batch_metrics_records_parse_failure() -> None:
    metrics = BatchMetrics()
    metrics.record(
        GenerationAttemptMetrics(
            task_id="B",
            attempt_number=1,
            raw_length=50,
            json_parse_success=False,
            schema_validation_success=False,
            validation_error="JSON parse error",
            overall_success=False,
        )
    )

    assert metrics.json_parse_failures == 1
    assert metrics.total_failures == 1


def test_batch_metrics_records_validation_failure() -> None:
    metrics = BatchMetrics()
    metrics.record(
        GenerationAttemptMetrics(
            task_id="C",
            attempt_number=1,
            raw_length=80,
            json_parse_success=True,
            schema_validation_success=False,
            validation_error="speaker_role missing",
            overall_success=False,
        )
    )

    assert metrics.schema_validation_failures == 1
    assert metrics.total_failures == 1


def test_batch_metrics_calculates_rates_and_per_task() -> None:
    metrics = BatchMetrics()
    metrics.record(
        GenerationAttemptMetrics(
            task_id="A",
            attempt_number=1,
            raw_length=100,
            repairs_applied=["fence_strip"],
            keys_removed=["schema_explanation"],
            enums_normalized=["register: HAO -> hao"],
            json_parse_success=True,
            schema_validation_success=True,
            overall_success=True,
        )
    )
    metrics.record(
        GenerationAttemptMetrics(
            task_id="A",
            attempt_number=2,
            raw_length=90,
            json_parse_success=True,
            schema_validation_success=False,
            validation_error="invalid enum",
            overall_success=False,
            retry_exhausted=True,
        )
    )

    assert metrics.repair_applied_rate == 0.5
    assert metrics.extra_key_rate == 0.5
    assert metrics.per_task["A"] == {"total": 2, "success": 1, "failure": 1}
    assert metrics.max_retries_exhausted == 1


def test_batch_metrics_summary_is_json_serializable() -> None:
    summary = BatchMetrics().summary()
    json.dumps(summary)
    assert summary["structured_success_rate"] == 0.0


def test_batch_metrics_zero_attempt_edge_case() -> None:
    metrics = BatchMetrics()
    assert metrics.structured_success_rate == 0.0
    assert metrics.json_parse_failure_rate == 0.0
    assert metrics.repair_applied_rate == 0.0
    assert metrics.extra_key_rate == 0.0
