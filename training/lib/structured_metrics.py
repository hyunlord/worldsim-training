from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GenerationAttemptMetrics:
    task_id: str
    attempt_number: int
    raw_length: int
    repairs_applied: list[str] = field(default_factory=list)
    keys_removed: list[str] = field(default_factory=list)
    enums_normalized: list[str] = field(default_factory=list)
    json_parse_success: bool = False
    schema_validation_success: bool = False
    validation_error: Optional[str] = None
    overall_success: bool = False
    retry_exhausted: bool = False


@dataclass
class BatchMetrics:
    total_attempts: int = 0
    total_successes: int = 0
    total_failures: int = 0
    json_parse_failures: int = 0
    schema_validation_failures: int = 0
    repair_applied_count: int = 0
    repairs_by_type: dict[str, int] = field(default_factory=dict)
    keys_removed_count: int = 0
    removed_keys_frequency: dict[str, int] = field(default_factory=dict)
    enums_normalized_count: int = 0
    first_attempt_success: int = 0
    required_retry: int = 0
    max_retries_exhausted: int = 0
    per_task: dict[str, dict[str, int]] = field(default_factory=dict)
    _unique_samples: int = field(default=0, repr=False)
    _unique_successes: int = field(default=0, repr=False)

    def record(self, attempt: GenerationAttemptMetrics) -> None:
        self.total_attempts += 1

        if attempt.overall_success:
            self.total_successes += 1
        else:
            self.total_failures += 1

        if not attempt.json_parse_success:
            self.json_parse_failures += 1

        if not attempt.schema_validation_success and attempt.json_parse_success:
            self.schema_validation_failures += 1

        if attempt.repairs_applied:
            self.repair_applied_count += 1
            for repair in attempt.repairs_applied:
                self.repairs_by_type[repair] = self.repairs_by_type.get(repair, 0) + 1

        if attempt.keys_removed:
            self.keys_removed_count += 1
            for key in attempt.keys_removed:
                self.removed_keys_frequency[key] = self.removed_keys_frequency.get(key, 0) + 1

        if attempt.enums_normalized:
            self.enums_normalized_count += 1

        if attempt.attempt_number == 1 and attempt.overall_success:
            self.first_attempt_success += 1
        elif attempt.attempt_number > 1 and attempt.overall_success:
            self.required_retry += 1

        if attempt.retry_exhausted:
            self.max_retries_exhausted += 1

        task_metrics = self.per_task.setdefault(attempt.task_id, {"total": 0, "success": 0, "failure": 0})
        task_metrics["total"] += 1
        if attempt.overall_success:
            task_metrics["success"] += 1
        else:
            task_metrics["failure"] += 1

    def record_sample_outcome(self, success: bool) -> None:
        self._unique_samples += 1
        if success:
            self._unique_successes += 1

    @property
    def structured_success_rate(self) -> float:
        return (self.total_successes / self.total_attempts) if self.total_attempts else 0.0

    @property
    def json_parse_failure_rate(self) -> float:
        return (self.json_parse_failures / self.total_attempts) if self.total_attempts else 0.0

    @property
    def repair_applied_rate(self) -> float:
        return (self.repair_applied_count / self.total_attempts) if self.total_attempts else 0.0

    @property
    def extra_key_rate(self) -> float:
        return (self.keys_removed_count / self.total_attempts) if self.total_attempts else 0.0

    @property
    def per_sample_success_rate(self) -> float:
        return (self._unique_successes / self._unique_samples) if self._unique_samples else 0.0

    def summary(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "json_parse_failures": self.json_parse_failures,
            "schema_validation_failures": self.schema_validation_failures,
            "repair_applied_count": self.repair_applied_count,
            "repairs_by_type": dict(sorted(self.repairs_by_type.items())),
            "keys_removed_count": self.keys_removed_count,
            "removed_keys_frequency": dict(sorted(self.removed_keys_frequency.items())),
            "enums_normalized_count": self.enums_normalized_count,
            "first_attempt_success": self.first_attempt_success,
            "required_retry": self.required_retry,
            "max_retries_exhausted": self.max_retries_exhausted,
            "unique_samples": self._unique_samples,
            "unique_successes": self._unique_successes,
            "per_task": self.per_task,
            "structured_success_rate": self.structured_success_rate,
            "per_sample_success_rate": self.per_sample_success_rate,
            "json_parse_failure_rate": self.json_parse_failure_rate,
            "repair_applied_rate": self.repair_applied_rate,
            "extra_key_rate": self.extra_key_rate,
        }

    def print_report(self) -> None:
        summary = self.summary()
        print("Structured generation metrics")
        for key in (
            "total_attempts",
            "total_successes",
            "total_failures",
            "json_parse_failures",
            "schema_validation_failures",
            "repair_applied_count",
            "keys_removed_count",
            "enums_normalized_count",
            "first_attempt_success",
            "required_retry",
            "max_retries_exhausted",
            "unique_samples",
            "unique_successes",
            "structured_success_rate",
            "per_sample_success_rate",
            "json_parse_failure_rate",
            "repair_applied_rate",
            "extra_key_rate",
        ):
            print(f"- {key}: {summary[key]}")


def save_metrics(metrics: BatchMetrics, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
