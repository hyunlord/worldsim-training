"""Structured generation guardrails for WorldSim sample generation.

This module keeps the existing generation contract intact while splitting the
implementation into explicit guardrail layers:

1. prompt contract (handled by caller)
2. deterministic decoding configuration (handled by caller/backend)
3. lightweight JSON repair
4. schema-aware key sanitization + enum normalization
5. validation feedback retries
6. metrics capture
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from training.lib.json_repair import repair_json as _repair_json_text
from training.lib.json_sanitize import TASK_ALLOWED_KEYS_REGISTRY, normalize_enum_values, sanitize_keys
from training.lib.structured_metrics import BatchMetrics, GenerationAttemptMetrics


DEFAULT_MAX_RETRY = 3
STRUCTURED_GENERATION_DEFAULTS = {
    "temperature": 0.0,
    "do_sample": False,
    "top_p": 1.0,
}
TASK_MAX_NEW_TOKENS: dict[str, int] = {
    "A": 256,
    "B": 256,
    "C": 384,
    "E": 256,
    "F": 384,
    "G": 384,
    "H": 384,
}


@dataclass(frozen=True, slots=True)
class StructuredConstraint:
    schema_name: str
    mode: str
    allowed_keys: tuple[str, ...]
    enum_fields: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RepairResult:
    repaired: bool
    repair_actions: list[dict[str, Any]]
    repaired_text: str


@dataclass(frozen=True, slots=True)
class StructuredDecodingMetadata:
    requested_mode: str
    used_mode: str
    enabled: bool
    supported: bool
    reason: str | None = None


@dataclass(slots=True)
class StructuredGenerationAttempt:
    attempt_index: int
    prompt: str
    raw_output: str
    candidate_output: str
    json_error: str | None
    validation_error: str | None
    error_kind: str | None
    normalization: str | None = None
    normalization_details: list[dict[str, str]] | None = None
    repair_actions: list[dict[str, Any]] = field(default_factory=list)
    keys_removed: list[str] = field(default_factory=list)
    enum_normalizations: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructuredGenerationResult:
    model: BaseModel
    payload: dict[str, Any]
    raw_output: str
    candidate_output: str
    attempts: list[StructuredGenerationAttempt]
    attempt_count: int
    last_error_kind: str | None
    normalization: str | None = None
    normalization_details: list[dict[str, str]] | None = None
    repair_actions: list[dict[str, Any]] = field(default_factory=list)
    structured_decoding: dict[str, Any] | None = None
    attempt_metrics: list[GenerationAttemptMetrics] = field(default_factory=list)


class StructuredGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: list[StructuredGenerationAttempt],
        last_output: str,
        last_raw_output: str,
        last_error_kind: str | None,
        normalization: str | None = None,
        normalization_details: list[dict[str, str]] | None = None,
        repair_actions: list[dict[str, Any]] | None = None,
        structured_decoding: dict[str, Any] | None = None,
        attempt_metrics: list[GenerationAttemptMetrics] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.attempt_count = len(attempts)
        self.last_output = last_output
        self.last_raw_output = last_raw_output
        self.last_error_kind = last_error_kind
        self.normalization = normalization
        self.normalization_details = normalization_details or []
        self.repair_actions = repair_actions or []
        self.structured_decoding = structured_decoding
        self.attempt_metrics = attempt_metrics or []


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        message = error.get("msg") or error.get("type")
        parts.append(f"{location}:{message}")
    return ", ".join(parts)


def _literal_choices(annotation: Any) -> tuple[str, ...]:
    origin = get_origin(annotation)
    if origin is None:
        return ()
    if str(origin).endswith("Literal"):
        return tuple(choice for choice in get_args(annotation) if isinstance(choice, str))

    nested_choices: list[str] = []
    for arg in get_args(annotation):
        nested_choices.extend(_literal_choices(arg))
    return tuple(nested_choices)


def _schema_allowed_keys(schema: type[BaseModel]) -> set[str]:
    return {field.alias or field_name for field_name, field in schema.model_fields.items()}


def _schema_enum_fields(schema: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    enum_fields: dict[str, tuple[str, ...]] = {}
    for field_name, field in schema.model_fields.items():
        choices = _literal_choices(field.annotation)
        if choices:
            enum_fields[field.alias or field_name] = choices
    return enum_fields


def _infer_task_id(schema: type[BaseModel], task_id: str) -> str:
    if task_id:
        return task_id
    if schema.__name__.startswith("Task") and schema.__name__.endswith("Output"):
        return schema.__name__[4:-6]
    return schema.__name__


def build_structured_constraint(schema: type[BaseModel], *, mode: str = "json_schema") -> StructuredConstraint:
    return StructuredConstraint(
        schema_name=schema.__name__,
        mode=mode,
        allowed_keys=tuple(sorted(_schema_allowed_keys(schema))),
        enum_fields=_schema_enum_fields(schema),
    )


def resolve_structured_decoding(
    constraint: StructuredConstraint | None,
    *,
    backend: str = "transformers",
) -> dict[str, Any]:
    if constraint is None:
        metadata = StructuredDecodingMetadata(
            requested_mode="none",
            used_mode="none",
            enabled=False,
            supported=False,
            reason="No structured constraint requested.",
        )
    else:
        metadata = StructuredDecodingMetadata(
            requested_mode=constraint.mode,
            used_mode="none",
            enabled=False,
            supported=False,
            reason=f"{backend} backend has no native JSON grammar support wired in this repo.",
        )

    return {
        "requested_mode": metadata.requested_mode,
        "used_mode": metadata.used_mode,
        "enabled": metadata.enabled,
        "supported": metadata.supported,
        "reason": metadata.reason,
    }


def repair_json_candidate(raw_text: str) -> RepairResult:
    repaired_text, repair_names = _repair_json_text(raw_text)
    repair_actions = [{"kind": repair_name} for repair_name in repair_names]
    return RepairResult(
        repaired=bool(repair_actions),
        repair_actions=repair_actions,
        repaired_text=repaired_text,
    )


def repair_json_legacy(raw_text: str) -> dict[str, Any]:
    repaired = repair_json_candidate(raw_text)
    return {"text": repaired.repaired_text, "repair_actions": repaired.repair_actions}


# Backward-compatible export for existing tests and callers.
repair_json = repair_json_legacy


def _categorize_validation_errors(exc: ValidationError) -> list[str]:
    categories: list[str] = []
    for error in exc.errors():
        error_type = str(error.get("type") or "")
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        input_value = error.get("input")

        if error_type == "extra_forbidden":
            categories.append(f'extra_key: "{location}"')
        elif "literal" in error_type:
            categories.append(f'invalid_enum: "{location}" -> {input_value!r}')
        elif error_type == "missing":
            categories.append(f'missing_required_field: "{location}"')
        else:
            categories.append(f'schema_validation_error: "{location}" -> {error.get("msg")}')
    return categories


def _build_retry_prompt(
    prompt: str,
    *,
    schema_name: str,
    attempt_index: int,
    last_error_kind: str | None,
    detail: str | None,
    bad_output: str,
    task_id: str,
    removed_keys: list[str],
    enum_changes: list[str],
) -> str:
    retry_feedback_parts = [
        "The previous output failed validation.",
        f"Problem type: {'json_parse_error' if last_error_kind == 'json' else 'schema_validation_error'}",
    ]

    if detail:
        retry_feedback_parts.append("Problems:")
        retry_feedback_parts.extend(f"- {line}" for line in detail.splitlines() if line.strip())

    if removed_keys:
        retry_feedback_parts.append(
            f'WARNING: The following keys were found in your output but are NOT allowed: {removed_keys}. '
            f"Only output these keys: {sorted(TASK_ALLOWED_KEYS_REGISTRY.get(task_id, []))}"
        )

    if enum_changes:
        retry_feedback_parts.append(
            f"WARNING: Some enum values were normalized: {enum_changes}. Use exact enum values from the allowed list."
        )

    retry_feedback_parts.extend(
        [
            "Fix the JSON so it matches the schema exactly.",
            "Return ONLY corrected JSON.",
            "Do not add new fields.",
            "Do not copy instructions, schema descriptions, examples, or rule text into the JSON output.",
            "All string values must use double quotes.",
            f"This correction must satisfy schema {schema_name}.",
            "Here is your previous JSON:",
            bad_output,
        ]
    )

    return f"{prompt}\n\n" + "\n".join(retry_feedback_parts)


def generate_structured(
    llm: Any,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retry: int = DEFAULT_MAX_RETRY,
    output_normalizer: Any | None = None,
    structured_constraint: StructuredConstraint | None = None,
    decoding_backend: str = "transformers",
    allow_key_filtering: bool = True,
    allow_enum_correction: bool = True,
    task_id: str = "",
    metrics_collector: BatchMetrics | None = None,
) -> StructuredGenerationResult:
    attempts: list[StructuredGenerationAttempt] = []
    attempt_metrics_rows: list[GenerationAttemptMetrics] = []
    last_error_kind: str | None = None
    last_detail: str | None = None
    last_output = ""
    last_raw_output = ""
    last_normalization: str | None = None
    last_normalization_details: list[dict[str, str]] = []
    last_repair_actions: list[dict[str, Any]] = []
    last_removed_keys: list[str] = []
    last_enum_changes: list[str] = []
    structured_decoding = resolve_structured_decoding(structured_constraint, backend=decoding_backend)
    effective_task_id = _infer_task_id(schema, task_id)

    for attempt_index in range(max_retry + 1):
        attempt_prompt = prompt if attempt_index == 0 else _build_retry_prompt(
            prompt,
            schema_name=schema.__name__,
            attempt_index=attempt_index,
            last_error_kind=last_error_kind,
            detail=last_detail,
            bad_output=last_output,
            task_id=effective_task_id,
            removed_keys=last_removed_keys,
            enum_changes=last_enum_changes,
        )
        raw_output = str(llm(attempt_prompt))
        candidate_output = raw_output
        normalization: str | None = None
        normalization_details: list[dict[str, str]] = []
        repair_actions: list[dict[str, Any]] = []
        removed_keys: list[str] = []
        enum_changes: list[str] = []
        attempt_metrics = GenerationAttemptMetrics(
            task_id=effective_task_id,
            attempt_number=attempt_index + 1,
            raw_length=len(raw_output),
        )

        if output_normalizer is not None:
            normalized = output_normalizer(raw_output)
            candidate_output = normalized["text"]
            normalization = normalized.get("normalization")
            normalization_details = list(normalized.get("normalization_details") or [])

        repaired_json = repair_json_candidate(candidate_output)
        candidate_output = repaired_json.repaired_text
        repair_actions.extend(repaired_json.repair_actions)
        attempt_metrics.repairs_applied = [action["kind"] for action in repaired_json.repair_actions]

        last_output = candidate_output
        last_raw_output = raw_output
        last_normalization = normalization
        last_normalization_details = normalization_details
        last_repair_actions = repair_actions

        try:
            payload = json.loads(candidate_output)
            attempt_metrics.json_parse_success = True
        except json.JSONDecodeError as exc:
            last_error_kind = "json"
            last_detail = f"json_parse_error: {exc.msg}"
            attempt_metrics.validation_error = last_detail
            attempt_metrics.retry_exhausted = attempt_index == max_retry
            attempts.append(
                StructuredGenerationAttempt(
                    attempt_index=attempt_index,
                    prompt=attempt_prompt,
                    raw_output=raw_output,
                    candidate_output=candidate_output,
                    json_error=type(exc).__name__,
                    validation_error=None,
                    error_kind=last_error_kind,
                    normalization=normalization,
                    normalization_details=normalization_details,
                    repair_actions=repair_actions,
                    keys_removed=removed_keys,
                    enum_normalizations=enum_changes,
                )
            )
            attempt_metrics_rows.append(attempt_metrics)
            if metrics_collector is not None:
                metrics_collector.record(attempt_metrics)
            continue

        sanitized_payload, removed_keys = sanitize_keys(payload, effective_task_id)
        normalized_payload, enum_changes = normalize_enum_values(sanitized_payload, effective_task_id)

        if allow_key_filtering:
            payload = sanitized_payload
        else:
            removed_keys = []

        if allow_enum_correction:
            payload = normalized_payload if allow_key_filtering else normalize_enum_values(payload, effective_task_id)[0]
        else:
            enum_changes = []

        payload_repairs: list[dict[str, Any]] = []
        if removed_keys:
            payload_repairs.append({"kind": "filter_extra_keys", "removed_keys": removed_keys})
        for enum_change in enum_changes:
            field_name, _, replacement = enum_change.partition(": ")
            before, _, after = replacement.partition(" -> ")
            payload_repairs.append(
                {
                    "kind": "correct_enum_value",
                    "field_name": field_name,
                    "from": before,
                    "to": after,
                }
            )

        if payload_repairs:
            repair_actions.extend(payload_repairs)
            candidate_output = json.dumps(payload, ensure_ascii=False)
            last_output = candidate_output
            last_repair_actions = repair_actions

        attempt_metrics.keys_removed = removed_keys
        attempt_metrics.enums_normalized = enum_changes
        last_removed_keys = removed_keys
        last_enum_changes = enum_changes

        try:
            model = schema.model_validate(payload)
            attempt_metrics.schema_validation_success = True
            attempt_metrics.overall_success = True
        except ValidationError as exc:
            last_error_kind = "validation"
            detail_lines = _categorize_validation_errors(exc)
            last_detail = "\n".join(detail_lines) if detail_lines else _format_validation_error(exc)
            attempt_metrics.validation_error = _format_validation_error(exc)
            attempt_metrics.retry_exhausted = attempt_index == max_retry
            attempts.append(
                StructuredGenerationAttempt(
                    attempt_index=attempt_index,
                    prompt=attempt_prompt,
                    raw_output=raw_output,
                    candidate_output=candidate_output,
                    json_error=None,
                    validation_error=last_detail,
                    error_kind=last_error_kind,
                    normalization=normalization,
                    normalization_details=normalization_details,
                    repair_actions=repair_actions,
                    keys_removed=removed_keys,
                    enum_normalizations=enum_changes,
                )
            )
            attempt_metrics_rows.append(attempt_metrics)
            if metrics_collector is not None:
                metrics_collector.record(attempt_metrics)
            continue

        attempts.append(
            StructuredGenerationAttempt(
                attempt_index=attempt_index,
                prompt=attempt_prompt,
                raw_output=raw_output,
                candidate_output=candidate_output,
                json_error=None,
                validation_error=None,
                error_kind=None,
                normalization=normalization,
                normalization_details=normalization_details,
                repair_actions=repair_actions,
                keys_removed=removed_keys,
                enum_normalizations=enum_changes,
            )
        )
        attempt_metrics_rows.append(attempt_metrics)
        if metrics_collector is not None:
            metrics_collector.record(attempt_metrics)
        canonical_payload = model.model_dump(mode="json", by_alias=True)
        canonical_output = json.dumps(canonical_payload, ensure_ascii=False)
        return StructuredGenerationResult(
            model=model,
            payload=canonical_payload,
            raw_output=raw_output,
            candidate_output=canonical_output,
            attempts=attempts,
            attempt_count=len(attempts),
            last_error_kind=None,
            normalization=normalization,
            normalization_details=normalization_details,
            repair_actions=repair_actions,
            structured_decoding=structured_decoding,
            attempt_metrics=attempt_metrics_rows,
        )

    raise StructuredGenerationError(
        f"Structured generation failed for {schema.__name__} after {max_retry + 1} attempts",
        attempts=attempts,
        last_output=last_output,
        last_raw_output=last_raw_output,
        last_error_kind=last_error_kind,
        normalization=last_normalization,
        normalization_details=last_normalization_details,
        repair_actions=last_repair_actions,
        structured_decoding=structured_decoding,
        attempt_metrics=attempt_metrics_rows,
    )
