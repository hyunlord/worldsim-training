"""Structured generation guardrails for WorldSim sample generation.

This module implements the generation-time layers that sit between the model
and downstream analyzers:

1. prompt contract (handled by caller)
2. constrained/deterministic decoding (handled by caller/backend)
3. lightweight JSON repair + schema-aware normalization
4. validation feedback retries

The goal is to improve structured JSON reliability without changing dataset
contracts or hiding raw model outputs from inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError


DEFAULT_MAX_RETRY = 3


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


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        message = error.get("msg") or error.get("type")
        parts.append(f"{location}:{message}")
    return ", ".join(parts)


def _strip_markdown_fence(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text, None

    lines = stripped.splitlines()
    if not lines:
        return text, None
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip(), {"kind": "strip_markdown_fence"}


def _strip_non_json_prefix(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped:
        return text, None

    object_index = stripped.find("{")
    array_index = stripped.find("[")
    indices = [index for index in (object_index, array_index) if index >= 0]
    if not indices:
        return text, None
    start_index = min(indices)
    if start_index == 0:
        return stripped, None
    return stripped[start_index:].strip(), {"kind": "strip_non_json_prefix"}


def _trim_trailing_text(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped:
        return text, None
    try:
        _, end_index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return text, None

    trailing = stripped[end_index:].strip()
    if not trailing:
        return stripped, None
    return stripped[:end_index].strip(), {"kind": "trim_trailing_text", "removed_text": trailing}


def _closing_suffix_for_unbalanced_json(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()

    if in_string:
        return '"' + "".join("}" if opener == "{" else "]" for opener in reversed(stack))
    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _close_unbalanced_json(text: str) -> tuple[str, dict[str, Any] | None]:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text, None

    suffix = _closing_suffix_for_unbalanced_json(stripped)
    if not suffix:
        return text, None
    candidate = stripped + suffix
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return text, None
    return candidate, {"kind": "close_unbalanced_json", "suffix": suffix}


def repair_json(text: str) -> dict[str, Any]:
    repaired = str(text)
    actions: list[dict[str, Any]] = []

    for repair in (_strip_markdown_fence, _strip_non_json_prefix, _trim_trailing_text, _close_unbalanced_json, _trim_trailing_text):
        repaired_next, action = repair(repaired)
        repaired = repaired_next
        if action is not None:
            actions.append(action)

    return {"text": repaired, "repair_actions": actions}


def _schema_allowed_keys(schema: type[BaseModel]) -> set[str]:
    return {field.alias or field_name for field_name, field in schema.model_fields.items()}


def _literal_choices(annotation: Any) -> tuple[str, ...]:
    origin = get_origin(annotation)
    if origin is None:
        return ()
    if str(origin).endswith("Literal"):
        choices = tuple(choice for choice in get_args(annotation) if isinstance(choice, str))
        return choices

    nested_choices: list[str] = []
    for arg in get_args(annotation):
        nested_choices.extend(_literal_choices(arg))
    return tuple(nested_choices)


def _normalize_enum_candidate(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _schema_enum_fields(schema: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    enum_fields: dict[str, tuple[str, ...]] = {}
    for field_name, field in schema.model_fields.items():
        choices = _literal_choices(field.annotation)
        if choices:
            enum_fields[field.alias or field_name] = choices
    return enum_fields


def _repair_payload(payload: Any, schema: type[BaseModel]) -> tuple[Any, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        return payload, []

    repaired = dict(payload)
    actions: list[dict[str, Any]] = []

    allowed_keys = _schema_allowed_keys(schema)
    extra_keys = sorted(key for key in repaired if key not in allowed_keys)
    if extra_keys:
        repaired = {key: value for key, value in repaired.items() if key in allowed_keys}
        actions.append({"kind": "filter_extra_keys", "removed_keys": extra_keys})

    for field_name, choices in _schema_enum_fields(schema).items():
        current = repaired.get(field_name)
        if not isinstance(current, str):
            continue
        normalized = _normalize_enum_candidate(current)
        canonical_by_normalized: dict[str, str] = {}
        for choice in choices:
            canonical_by_normalized.setdefault(_normalize_enum_candidate(choice), choice)
        canonical = canonical_by_normalized.get(normalized)
        if canonical is not None and canonical != current:
            repaired[field_name] = canonical
            actions.append(
                {
                    "kind": "correct_enum_value",
                    "field_name": field_name,
                    "from": current,
                    "to": canonical,
                }
            )

    return repaired, actions


def _build_retry_prompt(
    prompt: str,
    *,
    schema_name: str,
    attempt_index: int,
    last_error_kind: str | None,
    detail: str | None,
    bad_output: str,
) -> str:
    reason = detail or last_error_kind or "validation_failed"
    problem_label = "JSON parsing" if last_error_kind == "json" else "schema validation"
    return (
        f"{prompt}\n\n"
        "The JSON you returned failed "
        f"{problem_label} for {schema_name}.\n\n"
        "Error:\n"
        f"{reason}\n\n"
        "Here is your previous JSON:\n"
        f"{bad_output}\n\n"
        "Return ONLY corrected JSON.\n"
        "Do not add new fields.\n"
        "Do not copy placeholder or schema text.\n"
        f"This is retry attempt {attempt_index}.\n"
    )


def generate_structured(
    llm: Any,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retry: int = DEFAULT_MAX_RETRY,
    output_normalizer: Any | None = None,
) -> StructuredGenerationResult:
    attempts: list[StructuredGenerationAttempt] = []
    last_error_kind: str | None = None
    last_detail: str | None = None
    last_output = ""
    last_raw_output = ""
    last_normalization: str | None = None
    last_normalization_details: list[dict[str, str]] = []
    last_repair_actions: list[dict[str, Any]] = []

    for attempt_index in range(max_retry + 1):
        attempt_prompt = prompt if attempt_index == 0 else _build_retry_prompt(
            prompt,
            schema_name=schema.__name__,
            attempt_index=attempt_index,
            last_error_kind=last_error_kind,
            detail=last_detail,
            bad_output=last_output,
        )
        raw_output = str(llm(attempt_prompt))
        candidate_output = raw_output
        normalization: str | None = None
        normalization_details: list[dict[str, str]] = []
        repair_actions: list[dict[str, Any]] = []

        if output_normalizer is not None:
            normalized = output_normalizer(raw_output)
            candidate_output = normalized["text"]
            normalization = normalized.get("normalization")
            normalization_details = list(normalized.get("normalization_details") or [])

        repaired_json = repair_json(candidate_output)
        candidate_output = repaired_json["text"]
        repair_actions.extend(repaired_json["repair_actions"])

        last_output = candidate_output
        last_raw_output = raw_output
        last_normalization = normalization
        last_normalization_details = normalization_details
        last_repair_actions = repair_actions

        try:
            payload = json.loads(candidate_output)
        except json.JSONDecodeError as exc:
            last_error_kind = "json"
            last_detail = f"{type(exc).__name__}: {exc.msg}"
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
                )
            )
            continue

        payload, payload_repairs = _repair_payload(payload, schema)
        if payload_repairs:
            repair_actions.extend(payload_repairs)
            candidate_output = json.dumps(payload, ensure_ascii=False)
            last_output = candidate_output
            last_repair_actions = repair_actions

        try:
            model = schema.model_validate(payload)
        except ValidationError as exc:
            last_error_kind = "validation"
            last_detail = _format_validation_error(exc)
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
                )
            )
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
            )
        )
        return StructuredGenerationResult(
            model=model,
            payload=model.model_dump(mode="json", by_alias=True),
            raw_output=raw_output,
            candidate_output=candidate_output,
            attempts=attempts,
            attempt_count=len(attempts),
            last_error_kind=None,
            normalization=normalization,
            normalization_details=normalization_details,
            repair_actions=repair_actions,
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
    )
