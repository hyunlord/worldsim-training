from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError


DEFAULT_MAX_RETRY = 2


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
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.attempt_count = len(attempts)
        self.last_output = last_output
        self.last_raw_output = last_raw_output
        self.last_error_kind = last_error_kind
        self.normalization = normalization
        self.normalization_details = normalization_details or []


def _format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        parts.append(f"{location}:{error.get('type')}")
    return ", ".join(parts)


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

        if output_normalizer is not None:
            normalized = output_normalizer(raw_output)
            candidate_output = normalized["text"]
            normalization = normalized.get("normalization")
            normalization_details = list(normalized.get("normalization_details") or [])

        last_output = candidate_output
        last_raw_output = raw_output
        last_normalization = normalization
        last_normalization_details = normalization_details

        try:
            payload = json.loads(candidate_output)
        except json.JSONDecodeError as exc:
            last_error_kind = "json"
            last_detail = type(exc).__name__
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
                )
            )
            continue

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
        )

    raise StructuredGenerationError(
        f"Structured generation failed for {schema.__name__} after {max_retry + 1} attempts",
        attempts=attempts,
        last_output=last_output,
        last_raw_output=last_raw_output,
        last_error_kind=last_error_kind,
        normalization=last_normalization,
        normalization_details=last_normalization_details,
    )
