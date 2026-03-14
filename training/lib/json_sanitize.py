from __future__ import annotations

from training.lib.output_schema import TASK_ENUM_FIELDS, TASK_OUTPUT_SCHEMAS


TASK_ALLOWED_KEYS_REGISTRY: dict[str, set[str]] = {}

ENUM_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "emotion_expressed": {
        "sorrow": "sadness",
    }
}


def _build_allowed_keys_registry() -> None:
    for task_id, schema in TASK_OUTPUT_SCHEMAS.items():
        TASK_ALLOWED_KEYS_REGISTRY[task_id] = {
            field.alias or field_name
            for field_name, field in schema.model_fields.items()
        }


def sanitize_keys(parsed_dict: dict, task_id: str) -> tuple[dict, list[str]]:
    allowed = TASK_ALLOWED_KEYS_REGISTRY.get(task_id)
    if allowed is None:
        return parsed_dict, []

    removed_keys: list[str] = []
    sanitized: dict = {}
    for key, value in parsed_dict.items():
        if key in allowed:
            sanitized[key] = value
        else:
            removed_keys.append(key)
    return sanitized, removed_keys


def _normalize_enum_token(value: str) -> str:
    return value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _fuzzy_match_enum(value: str, allowed_values: list[str]) -> str | None:
    normalized_value = _normalize_enum_token(value)

    for allowed in allowed_values:
        if allowed.lower() == value.lower():
            return allowed

    for allowed in allowed_values:
        allowed_normalized = _normalize_enum_token(allowed)
        if normalized_value == allowed_normalized:
            return allowed
        if normalized_value in allowed_normalized or allowed_normalized in normalized_value:
            return allowed

    return None


def normalize_enum_values(parsed_dict: dict, task_id: str) -> tuple[dict, list[str]]:
    enum_fields = TASK_ENUM_FIELDS.get(task_id, {})
    normalizations: list[str] = []
    normalized_dict = dict(parsed_dict)

    for field_name, allowed_values in enum_fields.items():
        if field_name not in normalized_dict:
            continue
        current = normalized_dict[field_name]
        if not isinstance(current, str):
            continue

        alias_target = ENUM_VALUE_ALIASES.get(field_name, {}).get(current.strip().lower())
        if alias_target is not None:
            normalized_dict[field_name] = alias_target
            normalizations.append(f"{field_name}: {current} -> {alias_target}")
            continue

        matched = _fuzzy_match_enum(current, allowed_values)
        if matched is not None and current != matched:
            normalized_dict[field_name] = matched
            normalizations.append(f"{field_name}: {current} -> {matched}")

    return normalized_dict, normalizations


def sanitize_json_output(parsed_dict: dict, task_id: str) -> tuple[dict, list[dict]]:
    sanitized, removed_keys = sanitize_keys(parsed_dict, task_id)
    normalized, normalizations = normalize_enum_values(sanitized, task_id)

    actions: list[dict] = []
    if removed_keys:
        actions.append({"kind": "filter_extra_keys", "removed_keys": removed_keys})
    if normalizations:
        actions.append({"kind": "normalize_enum_values", "normalized": normalizations})
    return normalized, actions


_build_allowed_keys_registry()
