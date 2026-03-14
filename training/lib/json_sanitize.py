from __future__ import annotations

from training.lib.output_schema import TASK_ENUM_FIELDS, get_schema_for_task


TASK_ALLOWED_KEYS_REGISTRY: dict[str, set[str]] = {
    "A": set(),
    "B": set(),
    "C": set(),
    "E": set(),
    "F": set(),
    "G": set(),
    "H": set(),
}

ENUM_VALUE_ALIASES: dict[str, dict[str, str]] = {
    "emotion_expressed": {
        "sorrow": "sadness",
    }
}


def _build_allowed_keys_registry() -> None:
    for task_id in TASK_ALLOWED_KEYS_REGISTRY:
        schema = get_schema_for_task(task_id)
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

        candidate = current.strip().lower()
        alias_target = ENUM_VALUE_ALIASES.get(field_name, {}).get(candidate)
        if alias_target is not None:
            candidate = alias_target
        for allowed in allowed_values:
            if allowed.lower() == candidate:
                if current != allowed:
                    normalized_dict[field_name] = allowed
                    normalizations.append(f"{field_name}: {current} -> {allowed}")
                break

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
