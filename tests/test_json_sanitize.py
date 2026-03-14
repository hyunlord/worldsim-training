from __future__ import annotations

from training.lib.json_sanitize import TASK_ALLOWED_KEYS_REGISTRY, normalize_enum_values, sanitize_keys
from training.lib.output_schema import TASK_OUTPUT_SCHEMAS


def test_sanitize_keys_strips_extra_keys_and_preserves_allowed_keys() -> None:
    result, removed = sanitize_keys(
        {
            "text_ko": "hello",
            "text_en": "world",
            "register": "haera",
            "dominant_trait": "harm_avoidance",
            "temperament_expressed": "melancholic",
            "schema_explanation": "leaked",
        },
        "A",
    )

    assert removed == ["schema_explanation"]
    assert set(result) == {"text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"}


def test_sanitize_keys_unknown_task_returns_input_unchanged() -> None:
    payload = {"text_ko": "hello", "unexpected": "value"}
    result, removed = sanitize_keys(payload, "Z")
    assert result == payload
    assert removed == []


def test_normalize_enum_values_handles_case_and_whitespace() -> None:
    normalized, changes = normalize_enum_values(
        {
            "emotion": " Fear ",
            "previous_emotion": "Trust",
            "transition_type": " sudden ",
        },
        "F",
    )

    assert normalized["emotion"] == "fear"
    assert normalized["previous_emotion"] == "trust"
    assert normalized["transition_type"] == "sudden"
    assert "emotion:  Fear  -> fear" in changes
    assert "previous_emotion: Trust -> trust" in changes


def test_normalize_enum_values_leaves_unrecognized_value_unchanged() -> None:
    normalized, changes = normalize_enum_values({"emotion": "panic"}, "F")
    assert normalized["emotion"] == "panic"
    assert changes == []


def test_normalize_enum_values_maps_known_alias_to_allowed_literal() -> None:
    normalized, changes = normalize_enum_values({"emotion_expressed": "sorrow"}, "B")
    assert normalized["emotion_expressed"] == "sadness"
    assert changes == ["emotion_expressed: sorrow -> sadness"]


def test_normalize_enum_values_handles_empty_dict() -> None:
    normalized, changes = normalize_enum_values({}, "B")
    assert normalized == {}
    assert changes == []


def test_allowed_keys_registry_covers_all_task_output_schemas() -> None:
    assert set(TASK_ALLOWED_KEYS_REGISTRY) == set(TASK_OUTPUT_SCHEMAS)
    for task in "IJKLMN":
        assert task in TASK_ALLOWED_KEYS_REGISTRY
        assert TASK_ALLOWED_KEYS_REGISTRY[task]


def test_sanitize_keys_removes_extra_keys_for_task_k() -> None:
    result, removed = sanitize_keys(
        {
            "social_action_id": 2,
            "trust_delta": -0.32,
            "hint_ko": "test",
            "hint_en": "test",
            "relationship_intent": "alliance",
            "reciprocity_expectation": "service",
            "temperament": "choleric",
            "cause_ko": "test",
            "cause_en": "test",
        },
        "K",
    )

    assert set(result) == {
        "social_action_id",
        "trust_delta",
        "hint_ko",
        "hint_en",
        "relationship_intent",
        "reciprocity_expectation",
    }
    assert removed == ["temperament", "cause_ko", "cause_en"]


def test_normalize_enum_values_uses_fuzzy_matching_for_separator_drift() -> None:
    normalized, changes = normalize_enum_values(
        {"relationship_intent": "cautious observation"},
        "K",
    )

    assert normalized["relationship_intent"] == "cautious_observation"
    assert changes == [
        "relationship_intent: cautious observation -> cautious_observation"
    ]
