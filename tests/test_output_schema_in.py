from __future__ import annotations

from training.lib.output_schema import (
    TASK_ENUM_FIELDS,
    TASK_OUTPUT_SCHEMAS,
    TaskIOutput,
    TaskJOutput,
    TaskKOutput,
    TaskLOutput,
    TaskMOutput,
    TaskNOutput,
    get_schema_for_task,
)
from training.lib.qlora_smoke import DEFAULT_TASKS, TASK_ALLOWED_KEYS, _sample_generation_assistant_prefix


def test_task_output_schemas_register_i_through_n() -> None:
    assert TASK_OUTPUT_SCHEMAS["I"] is TaskIOutput
    assert TASK_OUTPUT_SCHEMAS["J"] is TaskJOutput
    assert TASK_OUTPUT_SCHEMAS["K"] is TaskKOutput
    assert TASK_OUTPUT_SCHEMAS["L"] is TaskLOutput
    assert TASK_OUTPUT_SCHEMAS["M"] is TaskMOutput
    assert TASK_OUTPUT_SCHEMAS["N"] is TaskNOutput

    assert get_schema_for_task("I") is TaskIOutput
    assert get_schema_for_task("J") is TaskJOutput
    assert get_schema_for_task("K") is TaskKOutput
    assert get_schema_for_task("L") is TaskLOutput
    assert get_schema_for_task("M") is TaskMOutput
    assert get_schema_for_task("N") is TaskNOutput


def test_task_enum_fields_cover_new_literal_outputs() -> None:
    assert TASK_ENUM_FIELDS["I"]["need_addressed"] == [
        "hunger",
        "thirst",
        "warmth",
        "rest",
        "safety",
        "belonging",
        "esteem",
        "curiosity",
        "reproduction",
        "comfort",
        "purpose",
        "transcendence",
        "play",
    ]
    assert TASK_ENUM_FIELDS["J"]["coping_type"] == [
        "active_avoidance",
        "emotional_release",
        "social_support",
        "ritualistic",
        "substance",
        "acceptance",
        "aggression",
    ]
    assert TASK_ENUM_FIELDS["K"]["relationship_intent"] == [
        "alliance",
        "cautious_observation",
        "hostile",
        "submissive",
        "dominant",
        "trade_partner",
        "ignore",
    ]
    assert TASK_ENUM_FIELDS["L"]["social_memory"][-1] == "none"
    assert TASK_ENUM_FIELDS["M"]["timeline"] == ["immediate", "delayed", "conditional"]
    assert TASK_ENUM_FIELDS["N"]["negotiation_stance"] == ["generous", "fair", "hard_bargain", "exploitative"]


def test_qlora_scaffold_registers_i_through_n() -> None:
    for task in ("I", "J", "K", "L", "M", "N"):
        assert task in DEFAULT_TASKS
        assert task in TASK_ALLOWED_KEYS

    assert _sample_generation_assistant_prefix("I") == '{"priority_id": '
    assert _sample_generation_assistant_prefix("J") == '{"coping_id": '
    assert _sample_generation_assistant_prefix("K") == '{"social_action_id": '
    assert _sample_generation_assistant_prefix("L") == '{"response_id": '
    assert _sample_generation_assistant_prefix("M") == '{"decision_id": '
    assert _sample_generation_assistant_prefix("N") == '{"accept": '
