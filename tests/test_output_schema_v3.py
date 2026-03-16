from __future__ import annotations

from training.lib.output_schema import (
    TASK_ENUM_FIELDS_V3,
    TASK_OUTPUT_SCHEMAS,
    TASK_OUTPUT_SCHEMAS_V3,
    TaskDOutput,
    TaskEOutput,
    TaskEOutput_v3,
    TaskIOutput_v3,
    TaskJOutput_v3,
    TaskKOutput_v3,
    TaskLOutput_v3,
    TaskMOutput_v3,
    TaskNOutput_v3,
    TaskOOutput,
    TaskPOutput,
    TaskQOutput,
    TaskROutput,
    TaskSOutput,
    TaskTOutput,
    get_schema_for_task,
)


def test_task_output_schemas_v3_register_expected_tasks() -> None:
    assert len(TASK_OUTPUT_SCHEMAS_V3) == 20

    for task in "ABCDEFGHIJKLMNOPQRST":
        assert task in TASK_OUTPUT_SCHEMAS_V3

    assert TASK_OUTPUT_SCHEMAS_V3["D"] is TaskDOutput
    assert TASK_OUTPUT_SCHEMAS["E"] is TaskEOutput
    assert TASK_OUTPUT_SCHEMAS_V3["E"] is TaskEOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["I"] is TaskIOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["J"] is TaskJOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["K"] is TaskKOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["L"] is TaskLOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["M"] is TaskMOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["N"] is TaskNOutput_v3
    assert TASK_OUTPUT_SCHEMAS_V3["O"] is TaskOOutput
    assert TASK_OUTPUT_SCHEMAS_V3["P"] is TaskPOutput
    assert TASK_OUTPUT_SCHEMAS_V3["Q"] is TaskQOutput
    assert TASK_OUTPUT_SCHEMAS_V3["R"] is TaskROutput
    assert TASK_OUTPUT_SCHEMAS_V3["S"] is TaskSOutput
    assert TASK_OUTPUT_SCHEMAS_V3["T"] is TaskTOutput


def test_get_schema_for_task_supports_versions() -> None:
    assert get_schema_for_task("E") is TaskEOutput
    assert get_schema_for_task("E", version=3) is TaskEOutput_v3
    assert get_schema_for_task("O", version=3) is TaskOOutput


def test_v3_logic_tasks_do_not_expose_korean_fields() -> None:
    for task in "EFHIJKLMN":
        schema = TASK_OUTPUT_SCHEMAS_V3[task]
        field_names = {field.alias or name for name, field in schema.model_fields.items()}
        assert not any("_ko" in name for name in field_names)


def test_v3_enum_fields_cover_new_tasks() -> None:
    assert TASK_ENUM_FIELDS_V3["E"]["personality_reasoning"] == [
        "novelty_seeking",
        "harm_avoidance",
        "reward_dependence",
        "persistence",
    ]
    assert TASK_ENUM_FIELDS_V3["O"]["deception_style"] == [
        "evasion",
        "half_truth",
        "outright_lie",
        "exaggeration",
        "omission",
    ]
    assert TASK_ENUM_FIELDS_V3["P"]["distortion_type"][-1] == "faithful"
    assert TASK_ENUM_FIELDS_V3["Q"]["duration"] == ["short_term", "long_term", "permanent"]
    assert TASK_ENUM_FIELDS_V3["R"]["action"] == [
        "accept",
        "counter_offer",
        "reject",
        "walk_away",
        "stall",
        "bluff",
    ]
    assert TASK_ENUM_FIELDS_V3["S"]["action"] == ["adopt", "modify", "reject", "oppose", "indifferent"]
    assert TASK_ENUM_FIELDS_V3["T"]["minority_action"] == [
        "comply",
        "grumble",
        "passive_resist",
        "splinter",
        "coup_attempt",
    ]


def test_new_task_schemas_validate_examples() -> None:
    task_o = TaskOOutput.model_validate(
        {
            "public_claim": "I only checked the empty trail and found nothing useful.",
            "private_truth": "I found a hidden cache of smoked fish near the ridge.",
            "deception_style": "omission",
            "lie_degree": 0.64,
            "detection_risk": 0.71,
            "confidence": 0.58,
        }
    )
    assert task_o.deception_style == "omission"

    task_p = TaskPOutput.model_validate(
        {
            "retold_version": "The stranger was not alone and kept reaching toward a hidden knife.",
            "distortion_type": "emotional_coloring",
            "added_detail": "hidden knife",
            "dropped_detail": "left when noticed",
            "emotional_charge": 0.66,
        }
    )
    assert task_p.distortion_type == "emotional_coloring"

    task_q = TaskQOutput.model_validate(
        {
            "trauma_response": "hypervigilance",
            "behavioral_change": "Sleeps in short bursts and checks every noise outside the shelter.",
            "trigger_situation": "A howl near camp after dark.",
            "intensity": 0.82,
            "duration": "long_term",
            "coping_mechanism": "Counts the firewood stack before resting.",
        }
    )
    assert task_q.duration == "long_term"

    task_r = TaskROutput.model_validate(
        {
            "action": "counter_offer",
            "counter_give": "obsidian_blades:2",
            "counter_want": "salt_blocks:4",
            "reasoning": "Push for better terms while keeping the other band engaged.",
            "emotional_state": "anticipation",
            "walk_away_threshold": 0.44,
        }
    )
    assert task_r.action == "counter_offer"

    task_s = TaskSOutput.model_validate(
        {
            "action": "modify",
            "modified_practice": "Adopt the moonfast feast but keep our elder blessing before eating.",
            "reasoning": "Blend the new custom with an older ritual the band already trusts.",
            "social_pressure": 0.61,
            "tradition_conflict": True,
        }
    )
    assert task_s.action == "modify"

    task_t = TaskTOutput.model_validate(
        {
            "decision_id": 2,
            "confidence": 0.69,
            "dissent_risk": 0.73,
            "minority_position": 4,
            "minority_action": "passive_resist",
            "spark_event": "betrayal",
            "reasoning": "Most hunters accept the deal, but the wounded faction will quietly undermine it.",
            "timeline": "conditional",
        }
    )
    assert task_t.spark_event == "betrayal"
