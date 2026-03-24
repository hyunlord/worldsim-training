from __future__ import annotations

import pytest
from pathlib import Path

from scripts.generate_data import build_response_format, load_generation_config
from training.lib.output_schema import TASK_OUTPUT_SCHEMAS_V3


COT_TASKS = {
    "E": ("personality_reasoning", "action_id"),
    "F": ("temperament_amplifier", "emotion"),
    "I": ("reasoning", "priority_id"),
    "J": ("hint", "coping_id"),
    "K": ("hint", "social_action_id"),
    "L": ("hint", "response_id"),
    "M": ("reasoning", "decision_id"),
    "R": ("reasoning", "action"),
    "T": ("reasoning", "decision_id"),
}


@pytest.mark.parametrize(
    ("task_id", "reasoning_field", "choice_field"),
    [(task_id, reasoning_field, choice_field) for task_id, (reasoning_field, choice_field) in COT_TASKS.items()],
)
def test_reasoning_before_choice(task_id: str, reasoning_field: str, choice_field: str) -> None:
    schema = TASK_OUTPUT_SCHEMAS_V3[task_id]
    fields = list(schema.model_fields.keys())
    r_idx = fields.index(reasoning_field)
    c_idx = fields.index(choice_field)
    assert r_idx < c_idx, f"Task {task_id}: {reasoning_field}@{r_idx} should precede {choice_field}@{c_idx}"


def test_non_cot_tasks_unchanged() -> None:
    assert list(TASK_OUTPUT_SCHEMAS_V3["O"].model_fields.keys())[0] == "public_claim"
    assert list(TASK_OUTPUT_SCHEMAS_V3["P"].model_fields.keys())[0] == "retold_version"
    assert list(TASK_OUTPUT_SCHEMAS_V3["Q"].model_fields.keys())[0] == "trauma_response"


@pytest.mark.parametrize(
    ("task_id", "reasoning_field", "choice_field"),
    [(task_id, reasoning_field, choice_field) for task_id, (reasoning_field, choice_field) in COT_TASKS.items()],
)
def test_generation_response_schema_orders_reasoning_before_choice(
    task_id: str,
    reasoning_field: str,
    choice_field: str,
) -> None:
    settings = load_generation_config(Path("config"))
    job = {
        "task": task_id,
        "schema_version": 3,
        "action_options": [{"id": 0}, {"id": 1}, {"id": 2}],
    }
    response_format, _ = build_response_format(job, settings, schema_version=3)
    assert response_format is not None
    schema = response_format["json_schema"]["schema"]
    fields = list(schema["properties"].keys())
    r_idx = fields.index(reasoning_field)
    c_idx = fields.index(choice_field)
    assert r_idx < c_idx, f"Task {task_id}: response_format has {reasoning_field}@{r_idx} after {choice_field}@{c_idx}"
