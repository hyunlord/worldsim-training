from __future__ import annotations

from enum import Enum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field


RegisterLiteral = Literal["haera", "hao", "hae"]
EmotionLiteral = Literal["joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"]
DominantTraitLiteral = Literal["novelty_seeking", "harm_avoidance", "reward_dependence", "persistence"]
SpeakerRoleLiteral = Literal["elder", "hunter", "shaman", "warrior", "healer", "gatherer", "craftsman", "chief", "scout", "observer"]
TransitionTypeLiteral = Literal["gradual", "sudden", "sustained"]
ActionTendencyLiteral = Literal["mobilize", "defend", "wait", "retreat", "celebrate", "mourn"]
MisinterpretationTypeLiteral = Literal[
    "overconfident_literal",
    "cautious_reversal",
    "optimistic_expansion",
    "passive_deferral",
    "symbolic_abstraction",
]


class StrictWorldSimModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, populate_by_name=True)


class TaskAOutput(StrictWorldSimModel):
    text_ko: str = Field(min_length=1)
    text_en: str = Field(min_length=1)
    register_value: RegisterLiteral = Field(alias="register")
    dominant_trait: DominantTraitLiteral
    temperament_expressed: str = Field(min_length=1)


class TaskBOutput(StrictWorldSimModel):
    text_ko: str = Field(min_length=1)
    text_en: str = Field(min_length=1)
    register_value: RegisterLiteral = Field(alias="register")
    emotion_expressed: EmotionLiteral
    intensity: float = Field(ge=0.0, le=1.0)
    mimetics: list[str]
    temperament_influence: str = Field(min_length=1)


class TaskCOutput(StrictWorldSimModel):
    speech_ko: str = Field(min_length=1)
    speech_en: str = Field(min_length=1)
    register_value: RegisterLiteral = Field(alias="register")
    emotion_expressed: EmotionLiteral
    speaker_role: SpeakerRoleLiteral
    temperament_tone: str = Field(min_length=1)


class TaskEOutput(StrictWorldSimModel):
    action_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    hint_ko: str = Field(min_length=1)
    hint_en: str = Field(min_length=1)
    personality_reasoning: str = Field(min_length=1)
    temperament_factor: str = Field(min_length=1)


class TaskFOutput(StrictWorldSimModel):
    emotion: EmotionLiteral
    intensity: float = Field(ge=0.0, le=1.0)
    cause_ko: str = Field(min_length=1)
    cause_en: str = Field(min_length=1)
    previous_emotion: EmotionLiteral
    transition_type: TransitionTypeLiteral
    temperament_amplifier: str = Field(min_length=1)


class TaskGOutput(StrictWorldSimModel):
    interpretation_ko: str = Field(min_length=1)
    interpretation_en: str = Field(min_length=1)
    action_tendency: ActionTendencyLiteral
    confidence: float = Field(ge=0.0, le=1.0)
    register_value: RegisterLiteral = Field(alias="register")
    misinterpretation_type: MisinterpretationTypeLiteral
    temperament_bias: str = Field(min_length=1)


class ResourceModifier(StrictWorldSimModel):
    target: str = Field(min_length=1)
    multiplier: float = Field(ge=0.0, le=5.0)


class SpecialZone(StrictWorldSimModel):
    kind: str = Field(min_length=1)
    spawn_count_min: int = Field(ge=0)
    spawn_count_max: int = Field(ge=0)


class SpecialResource(StrictWorldSimModel):
    name: str = Field(min_length=1)
    tags: list[str]


class AgentModifier(StrictWorldSimModel):
    system: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    effect: str = Field(min_length=1)


class TaskHOutput(StrictWorldSimModel):
    name: str = Field(min_length=1, pattern=r"^[A-Z][a-zA-Z]+$")
    description_en: str = Field(min_length=10)
    resource_modifiers: list[ResourceModifier]
    special_zones: list[SpecialZone]
    special_resources: list[SpecialResource]
    agent_modifiers: list[AgentModifier]


TASK_OUTPUT_SCHEMAS = {
    "A": TaskAOutput,
    "B": TaskBOutput,
    "C": TaskCOutput,
    "E": TaskEOutput,
    "F": TaskFOutput,
    "G": TaskGOutput,
    "H": TaskHOutput,
}

TASK_A_SCHEMA = TaskAOutput
TASK_B_SCHEMA = TaskBOutput
TASK_C_SCHEMA = TaskCOutput
TASK_E_SCHEMA = TaskEOutput
TASK_F_SCHEMA = TaskFOutput
TASK_G_SCHEMA = TaskGOutput
TASK_H_SCHEMA = TaskHOutput


def get_schema_for_task(task_id: str) -> type[BaseModel]:
    try:
        return TASK_OUTPUT_SCHEMAS[task_id]
    except KeyError as exc:  # pragma: no cover - defensive caller error
        raise ValueError(f"Unknown task_id: {task_id}") from exc


def _literal_values(annotation: Any) -> tuple[str, ...]:
    origin = get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return tuple(str(member.value) for member in annotation)
        return ()
    if str(origin).endswith("Literal"):
        return tuple(str(value) for value in get_args(annotation))

    values: list[str] = []
    for arg in get_args(annotation):
        values.extend(_literal_values(arg))
    return tuple(values)


def _build_task_enum_fields() -> dict[str, dict[str, list[str]]]:
    task_enum_fields: dict[str, dict[str, list[str]]] = {}
    for task_id, schema in TASK_OUTPUT_SCHEMAS.items():
        enum_fields: dict[str, list[str]] = {}
        for field_name, field in schema.model_fields.items():
            values = _literal_values(field.annotation)
            if values:
                enum_fields[field.alias or field_name] = list(values)
        task_enum_fields[task_id] = enum_fields
    return task_enum_fields


TASK_ENUM_FIELDS: dict[str, dict[str, list[str]]] = _build_task_enum_fields()
