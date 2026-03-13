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
NeedLiteral = Literal[
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
CopingTypeLiteral = Literal[
    "active_avoidance",
    "emotional_release",
    "social_support",
    "ritualistic",
    "substance",
    "acceptance",
    "aggression",
]
SideEffectLiteral = Literal[
    "aggression_increase",
    "isolation",
    "faith_increase",
    "trust_decrease",
    "morale_boost",
    "exhaustion",
    "none",
]
RelationshipIntentLiteral = Literal[
    "alliance",
    "cautious_observation",
    "hostile",
    "submissive",
    "dominant",
    "trade_partner",
    "ignore",
]
ReciprocityExpectationLiteral = Literal["none", "gift", "service", "alliance"]
SocialMemoryLiteral = Literal[
    "theft_betrayal",
    "aid_gratitude",
    "shared_danger",
    "insult_resentment",
    "gift_goodwill",
    "combat_respect",
    "abandonment",
    "none",
]
NegotiationStanceLiteral = Literal["generous", "fair", "hard_bargain", "exploitative"]
TimelineLiteral = Literal["immediate", "delayed", "conditional"]
ResourceCommitmentLiteral = Literal["food", "tools", "labor", "weapons", "none"]


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


class TaskIOutput(StrictWorldSimModel):
    priority_id: int = Field(ge=0, le=9)
    reasoning_ko: str = Field(min_length=5)
    reasoning_en: str = Field(min_length=5)
    need_addressed: NeedLiteral
    urgency: float = Field(ge=0.0, le=1.0)


class TaskJOutput(StrictWorldSimModel):
    coping_id: int = Field(ge=0, le=9)
    coping_type: CopingTypeLiteral
    stress_delta: float = Field(ge=-1.0, le=0.0)
    hint_ko: str = Field(min_length=5)
    hint_en: str = Field(min_length=5)
    side_effect: SideEffectLiteral


class TaskKOutput(StrictWorldSimModel):
    social_action_id: int = Field(ge=0, le=9)
    trust_delta: float = Field(ge=-0.5, le=0.5)
    hint_ko: str = Field(min_length=5)
    hint_en: str = Field(min_length=5)
    relationship_intent: RelationshipIntentLiteral
    reciprocity_expectation: ReciprocityExpectationLiteral


class TaskLOutput(StrictWorldSimModel):
    response_id: int = Field(ge=0, le=9)
    trust_delta: float = Field(ge=-0.5, le=0.5)
    hint_ko: str = Field(min_length=5)
    hint_en: str = Field(min_length=5)
    forgiveness_threshold: float = Field(ge=0.0, le=1.0)
    social_memory: SocialMemoryLiteral


class TaskMOutput(StrictWorldSimModel):
    decision_id: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    dissent_risk: float = Field(ge=0.0, le=1.0)
    reasoning_ko: str = Field(min_length=5)
    reasoning_en: str = Field(min_length=5)
    resource_commitment: ResourceCommitmentLiteral
    timeline: TimelineLiteral


class TaskNOutput(StrictWorldSimModel):
    accept: bool
    counter_offer_give: str = Field(min_length=1)
    counter_offer_want: str = Field(min_length=1)
    hint_ko: str = Field(min_length=5)
    hint_en: str = Field(min_length=5)
    negotiation_stance: NegotiationStanceLiteral
    walk_away_threshold: float = Field(ge=0.0, le=1.0)


TASK_OUTPUT_SCHEMAS = {
    "A": TaskAOutput,
    "B": TaskBOutput,
    "C": TaskCOutput,
    "E": TaskEOutput,
    "F": TaskFOutput,
    "G": TaskGOutput,
    "H": TaskHOutput,
    "I": TaskIOutput,
    "J": TaskJOutput,
    "K": TaskKOutput,
    "L": TaskLOutput,
    "M": TaskMOutput,
    "N": TaskNOutput,
}

TASK_A_SCHEMA = TaskAOutput
TASK_B_SCHEMA = TaskBOutput
TASK_C_SCHEMA = TaskCOutput
TASK_E_SCHEMA = TaskEOutput
TASK_F_SCHEMA = TaskFOutput
TASK_G_SCHEMA = TaskGOutput
TASK_H_SCHEMA = TaskHOutput
TASK_I_SCHEMA = TaskIOutput
TASK_J_SCHEMA = TaskJOutput
TASK_K_SCHEMA = TaskKOutput
TASK_L_SCHEMA = TaskLOutput
TASK_M_SCHEMA = TaskMOutput
TASK_N_SCHEMA = TaskNOutput


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
