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
DeceptionStyleLiteral = Literal["evasion", "half_truth", "outright_lie", "exaggeration", "omission"]
DistortionTypeLiteral = Literal[
    "exaggeration",
    "minimization",
    "malicious_twist",
    "emotional_coloring",
    "detail_invention",
    "source_confusion",
    "faithful",
]
TraumaResponseLiteral = Literal[
    "avoidance",
    "overprotection",
    "aggression",
    "withdrawal",
    "hypervigilance",
    "ritual_coping",
    "resilience",
]
DurationLiteral = Literal["short_term", "long_term", "permanent"]
NegotiateActionLiteral = Literal["accept", "counter_offer", "reject", "walk_away", "stall", "bluff"]
CultureActionLiteral = Literal["adopt", "modify", "reject", "oppose", "indifferent"]
MinorityActionLiteral = Literal["comply", "grumble", "passive_resist", "splinter", "coup_attempt"]
SparkEventLiteral = Literal[
    "food_shortage",
    "battle_loss",
    "oracle_conflict",
    "leader_death",
    "betrayal",
    "resource_discovery",
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


class TaskDOutput(StrictWorldSimModel):
    text_ko: str = Field(min_length=1)
    text_en: str = Field(min_length=1)
    event_type: str = Field(min_length=1)


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


class TaskEOutput_v3(StrictWorldSimModel):
    action_id: int
    confidence: float = Field(ge=0.0, le=1.0)
    hint: str = Field(min_length=1)
    personality_reasoning: DominantTraitLiteral
    temperament_factor: str = Field(min_length=1)


class TaskFOutput_v3(StrictWorldSimModel):
    emotion: EmotionLiteral
    intensity: float = Field(ge=0.0, le=1.0)
    cause: str = Field(min_length=1)
    previous_emotion: EmotionLiteral
    transition_type: TransitionTypeLiteral
    temperament_amplifier: str = Field(min_length=1)


class TaskIOutput_v3(StrictWorldSimModel):
    priority_id: int = Field(ge=0, le=9)
    reasoning: str = Field(min_length=5)
    need_addressed: NeedLiteral
    urgency: float = Field(ge=0.0, le=1.0)


class TaskJOutput_v3(StrictWorldSimModel):
    coping_id: int = Field(ge=0, le=9)
    coping_type: CopingTypeLiteral
    stress_delta: float = Field(ge=-1.0, le=0.0)
    hint: str = Field(min_length=5)
    side_effect: SideEffectLiteral


class TaskKOutput_v3(StrictWorldSimModel):
    social_action_id: int = Field(ge=0, le=9)
    trust_delta: float = Field(ge=-0.5, le=0.5)
    hint: str = Field(min_length=5)
    relationship_intent: RelationshipIntentLiteral
    reciprocity_expectation: ReciprocityExpectationLiteral


class TaskLOutput_v3(StrictWorldSimModel):
    response_id: int = Field(ge=0, le=9)
    trust_delta: float = Field(ge=-0.5, le=0.5)
    hint: str = Field(min_length=5)
    forgiveness_threshold: float = Field(ge=0.0, le=1.0)
    social_memory: SocialMemoryLiteral


class TaskMOutput_v3(StrictWorldSimModel):
    decision_id: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    dissent_risk: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=5)
    resource_commitment: ResourceCommitmentLiteral
    timeline: TimelineLiteral


class TaskNOutput_v3(StrictWorldSimModel):
    accept: bool
    counter_offer_give: str = Field(min_length=1)
    counter_offer_want: str = Field(min_length=1)
    hint: str = Field(min_length=5)
    negotiation_stance: NegotiationStanceLiteral
    walk_away_threshold: float = Field(ge=0.0, le=1.0)


class TaskOOutput(StrictWorldSimModel):
    """Deception — public claim vs private truth."""

    public_claim: str = Field(min_length=5)
    private_truth: str = Field(min_length=5)
    deception_style: DeceptionStyleLiteral
    lie_degree: float = Field(ge=0.0, le=1.0)
    detection_risk: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class TaskPOutput(StrictWorldSimModel):
    """Rumor — personality-biased information distortion."""

    retold_version: str = Field(min_length=10)
    distortion_type: DistortionTypeLiteral
    added_detail: str = Field(min_length=1)
    dropped_detail: str = Field(min_length=1)
    emotional_charge: float = Field(ge=-1.0, le=1.0)


class TaskQOutput(StrictWorldSimModel):
    """Trauma — past event affecting present behavior."""

    trauma_response: TraumaResponseLiteral
    behavioral_change: str = Field(min_length=5)
    trigger_situation: str = Field(min_length=5)
    intensity: float = Field(ge=0.0, le=1.0)
    duration: DurationLiteral
    coping_mechanism: str = Field(min_length=5)


class TaskROutput(StrictWorldSimModel):
    """Negotiate — single round of personality-driven negotiation."""

    action: NegotiateActionLiteral
    counter_give: str = Field(min_length=1)
    counter_want: str = Field(min_length=1)
    reasoning: str = Field(min_length=5)
    emotional_state: EmotionLiteral
    walk_away_threshold: float = Field(ge=0.0, le=1.0)


class TaskSOutput(StrictWorldSimModel):
    """Culture — adopt/modify/reject cultural elements."""

    action: CultureActionLiteral
    modified_practice: str = Field(min_length=1)
    reasoning: str = Field(min_length=5)
    social_pressure: float = Field(ge=0.0, le=1.0)
    tradition_conflict: bool


class TaskTOutput(StrictWorldSimModel):
    """Group dissent — collective decision with minority faction."""

    decision_id: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    dissent_risk: float = Field(ge=0.0, le=1.0)
    minority_position: int = Field(ge=0, le=9)
    minority_action: MinorityActionLiteral
    spark_event: SparkEventLiteral
    reasoning: str = Field(min_length=5)
    timeline: TimelineLiteral


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

TASK_OUTPUT_SCHEMAS_V3 = {
    "A": TaskAOutput,
    "B": TaskBOutput,
    "C": TaskCOutput,
    "D": TaskDOutput,
    "E": TaskEOutput_v3,
    "F": TaskFOutput_v3,
    "G": TaskGOutput,
    "H": TaskHOutput,
    "I": TaskIOutput_v3,
    "J": TaskJOutput_v3,
    "K": TaskKOutput_v3,
    "L": TaskLOutput_v3,
    "M": TaskMOutput_v3,
    "N": TaskNOutput_v3,
    "O": TaskOOutput,
    "P": TaskPOutput,
    "Q": TaskQOutput,
    "R": TaskROutput,
    "S": TaskSOutput,
    "T": TaskTOutput,
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
TASK_O_SCHEMA = TaskOOutput
TASK_P_SCHEMA = TaskPOutput
TASK_Q_SCHEMA = TaskQOutput
TASK_R_SCHEMA = TaskROutput
TASK_S_SCHEMA = TaskSOutput
TASK_T_SCHEMA = TaskTOutput


def get_schema_for_task(task_id: str, *, version: int = 2) -> type[BaseModel]:
    registry = TASK_OUTPUT_SCHEMAS_V3 if version == 3 else TASK_OUTPUT_SCHEMAS
    try:
        return registry[task_id]
    except KeyError as exc:  # pragma: no cover - defensive caller error
        raise ValueError(f"Unknown task_id: {task_id} (version={version})") from exc


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


def _build_task_enum_fields(task_output_schemas: dict[str, type[BaseModel]]) -> dict[str, dict[str, list[str]]]:
    task_enum_fields: dict[str, dict[str, list[str]]] = {}
    for task_id, schema in task_output_schemas.items():
        enum_fields: dict[str, list[str]] = {}
        for field_name, field in schema.model_fields.items():
            values = _literal_values(field.annotation)
            if values:
                enum_fields[field.alias or field_name] = list(values)
        task_enum_fields[task_id] = enum_fields
    return task_enum_fields


TASK_ENUM_FIELDS: dict[str, dict[str, list[str]]] = _build_task_enum_fields(TASK_OUTPUT_SCHEMAS)
TASK_ENUM_FIELDS_V3: dict[str, dict[str, list[str]]] = _build_task_enum_fields(TASK_OUTPUT_SCHEMAS_V3)
