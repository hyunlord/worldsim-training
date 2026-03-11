from __future__ import annotations

from typing import Literal

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
