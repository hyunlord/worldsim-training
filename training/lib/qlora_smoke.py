from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from scripts.common import ensure_directory, read_jsonl, write_jsonl
from scripts.prepare_dataset import _validate_messages_row
from training.lib.output_schema import TASK_OUTPUT_SCHEMAS
from training.lib.json_sanitize import sanitize_json_output
from training.lib.structured_generation import (
    OUTLINES_REPETITION_PENALTY,
    StructuredGenerationError,
    TASK_MAX_NEW_TOKENS,
    STRUCTURED_GENERATION_DEFAULTS,
    build_structured_constraint,
    generate_structured,
)
from training.lib.structured_metrics import BatchMetrics, GenerationAttemptMetrics

try:
    import outlines as _outlines_module

    OUTLINES_AVAILABLE = True
except ImportError:
    _outlines_module = None  # type: ignore[assignment]
    OUTLINES_AVAILABLE = False


DEFAULT_RUN_MODE = "smoke"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
BASELINE_DATASET_ID = "worldsim-v31-mix-v1"
DEFAULT_TRAIN_FILE = Path("data/training/worldsim-v31-mix-v1/train_converted.jsonl")
DEFAULT_DEV_FILE = Path("data/training/worldsim-v31-mix-v1/dev_converted.jsonl")
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_TASKS = ("A", "B", "C", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N")
SAMPLES_PER_TASK: int = 5
MODEL_REGISTRY_PATH = Path("outputs") / "model_registry.json"
BEST_ADAPTER_POINTER_PATH = Path("outputs") / "best_adapter.txt"
RUN_MODE_DEFAULTS = {
    "smoke": {
        "model_name": DEFAULT_MODEL_NAME,
        "output_root": Path("outputs") / "smoke" / "worldsim-v31-mix-v1",
        "max_steps": 5,
        "max_train_samples": 32,
        "max_eval_samples": 16,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2e-4,
        "logging_steps": 1,
        "eval_steps": 0,
        "save_steps": 0,
        "save_total_limit": 1,
    },
    "baseline": {
        "model_name": BASELINE_MODEL_NAME,
        "output_root": Path("outputs") / "baseline" / "worldsim-v31-mix-v1",
        "max_steps": 200,
        "max_train_samples": 0,
        "max_eval_samples": 0,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1e-4,
        "logging_steps": 5,
        "eval_steps": 25,
        "save_steps": 25,
        "save_total_limit": 2,
    },
}
SAMPLE_GENERATION_REMINDER = (
    "\n\n[SYSTEM ROLE]\n"
    "You are filling one WorldSim JSON object from the task context above.\n"
    "\n[OUTPUT RULES]\n"
    "- Output must be a single JSON object.\n"
    "- The first character must be { and the last character must be }.\n"
    "- Use double quotes for every JSON key and every string value.\n"
    "- Do not output markdown, code fences, explanations, or extra text.\n"
    "- Only output the keys listed below.\n"
    "- Do not add any extra keys.\n"
    "- Do not copy instructions, schema descriptions, examples, enum lists, or placeholder text into the JSON values.\n"
    "- Fill every field with concrete values from the task context above.\n"
    "\n[TASK CONTEXT]\n"
    "- Use only the task context already provided above.\n"
)
LEAKY_GENERATION_SECTION_LABELS = {
    "출력 형식",
    "유효값 다시 보기",
    "규칙",
    "어투",
    "말투",
    "감정 선택지",
    "register 선택지",
    "화자 역할 선택지",
    "판단 근거 선택지",
    "이전 감정 고르기",
    "전이 방식 선택지",
    "행동 기울기 선택지",
    "오해 방식 선택지",
    "지배 기질축 선택지",
    "세계관",
    "WORLD",
    "WORLD_DESC",
    "WORLD_VOCAB",
    "world",
    "world_desc",
    "world_vocab",
}
TASK_G_SUPPRESSED_SECTION_LABELS = {
    "기질 이름",
    "기질 키워드",
    "인물 성격",
    "성격 이름",
    "성격 설명",
    "성격 키워드",
}
NOTEBOOK_RUN_MODES = {
    "preflight": {"max_steps": 0, "max_train_samples": 8, "max_eval_samples": 4},
    "smoke": {"max_steps": 3, "max_train_samples": 32, "max_eval_samples": 16},
    "longer_smoke": {"max_steps": 25, "max_train_samples": 256, "max_eval_samples": 64},
}
VALID_EMOTIONS = {"joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"}
VALID_REGISTERS = {"haera", "hao", "hae"}
VALID_TRANSITION_TYPES = {"gradual", "sudden", "sustained"}
VALID_ACTION_TENDENCIES = {"mobilize", "defend", "wait", "retreat", "celebrate", "mourn"}
VALID_MISINTERPRETATION_TYPES = {
    "overconfident_literal",
    "cautious_reversal",
    "optimistic_expansion",
    "passive_deferral",
    "symbolic_abstraction",
}
TASK_ALLOWED_KEYS = {
    "A": ("text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"),
    "B": ("text_ko", "text_en", "register", "emotion_expressed", "intensity", "mimetics", "temperament_influence"),
    "C": ("speech_ko", "speech_en", "register", "emotion_expressed", "speaker_role", "temperament_tone"),
    "E": ("action_id", "confidence", "hint_ko", "hint_en", "personality_reasoning", "temperament_factor"),
    "F": ("emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type", "temperament_amplifier"),
    "G": ("interpretation_ko", "interpretation_en", "action_tendency", "confidence", "register", "misinterpretation_type", "temperament_bias"),
    "H": ("name", "description_en", "resource_modifiers", "special_zones", "special_resources", "agent_modifiers"),
    "I": ("priority_id", "reasoning_ko", "reasoning_en", "need_addressed", "urgency"),
    "J": ("coping_id", "coping_type", "stress_delta", "hint_ko", "hint_en", "side_effect"),
    "K": ("social_action_id", "trust_delta", "hint_ko", "hint_en", "relationship_intent", "reciprocity_expectation"),
    "L": ("response_id", "trust_delta", "hint_ko", "hint_en", "forgiveness_threshold", "social_memory"),
    "M": ("decision_id", "confidence", "dissent_risk", "reasoning_ko", "reasoning_en", "resource_commitment", "timeline"),
    "N": ("accept", "counter_offer_give", "counter_offer_want", "hint_ko", "hint_en", "negotiation_stance", "walk_away_threshold"),
}
TASK_ENUM_CONSTRAINTS = {
    "A": (
        "register must be exactly one of: haera, hao, hae",
        "dominant_trait must be exactly one of: novelty_seeking, harm_avoidance, reward_dependence, persistence",
    ),
    "B": (
        "register must be exactly one of: haera, hao, hae",
        "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation",
    ),
    "C": (
        "register must be exactly one of: haera, hao, hae",
        "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation",
        "speaker_role must be exactly one of: elder, hunter, shaman, warrior, healer, gatherer, craftsman, chief, scout, observer",
    ),
    "E": ("personality_reasoning must be exactly one of: high_NS, high_HA, high_RD, high_P",),
    "F": (
        "emotion must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation",
        "previous_emotion must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation",
        "transition_type must be exactly one of: gradual, sudden, sustained",
    ),
    "G": (
        "register must be exactly one of: haera, hao, hae",
        "action_tendency must be exactly one of: mobilize, defend, wait, retreat, celebrate, mourn",
        "misinterpretation_type must be exactly one of: overconfident_literal, cautious_reversal, optimistic_expansion, passive_deferral, symbolic_abstraction",
    ),
    "H": (),
    "I": (
        "need_addressed must be exactly one of: hunger, thirst, warmth, rest, safety, belonging, esteem, curiosity, reproduction, comfort, purpose, transcendence, play",
    ),
    "J": (
        "coping_type must be exactly one of: active_avoidance, emotional_release, social_support, ritualistic, substance, acceptance, aggression",
        "side_effect must be exactly one of: aggression_increase, isolation, faith_increase, trust_decrease, morale_boost, exhaustion, none",
    ),
    "K": (
        "relationship_intent must be exactly one of: alliance, cautious_observation, hostile, submissive, dominant, trade_partner, ignore",
        "reciprocity_expectation must be exactly one of: none, gift, service, alliance",
    ),
    "L": (
        "social_memory must be exactly one of: theft_betrayal, aid_gratitude, shared_danger, insult_resentment, gift_goodwill, combat_respect, abandonment, none",
    ),
    "M": (
        "resource_commitment must be exactly one of: food, tools, labor, weapons, none",
        "timeline must be exactly one of: immediate, delayed, conditional",
    ),
    "N": (
        "negotiation_stance must be exactly one of: generous, fair, hard_bargain, exploitative",
        "accept must be true or false",
    ),
}
TASK_GENERATION_RULES = {
    "A": (
        "text_ko and text_en must be concrete persona descriptions, not labels or placeholders.",
        "Do not write self-introduction, dialogue labels, or trait names alone.",
    ),
    "B": (
        "text_ko and text_en must be real emotional reactions, not length instructions or schema text.",
        "emotion_expressed must be one value only, never an enum list.",
    ),
    "C": (
        "speech_ko and speech_en must be direct utterances only, not explanations or copied instructions.",
        "emotion_expressed must be one string value only, never an array or enum list.",
    ),
    "E": (
        "hint_ko and hint_en must be concrete reasons, not template wording or length instructions.",
        "Only action_id and confidence are numeric; every other field is a string.",
    ),
    "F": (
        "cause_ko and cause_en must be concrete causes, not copied rule text or choice lists.",
        "emotion and previous_emotion must be enum strings, never numbers or lists.",
    ),
    "G": (
        "interpretation_ko must be exactly one Korean sentence that interprets the oracle meaning only.",
        "Do not describe personality, situation summary, reasoning steps, or self-introduction.",
        "Do not copy enum lists or placeholder text into action_tendency or misinterpretation_type.",
    ),
    "H": (
        "Output only the worldbuilding IR object and use [] for empty lists.",
        "Do not copy schema descriptions, example prose, or title-case placeholder text into values.",
    ),
    "I": (
        "reasoning_ko must be pure Korean only with no English words or placeholders.",
        "need_addressed must be one allowed need name and urgency must stay between 0.0 and 1.0.",
    ),
    "J": (
        "hint_ko must be pure Korean only with no English words or copied rules.",
        "stress_delta must be between -1.0 and 0.0 and side_effect must be one allowed enum value.",
    ),
    "K": (
        "hint_ko must be pure Korean only with no English words or copied schema text.",
        "trust_delta must stay between -0.5 and 0.5 and relationship_intent must be one enum value.",
    ),
    "L": (
        "hint_ko must be pure Korean only with no English words or copied schema text.",
        "forgiveness_threshold must stay between 0.0 and 1.0 and social_memory must be one enum value.",
    ),
    "M": (
        "reasoning_ko must be pure Korean only with no English words or placeholders.",
        "resource_commitment and timeline must each be one allowed enum value only.",
    ),
    "N": (
        "hint_ko must be pure Korean only with no English words or copied rules.",
        "counter_offer_give and counter_offer_want must be compact item:count strings such as fur:2.",
    ),
}
TASK_OUTPUT_EXAMPLES = {
    "A": '{"text_ko":"그는 앞장서되 위험을 먼저 헤아린다.","text_en":"They lead from the front while weighing danger first.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}',
    "B": '{"text_ko":"숨을 죽이고 주위를 살피며 천천히 뒤로 물러난다.","text_en":"They hold their breath, scan the area, and step back slowly.","register":"haera","emotion_expressed":"fear","intensity":0.72,"mimetics":["숨을 죽이고"],"temperament_influence":"caution sharpens retreat"}',
    "C": '{"speech_ko":"지금은 서두르지 말고 불빛 가까이 모여라.","speech_en":"Do not rush now; gather near the firelight.","register":"hao","emotion_expressed":"trust","speaker_role":"elder","temperament_tone":"steady guidance"}',
    "E": '{"action_id":2,"confidence":0.81,"hint_ko":"위협을 정면에서 막는 편이 무리를 지키기 쉽다.","hint_en":"Holding the threat in front protects the group more reliably.","personality_reasoning":"high_NS","temperament_factor":"choleric urgency"}',
    "F": '{"emotion":"fear","intensity":0.68,"cause_ko":"낯선 그림자가 갑자기 가까워졌다.","cause_en":"An unfamiliar shadow suddenly drew near.","previous_emotion":"trust","transition_type":"sudden","temperament_amplifier":"melancholic vigilance"}',
    "G": '{"interpretation_ko":"이 말은 지금은 공격보다 방어를 준비하라고 판단한다.","interpretation_en":"This means it is time to prepare a defense rather than attack.","action_tendency":"defend","confidence":0.77,"register":"hao","misinterpretation_type":"cautious_reversal","temperament_bias":"melancholic caution"}',
    "H": '{"name":"AmberGrove","description_en":"A sheltered grove with rich soil and mild air.","resource_modifiers":[{"target":"berries","multiplier":1.2}],"special_zones":[],"special_resources":[],"agent_modifiers":[]}',
    "I": '{"priority_id":1,"reasoning_ko":"비바람부터 막아야 온몸이 산다.","reasoning_en":"Securing shelter comes first if the whole body is to endure.","need_addressed":"safety","urgency":0.93}',
    "J": '{"coping_id":2,"coping_type":"social_support","stress_delta":-0.34,"hint_ko":"가까운 이를 붙들면 마음이 덜 흔들린다.","hint_en":"Holding close to trusted people steadies the mind.","side_effect":"morale_boost"}',
    "K": '{"social_action_id":3,"trust_delta":0.21,"hint_ko":"먼저 먹거리를 나누면 손잡을 틈이 열린다.","hint_en":"Sharing food first opens a path toward cooperation.","relationship_intent":"alliance","reciprocity_expectation":"gift"}',
    "L": '{"response_id":4,"trust_delta":-0.28,"hint_ko":"한번 버린 이는 다시 등을 돌릴 수 있다.","hint_en":"Someone who abandoned us once may turn away again.","forgiveness_threshold":0.62,"social_memory":"abandonment"}',
    "M": '{"decision_id":2,"confidence":0.71,"dissent_risk":0.24,"reasoning_ko":"먹거리를 먼저 모아야 무리가 오래 버틴다.","reasoning_en":"Gathering food first gives the band its best chance to endure.","resource_commitment":"labor","timeline":"immediate"}',
    "N": '{"accept":false,"counter_offer_give":"모피:2","counter_offer_want":"뼈칼:3","hint_ko":"이 값이면 우리 몫이 너무 적다.","hint_en":"At this price our side gets too little.","negotiation_stance":"hard_bargain","walk_away_threshold":0.58}',
}
INTERPRETATION_VERBS = ("해석", "판단", "생각", "느끼", "여기")
SEMANTIC_HINTS = {
    "overconfident_literal": ("과신", "단순", "확신"),
    "fear_projection": ("두려움", "위협", "공포"),
    "defensive_bias": ("방어", "정당화"),
}


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    use_qlora: bool
    fallback_reason: str | None
    torch_dtype: str


@dataclass(slots=True)
class SmokeRunConfig:
    run_mode: str = DEFAULT_RUN_MODE
    model_name: str = DEFAULT_MODEL_NAME
    train_file: Path = DEFAULT_TRAIN_FILE
    dev_file: Path = DEFAULT_DEV_FILE
    output_dir: Path | None = None
    max_steps: int = 5
    max_train_samples: int = 32
    max_eval_samples: int = 16
    max_length: int = 512
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    logging_steps: int = 1
    eval_steps: int = 0
    save_steps: int = 0
    save_total_limit: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_TARGET_MODULES))
    seed: int = 42
    trust_remote_code: bool = False
    disable_qlora: bool = False
    require_qlora: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_file"] = str(self.train_file)
        payload["dev_file"] = str(self.dev_file)
        payload["output_dir"] = str(self.output_dir) if self.output_dir is not None else None
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass(slots=True)
class SmokeRunResult:
    success: bool
    status: str
    used_true_qlora: bool
    runtime: dict[str, Any] | None
    environment: dict[str, Any]
    output_dir: str
    summary_path: str
    config_snapshot: str | None
    metrics_path: str | None
    sample_path: str | None
    adapter_dir: str | None
    train_rows: int
    eval_rows: int
    train_task_counts: dict[str, int]
    eval_task_counts: dict[str, int]
    train_loss: float | None
    eval_loss: float | None
    finite_losses: bool | None
    blocker_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmokeRunBlockedError(RuntimeError):
    """Raised when a smoke run cannot proceed under the requested constraints."""


def _normalize_run_mode(run_mode: str) -> str:
    normalized = str(run_mode).strip().lower()
    if normalized not in RUN_MODE_DEFAULTS:
        valid = ", ".join(sorted(RUN_MODE_DEFAULTS))
        raise ValueError(f"Unsupported run mode '{run_mode}'. Expected one of: {valid}")
    return normalized


def _run_mode_defaults(run_mode: str) -> dict[str, Any]:
    normalized = _normalize_run_mode(run_mode)
    defaults = dict(RUN_MODE_DEFAULTS[normalized])
    defaults["run_mode"] = normalized
    return defaults


def coerce_smoke_config(
    value: SmokeRunConfig | argparse.Namespace | Mapping[str, Any],
    *,
    default_run_mode: str = DEFAULT_RUN_MODE,
) -> SmokeRunConfig:
    if isinstance(value, SmokeRunConfig):
        normalized_run_mode = _normalize_run_mode(value.run_mode)
        return SmokeRunConfig(
            run_mode=normalized_run_mode,
            model_name=str(value.model_name),
            train_file=Path(value.train_file),
            dev_file=Path(value.dev_file),
            output_dir=Path(value.output_dir) if value.output_dir is not None else None,
            max_steps=int(value.max_steps),
            max_train_samples=int(value.max_train_samples),
            max_eval_samples=int(value.max_eval_samples),
            max_length=int(value.max_length),
            per_device_train_batch_size=int(value.per_device_train_batch_size),
            per_device_eval_batch_size=int(value.per_device_eval_batch_size),
            gradient_accumulation_steps=int(value.gradient_accumulation_steps),
            learning_rate=float(value.learning_rate),
            logging_steps=int(value.logging_steps),
            eval_steps=int(value.eval_steps),
            save_steps=int(value.save_steps),
            save_total_limit=int(value.save_total_limit),
            lora_r=int(value.lora_r),
            lora_alpha=int(value.lora_alpha),
            lora_dropout=float(value.lora_dropout),
            target_modules=tuple(str(module) for module in value.target_modules),
            seed=int(value.seed),
            trust_remote_code=bool(value.trust_remote_code),
            disable_qlora=bool(value.disable_qlora),
            require_qlora=bool(value.require_qlora),
            dry_run=bool(value.dry_run),
        )

    if isinstance(value, argparse.Namespace):
        payload = vars(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"Unsupported smoke config input: {type(value).__name__}")

    defaults = _run_mode_defaults(str(payload.get("run_mode", default_run_mode)))
    target_modules = payload.get("target_modules", DEFAULT_TARGET_MODULES)
    output_dir = payload.get("output_dir")
    return SmokeRunConfig(
        run_mode=str(defaults["run_mode"]),
        model_name=str(payload.get("model_name", defaults["model_name"])),
        train_file=Path(payload.get("train_file", DEFAULT_TRAIN_FILE)),
        dev_file=Path(payload.get("dev_file", DEFAULT_DEV_FILE)),
        output_dir=Path(output_dir) if output_dir is not None else None,
        max_steps=int(payload.get("max_steps", defaults["max_steps"])),
        max_train_samples=int(payload.get("max_train_samples", defaults["max_train_samples"])),
        max_eval_samples=int(payload.get("max_eval_samples", defaults["max_eval_samples"])),
        max_length=int(payload.get("max_length", 512)),
        per_device_train_batch_size=int(payload.get("per_device_train_batch_size", defaults["per_device_train_batch_size"])),
        per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size", defaults["per_device_eval_batch_size"])),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", defaults["gradient_accumulation_steps"])),
        learning_rate=float(payload.get("learning_rate", defaults["learning_rate"])),
        logging_steps=int(payload.get("logging_steps", defaults["logging_steps"])),
        eval_steps=int(payload.get("eval_steps", defaults["eval_steps"])),
        save_steps=int(payload.get("save_steps", defaults["save_steps"])),
        save_total_limit=int(payload.get("save_total_limit", defaults["save_total_limit"])),
        lora_r=int(payload.get("lora_r", 16)),
        lora_alpha=int(payload.get("lora_alpha", 32)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        target_modules=tuple(str(module) for module in target_modules),
        seed=int(payload.get("seed", 42)),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        disable_qlora=bool(payload.get("disable_qlora", False)),
        require_qlora=bool(payload.get("require_qlora", False)),
        dry_run=bool(payload.get("dry_run", False)),
    )


def load_message_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    validated: list[dict] = []
    for index, row in enumerate(rows, start=1):
        try:
            _validate_messages_row(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{index}: {exc}") from exc
        messages = row["messages"]
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise ValueError(f"{path}:{index}: assistant target must be present and non-empty")
        validated.append(row)
    return validated


def render_conversation(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    rendered: list[str] = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def _load_training_libraries():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def detect_runtime(prefer_qlora: bool, require_qlora: bool) -> RuntimeConfig:
    import torch

    device = "cpu"
    fallback_reason: str | None = None
    torch_dtype = "float32"
    use_qlora = False

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if prefer_qlora and _bitsandbytes_available():
            use_qlora = True
        elif prefer_qlora:
            fallback_reason = "bitsandbytes is unavailable in this environment"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = "float32"
        if prefer_qlora:
            fallback_reason = "Apple Silicon MPS does not support bitsandbytes 4-bit QLoRA"
    elif prefer_qlora:
        fallback_reason = "CUDA is unavailable; true QLoRA requires CUDA + bitsandbytes"

    if require_qlora and not use_qlora:
        raise RuntimeError(f"QLoRA was required but is unavailable: {fallback_reason or 'unknown reason'}")

    return RuntimeConfig(
        device=device,
        use_qlora=use_qlora,
        fallback_reason=fallback_reason,
        torch_dtype=torch_dtype,
    )


def get_environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cwd": str(Path.cwd()),
    }

    try:
        import torch

        torch_info: dict[str, Any] = {
            "available": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_count"] = torch.cuda.device_count()
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            torch_info["cuda_device_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            torch_info["cuda_bf16_supported"] = torch.cuda.is_bf16_supported()
        summary["torch"] = torch_info
    except Exception as exc:  # noqa: BLE001
        summary["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    for module_name in ("transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes"):
        try:
            module = __import__(module_name)
            summary[module_name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:  # noqa: BLE001
            summary[module_name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return summary


def get_true_qlora_preflight() -> dict[str, Any]:
    environment = get_environment_summary()
    try:
        runtime = detect_runtime(prefer_qlora=True, require_qlora=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "runtime": None,
            "environment": environment,
            "blocker_reason": f"{type(exc).__name__}: {exc}",
        }

    return {
        "ok": True,
        "runtime": asdict(runtime),
        "environment": environment,
        "blocker_reason": None,
    }


def resolve_notebook_run_mode(run_mode: str, *, run_id: str | None = None) -> dict[str, Any]:
    normalized = str(run_mode).strip().lower()
    if normalized not in NOTEBOOK_RUN_MODES:
        valid = ", ".join(sorted(NOTEBOOK_RUN_MODES))
        raise ValueError(f"Unsupported RUN_MODE '{run_mode}'. Expected one of: {valid}")

    resolved = dict(NOTEBOOK_RUN_MODES[normalized])
    resolved["run_mode"] = normalized
    resolved["run_id"] = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return resolved


def resolve_baseline_notebook_config(
    run_id: str | None = None,
    *,
    output_root: Path | str | None = None,
    output_dir_override: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _run_mode_defaults("baseline")
    config: dict[str, Any] = {
        "run_mode": "baseline",
        "run_id": run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": False,
        "model_name": defaults["model_name"],
        "dataset": BASELINE_DATASET_ID,
        "train_file": DEFAULT_TRAIN_FILE,
        "dev_file": DEFAULT_DEV_FILE,
        "max_steps": defaults["max_steps"],
        "max_train_samples": defaults["max_train_samples"],
        "max_eval_samples": defaults["max_eval_samples"],
        "per_device_train_batch_size": defaults["per_device_train_batch_size"],
        "per_device_eval_batch_size": defaults["per_device_eval_batch_size"],
        "gradient_accumulation_steps": defaults["gradient_accumulation_steps"],
        "learning_rate": defaults["learning_rate"],
        "logging_steps": defaults["logging_steps"],
        "eval_steps": defaults["eval_steps"],
        "save_steps": defaults["save_steps"],
        "save_total_limit": defaults["save_total_limit"],
        "require_qlora": True,
        "seed": 42,
    }
    if overrides:
        config.update(dict(overrides))
    resolved_output_root = Path(output_root) if output_root is not None else Path(defaults["output_root"])
    if output_dir_override is not None:
        config["output_dir"] = Path(output_dir_override)
    else:
        config["output_dir"] = Path(config.get("output_dir") or resolved_output_root / str(config["run_id"]))
    config["train_file"] = Path(config["train_file"])
    config["dev_file"] = Path(config["dev_file"])
    return config


def _torch_dtype(runtime: RuntimeConfig, torch: Any) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[runtime.torch_dtype]


def _tokenize_rows(rows: list[dict], tokenizer: Any, max_length: int, Dataset: Any) -> Any:
    texts = [render_conversation(tokenizer, row["messages"], add_generation_prompt=False) for row in rows]
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return Dataset.from_dict(dict(encoded))


def _load_model_and_tokenizer(config: SmokeRunConfig, runtime: RuntimeConfig, libs: dict[str, Any]) -> tuple[Any, Any]:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForCausalLM = libs["AutoModelForCausalLM"]
    BitsAndBytesConfig = libs["BitsAndBytesConfig"]
    prepare_model_for_kbit_training = libs["prepare_model_for_kbit_training"]
    get_peft_model = libs["get_peft_model"]
    LoraConfig = libs["LoraConfig"]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if runtime.use_qlora:
        quant_dtype = _torch_dtype(runtime, torch)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = _torch_dtype(runtime, torch)

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if runtime.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model, tokenizer


def pick_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    task_first: list[dict] = []
    seen_tasks: set[str] = set()
    for row in rows:
        task = str(row.get("task", "unknown"))
        if task not in seen_tasks:
            task_first.append(row)
            seen_tasks.add(task)
    if len(task_first) >= limit:
        return task_first[:limit]

    remaining = [row for row in rows if row not in task_first]
    rng.shuffle(remaining)
    return task_first + remaining[: limit - len(task_first)]


def _count_tasks(rows: list[dict]) -> dict[str, int]:
    counter = Counter(row.get("task", "unknown") for row in rows)
    return dict(sorted(counter.items()))


def _resolve_output_dir(base_output_dir: Path | None, run_mode: str) -> Path:
    if base_output_dir is not None:
        return ensure_directory(base_output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    defaults = _run_mode_defaults(run_mode)
    return ensure_directory(Path(defaults["output_root"]) / timestamp)


def build_training_arguments_kwargs(
    runtime: RuntimeConfig,
    *,
    available_parameters: set[str],
    output_dir: str,
    max_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    logging_steps: int,
    eval_steps: int,
    save_steps: int,
    save_total_limit: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_steps": logging_steps,
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    if eval_steps > 0:
        if "eval_strategy" in available_parameters:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in available_parameters:
            kwargs["evaluation_strategy"] = "steps"
        if "eval_steps" in available_parameters:
            kwargs["eval_steps"] = eval_steps
    else:
        if "eval_strategy" in available_parameters:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in available_parameters:
            kwargs["evaluation_strategy"] = "no"
    if save_steps > 0:
        if "save_strategy" in available_parameters:
            kwargs["save_strategy"] = "steps"
        if "save_steps" in available_parameters:
            kwargs["save_steps"] = save_steps
        if "save_total_limit" in available_parameters:
            kwargs["save_total_limit"] = save_total_limit
    elif "save_strategy" in available_parameters:
        kwargs["save_strategy"] = "no"
    if runtime.device == "cpu" and "use_cpu" in available_parameters:
        kwargs["use_cpu"] = True
    elif runtime.device == "cpu" and "no_cuda" in available_parameters:
        kwargs["no_cuda"] = True
    elif runtime.device == "mps" and "use_mps_device" in available_parameters:
        kwargs["use_mps_device"] = True
    return kwargs


def build_trainer_kwargs(
    *,
    available_parameters: set[str],
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    if "processing_class" in available_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in available_parameters:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def _select_generation_rows(
    train_rows: list[dict],
    eval_rows: list[dict],
    *,
    per_task: int = SAMPLES_PER_TASK,
) -> list[dict]:
    candidates = eval_rows + train_rows
    picked: list[dict] = []
    task_counts = {task: 0 for task in DEFAULT_TASKS}
    for row in candidates:
        task = row.get("task")
        if task not in task_counts:
            continue
        if task_counts[task] >= per_task:
            continue
        picked.append(row)
        task_counts[task] += 1
    return picked


def _strip_labeled_sections(content: str, labels: set[str]) -> str:
    lines = content.splitlines()
    kept: list[str] = []
    skip = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) > 2:
            label = stripped[1:-1].strip()
            skip = label in labels
            if skip:
                continue
        if not skip:
            kept.append(line)

    return "\n".join(kept).strip()


def _sanitize_generation_user_content(content: str, task: str) -> str:
    labels = set(LEAKY_GENERATION_SECTION_LABELS)
    if task == "G":
        labels.update(TASK_G_SUPPRESSED_SECTION_LABELS)
    return _strip_labeled_sections(content, labels)


def _build_sample_prompt_messages(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt_messages = [dict(message) for message in row["messages"][:-1]]
    if prompt_messages and prompt_messages[-1]["role"] == "user":
        task = str(row.get("task", "unknown"))
        prompt_messages[-1] = dict(prompt_messages[-1])
        prompt_messages[-1]["content"] = _sanitize_generation_user_content(str(prompt_messages[-1]["content"]), task)
        prompt_messages[-1]["content"] = (
            prompt_messages[-1]["content"].rstrip()
            + SAMPLE_GENERATION_REMINDER
            + _task_specific_generation_reminder(task)
        )
    return prompt_messages


def _sample_generation_max_new_tokens(task: str) -> int:
    return TASK_MAX_NEW_TOKENS.get(task, 384)


def _sample_generation_assistant_prefix(task: str) -> str:
    if task == "A":
        return "{\"text_ko\": \""
    if task == "B":
        return "{\"text_ko\": \""
    if task == "C":
        return "{\"speech_ko\": \""
    if task == "E":
        return "{\"action_id\": "
    if task == "F":
        return "{\"emotion\": \""
    if task == "G":
        return "{\"interpretation_ko\": \"이 말은 "
    if task == "I":
        return "{\"priority_id\": "
    if task == "J":
        return "{\"coping_id\": "
    if task == "K":
        return "{\"social_action_id\": "
    if task == "L":
        return "{\"response_id\": "
    if task == "M":
        return "{\"decision_id\": "
    if task == "N":
        return "{\"accept\": "
    return "{"


def _sample_generation_output_schema(task: str) -> Any | None:
    return TASK_OUTPUT_SCHEMAS.get(task)


def _task_specific_generation_reminder(task: str) -> str:
    allowed_keys = TASK_ALLOWED_KEYS.get(task, ())
    enum_rules = TASK_ENUM_CONSTRAINTS.get(task, ())
    task_rules = TASK_GENERATION_RULES.get(task, ())
    example_output = TASK_OUTPUT_EXAMPLES.get(task)

    sections = []
    if allowed_keys:
        sections.append(
            "[ALLOWED JSON KEYS]\n"
            + ", ".join(allowed_keys)
            + "\n- Only output the keys listed above.\n- Do not output any other keys.\n"
        )
    if enum_rules:
        sections.append("[ENUM CONSTRAINTS]\n" + "\n".join(f"- {rule}" for rule in enum_rules) + "\n")
    if task_rules:
        sections.append("[TASK RULES]\n" + "\n".join(f"- {rule}" for rule in task_rules) + "\n")
    if example_output:
        sections.append("[EXAMPLE OUTPUT]\n" + example_output + "\n")
    if not sections:
        return ""
    return "\n" + "\n".join(section.rstrip() for section in sections)


def _json_object_complete(text: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    saw_open = False

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue

        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
            saw_open = True
        elif char == "}":
            depth -= 1
            if saw_open and depth == 0:
                return True

    return False


def _normalize_generation_candidate(text: str) -> dict[str, Any]:
    normalized_text, normalization = _trim_trivial_json_tail(text)
    normalized_text, extra_normalization = _trim_follow_on_json_object(normalized_text)
    if extra_normalization:
        normalization = extra_normalization if normalization is None else f"{normalization}+{extra_normalization}"
    return {
        "text": normalized_text,
        "normalization": normalization,
        "normalization_details": [],
    }


def _generate_sample_once(
    *,
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    assistant_prefix: str,
    max_new_tokens: int,
    encoded_inputs: dict[str, Any] | None,
    device: str,
    torch: Any,
) -> str:
    from transformers import StoppingCriteria, StoppingCriteriaList

    encoded = tokenizer(prompt_text, return_tensors="pt") if encoded_inputs is None else encoded_inputs
    encoded = {key: value.to(device) for key, value in encoded.items()}

    class JsonObjectStopper(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001, D401
            generated_tokens = input_ids[0][encoded["input_ids"].shape[1] :]
            generated_text = assistant_prefix + tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            return _json_object_complete(generated_text)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=STRUCTURED_GENERATION_DEFAULTS["do_sample"],
            temperature=STRUCTURED_GENERATION_DEFAULTS["temperature"],
            top_p=STRUCTURED_GENERATION_DEFAULTS["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([JsonObjectStopper()]),
        )
    generated_tokens = generated[0][encoded["input_ids"].shape[1] :]
    return assistant_prefix + tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _create_outlines_model(model: Any, tokenizer: Any) -> Any | None:
    """Wrap a Transformers model for outlines constrained decoding once per run."""
    if not OUTLINES_AVAILABLE or model is None or tokenizer is None:
        return None
    try:
        from_transformers = getattr(_outlines_module, "from_transformers", None)
        if from_transformers is None:
            return None
        return from_transformers(model, tokenizer)
    except Exception:
        return None


def _generate_sample_outlines(
    *,
    outlines_model: Any,
    schema: type[BaseModel],
    prompt_text: str,
    max_new_tokens: int,
) -> str:
    """Generate a single structured JSON sample through outlines constrained decoding."""
    generator_cls = getattr(_outlines_module, "Generator", None)
    if generator_cls is None:
        raise RuntimeError("Outlines Generator API unavailable")

    schema_target: Any = schema
    json_schema_builder = getattr(_outlines_module, "json_schema", None)
    if json_schema_builder is not None:
        try:
            schema_target = json_schema_builder(schema)
        except Exception:
            schema_target = schema

    generator = generator_cls(outlines_model, schema_target)
    result = generator(
        prompt_text,
        max_new_tokens=max_new_tokens,
        repetition_penalty=OUTLINES_REPETITION_PENALTY,
    )
    if isinstance(result, dict):
        return json.dumps(result, ensure_ascii=False)
    if isinstance(result, BaseModel):
        return result.model_dump_json(by_alias=True)
    return str(result)


def _generate_samples(
    model: Any,
    tokenizer: Any,
    rows: list[dict],
    output_path: Path,
    runtime: RuntimeConfig,
    torch: Any,
) -> tuple[list[dict], dict[str, Any]]:
    if not rows:
        write_jsonl(output_path, [])
        return [], BatchMetrics().summary()

    samples: list[dict] = []
    metrics_collector = BatchMetrics()
    model.eval()
    model.config.use_cache = True
    device = runtime.device
    outlines_model = _create_outlines_model(model, tokenizer)
    if outlines_model is not None:
        print("[outlines] Constrained decoding ENABLED (model wrapped successfully)")
    else:
        print("[outlines] NOT available — using fallback repair/sanitize pipeline")

    for row in rows:
        task = str(row.get("task", "unknown"))
        prompt_messages = _build_sample_prompt_messages(row)
        assistant_prefix = _sample_generation_assistant_prefix(task)
        prompt_text = render_conversation(tokenizer, prompt_messages, add_generation_prompt=True)
        max_new_tokens = _sample_generation_max_new_tokens(task)
        schema = _sample_generation_output_schema(task)
        normalization_details: list[dict[str, str]] = []
        normalization: str | None = None
        parse_error = None
        validation_error = None
        structured_attempt_count = 1
        structured_attempts: list[dict[str, Any]] = []
        structured_validation_metadata: dict[str, Any] | None = None
        structured_repair_applied = False
        structured_repair_actions: list[dict[str, Any]] = []
        structured_decoding: dict[str, Any] | None = None
        outlines_fallback_reason: str | None = None
        sample_success = False

        def llm(generation_prompt: str) -> str:
            return _generate_sample_once(
                model=model,
                tokenizer=tokenizer,
                prompt_text=generation_prompt + assistant_prefix,
                assistant_prefix=assistant_prefix,
                max_new_tokens=max_new_tokens,
                encoded_inputs=None,
                device=device,
                torch=torch,
            )

        try:
            if schema is None:
                raw_generated_text = llm(prompt_text)
                normalized = _normalize_generation_candidate(raw_generated_text)
                generated_text = normalized["text"]
                normalization = normalized["normalization"]
                payload = json.loads(generated_text)
                generated_text = json.dumps(payload, ensure_ascii=False)
                sample_success = True
            elif outlines_model is not None:
                outlines_last_error: Exception | None = None
                outlines_attempt_metrics: list[GenerationAttemptMetrics] = []
                outlines_attempt_records: list[dict[str, Any]] = []
                for outlines_attempt_idx in range(2):
                    raw_generated_text = ""
                    sanitize_actions: list[dict[str, Any]] = []
                    keys_removed: list[str] = []
                    enum_normalizations: list[str] = []
                    try:
                        raw_generated_text = _generate_sample_outlines(
                            outlines_model=outlines_model,
                            schema=schema,
                            prompt_text=prompt_text,
                            max_new_tokens=max_new_tokens,
                        )
                        payload = json.loads(raw_generated_text)
                        sanitized_payload, sanitize_actions = sanitize_json_output(payload, task)
                        keys_removed = [
                            removed_key
                            for action in sanitize_actions
                            if action.get("kind") == "filter_extra_keys"
                            for removed_key in action.get("removed_keys", [])
                        ]
                        enum_normalizations = [
                            normalized_value
                            for action in sanitize_actions
                            if action.get("kind") == "normalize_enum_values"
                            for normalized_value in action.get("normalized", [])
                        ]
                        if sanitize_actions:
                            structured_repair_applied = True
                            structured_repair_actions = list(sanitize_actions)
                        validated = schema.model_validate(sanitized_payload)
                        generated_text = json.dumps(validated.model_dump(mode="json", by_alias=True), ensure_ascii=False)
                        structured_decoding = {
                            "requested_mode": "outlines_json_schema",
                            "used_mode": "outlines_json_schema",
                            "enabled": True,
                            "supported": True,
                            "reason": None,
                        }
                        outlines_attempt = GenerationAttemptMetrics(
                            task_id=task,
                            attempt_number=outlines_attempt_idx + 1,
                            raw_length=len(raw_generated_text),
                            repairs_applied=[action.get("kind", "unknown") for action in sanitize_actions],
                            keys_removed=keys_removed,
                            enums_normalized=enum_normalizations,
                            json_parse_success=True,
                            schema_validation_success=True,
                            overall_success=True,
                        )
                        metrics_collector.record(outlines_attempt)
                        outlines_attempt_metrics.append(outlines_attempt)
                        outlines_attempt_records.append(
                            {
                                "attempt_index": outlines_attempt_idx,
                                "raw_output": raw_generated_text,
                                "candidate_output": generated_text,
                                "json_error": None,
                                "validation_error": None,
                                "error_kind": None,
                                "repair_actions": sanitize_actions,
                                "keys_removed": keys_removed,
                                "enum_normalizations": enum_normalizations,
                            }
                        )
                        structured_attempt_count = outlines_attempt_idx + 1
                        structured_attempts = list(outlines_attempt_records)
                        structured_validation_metadata = {
                            "attempt_count": structured_attempt_count,
                            "last_error_kind": None,
                            "attempts": structured_attempts,
                            "repair_actions": structured_repair_actions,
                            "structured_decoding": structured_decoding,
                            "attempt_metrics": [
                                {
                                    "task_id": metric.task_id,
                                    "attempt_number": metric.attempt_number,
                                    "raw_length": metric.raw_length,
                                    "repairs_applied": metric.repairs_applied,
                                    "keys_removed": metric.keys_removed,
                                    "enums_normalized": metric.enums_normalized,
                                    "json_parse_success": metric.json_parse_success,
                                    "schema_validation_success": metric.schema_validation_success,
                                    "validation_error": metric.validation_error,
                                    "overall_success": metric.overall_success,
                                }
                                for metric in outlines_attempt_metrics
                            ],
                        }
                        sample_success = True
                        break
                    except (json.JSONDecodeError, ValidationError) as outlines_validation_exc:
                        outlines_last_error = outlines_validation_exc
                        validation_text = str(outlines_validation_exc)
                        outlines_attempt = GenerationAttemptMetrics(
                            task_id=task,
                            attempt_number=outlines_attempt_idx + 1,
                            raw_length=len(raw_generated_text),
                            repairs_applied=[action.get("kind", "unknown") for action in sanitize_actions],
                            keys_removed=keys_removed,
                            enums_normalized=enum_normalizations,
                            json_parse_success=not isinstance(outlines_validation_exc, json.JSONDecodeError),
                            schema_validation_success=False,
                            validation_error=validation_text,
                            overall_success=False,
                            retry_exhausted=outlines_attempt_idx == 1,
                        )
                        metrics_collector.record(outlines_attempt)
                        outlines_attempt_metrics.append(outlines_attempt)
                        outlines_attempt_records.append(
                            {
                                "attempt_index": outlines_attempt_idx,
                                "raw_output": raw_generated_text,
                                "candidate_output": raw_generated_text,
                                "json_error": validation_text if isinstance(outlines_validation_exc, json.JSONDecodeError) else None,
                                "validation_error": None if isinstance(outlines_validation_exc, json.JSONDecodeError) else validation_text,
                                "error_kind": "json" if isinstance(outlines_validation_exc, json.JSONDecodeError) else "validation",
                                "repair_actions": sanitize_actions,
                                "keys_removed": keys_removed,
                                "enum_normalizations": enum_normalizations,
                            }
                        )
                        structured_attempt_count = outlines_attempt_idx + 1
                        continue
                    except Exception as outlines_exc:  # noqa: BLE001
                        outlines_last_error = outlines_exc
                        outlines_fallback_reason = f"outlines generation failed: {outlines_exc}"
                        break
                else:
                    outlines_last_error = None

                if not sample_success:
                    if outlines_fallback_reason is not None:
                        structured = generate_structured(
                            llm,
                            prompt_text,
                            schema,
                            output_normalizer=_normalize_generation_candidate,
                            structured_constraint=build_structured_constraint(schema),
                            task_id=task,
                            metrics_collector=metrics_collector,
                        )
                        raw_generated_text = structured.raw_output
                        generated_text = json.dumps(structured.payload, ensure_ascii=False)
                        normalization = structured.normalization
                        normalization_details = list(structured.normalization_details or [])
                        structured_attempt_count = structured.attempt_count
                        structured_repair_actions = list(structured.repair_actions)
                        structured_repair_applied = bool(structured.repair_actions)
                        structured_attempts = [
                            {
                                "attempt_index": attempt.attempt_index,
                                "raw_output": attempt.raw_output,
                                "candidate_output": attempt.candidate_output,
                                "json_error": attempt.json_error,
                                "validation_error": attempt.validation_error,
                                "error_kind": attempt.error_kind,
                                "repair_actions": attempt.repair_actions,
                                "keys_removed": attempt.keys_removed,
                                "enum_normalizations": attempt.enum_normalizations,
                            }
                            for attempt in structured.attempts
                        ]
                        structured_decoding = {
                            "requested_mode": "outlines_json_schema",
                            "used_mode": "repair_sanitize_fallback",
                            "enabled": True,
                            "supported": True,
                            "reason": outlines_fallback_reason,
                        }
                        structured_validation_metadata = {
                            "attempt_count": structured.attempt_count,
                            "last_error_kind": structured.last_error_kind,
                            "attempts": structured_attempts,
                            "repair_actions": structured.repair_actions,
                            "structured_decoding": structured_decoding,
                            "attempt_metrics": [
                                {
                                    "task_id": metric.task_id,
                                    "attempt_number": metric.attempt_number,
                                    "raw_length": metric.raw_length,
                                    "repairs_applied": metric.repairs_applied,
                                    "keys_removed": metric.keys_removed,
                                    "enums_normalized": metric.enums_normalized,
                                    "json_parse_success": metric.json_parse_success,
                                    "schema_validation_success": metric.schema_validation_success,
                                    "validation_error": metric.validation_error,
                                    "overall_success": metric.overall_success,
                                }
                                for metric in structured.attempt_metrics
                            ],
                        }
                        sample_success = True
                    else:
                        raw_generated_text = outlines_attempt_records[-1]["raw_output"] if outlines_attempt_records else ""
                        generated_text = outlines_attempt_records[-1]["candidate_output"] if outlines_attempt_records else ""
                        structured_attempts = list(outlines_attempt_records)
                        structured_repair_actions = list(structured_repair_actions)
                        structured_decoding = {
                            "requested_mode": "outlines_json_schema",
                            "used_mode": "outlines_json_schema",
                            "enabled": True,
                            "supported": True,
                            "reason": str(outlines_last_error) if outlines_last_error else None,
                        }
                        structured_validation_metadata = {
                            "attempt_count": structured_attempt_count,
                            "last_error_kind": "validation" if isinstance(outlines_last_error, ValidationError) else "json",
                            "attempts": structured_attempts,
                            "repair_actions": structured_repair_actions,
                            "structured_decoding": structured_decoding,
                            "attempt_metrics": [
                                {
                                    "task_id": metric.task_id,
                                    "attempt_number": metric.attempt_number,
                                    "raw_length": metric.raw_length,
                                    "repairs_applied": metric.repairs_applied,
                                    "keys_removed": metric.keys_removed,
                                    "enums_normalized": metric.enums_normalized,
                                    "json_parse_success": metric.json_parse_success,
                                    "schema_validation_success": metric.schema_validation_success,
                                    "validation_error": metric.validation_error,
                                    "overall_success": metric.overall_success,
                                }
                                for metric in outlines_attempt_metrics
                            ],
                        }
                        validation_error = str(outlines_last_error) if outlines_last_error else "outlines_validation_failed"
            else:
                structured = generate_structured(
                    llm,
                    prompt_text,
                    schema,
                    output_normalizer=_normalize_generation_candidate,
                    structured_constraint=build_structured_constraint(schema),
                    task_id=task,
                    metrics_collector=metrics_collector,
                )
                raw_generated_text = structured.raw_output
                generated_text = json.dumps(structured.payload, ensure_ascii=False)
                normalization = structured.normalization
                normalization_details = list(structured.normalization_details)
                structured_attempt_count = structured.attempt_count
                structured_repair_actions = list(structured.repair_actions)
                structured_repair_applied = bool(structured.repair_actions)
                structured_decoding = structured.structured_decoding
                structured_attempts = [
                    {
                        "attempt_index": attempt.attempt_index,
                        "raw_output": attempt.raw_output,
                        "candidate_output": attempt.candidate_output,
                        "json_error": attempt.json_error,
                        "validation_error": attempt.validation_error,
                        "error_kind": attempt.error_kind,
                        "repair_actions": attempt.repair_actions,
                        "keys_removed": attempt.keys_removed,
                        "enum_normalizations": attempt.enum_normalizations,
                    }
                    for attempt in structured.attempts
                ]
                structured_validation_metadata = {
                    "attempt_count": structured.attempt_count,
                    "last_error_kind": structured.last_error_kind,
                    "attempts": structured_attempts,
                    "repair_actions": structured.repair_actions,
                    "structured_decoding": structured.structured_decoding,
                    "attempt_metrics": [
                        {
                            "task_id": metric.task_id,
                            "attempt_number": metric.attempt_number,
                            "raw_length": metric.raw_length,
                            "repairs_applied": metric.repairs_applied,
                            "keys_removed": metric.keys_removed,
                            "enums_normalized": metric.enums_normalized,
                            "json_parse_success": metric.json_parse_success,
                            "schema_validation_success": metric.schema_validation_success,
                            "validation_error": metric.validation_error,
                            "overall_success": metric.overall_success,
                        }
                        for metric in structured.attempt_metrics
                    ],
                }
                sample_success = True
        except StructuredGenerationError as exc:
            raw_generated_text = exc.last_raw_output
            generated_text = exc.last_output
            normalization = exc.normalization
            normalization_details = list(exc.normalization_details or [])
            structured_attempt_count = exc.attempt_count
            structured_repair_actions = list(exc.repair_actions)
            structured_repair_applied = bool(exc.repair_actions)
            structured_decoding = exc.structured_decoding
            if outlines_fallback_reason is not None:
                structured_decoding = {
                    "requested_mode": "outlines_json_schema",
                    "used_mode": "repair_sanitize_fallback",
                    "enabled": True,
                    "supported": True,
                    "reason": outlines_fallback_reason,
                }
            structured_attempts = [
                {
                    "attempt_index": attempt.attempt_index,
                    "raw_output": attempt.raw_output,
                    "candidate_output": attempt.candidate_output,
                    "json_error": attempt.json_error,
                    "validation_error": attempt.validation_error,
                    "error_kind": attempt.error_kind,
                    "repair_actions": attempt.repair_actions,
                    "keys_removed": attempt.keys_removed,
                    "enum_normalizations": attempt.enum_normalizations,
                }
                for attempt in exc.attempts
            ]
            structured_validation_metadata = {
                "attempt_count": exc.attempt_count,
                "last_error_kind": exc.last_error_kind,
                "attempts": structured_attempts,
                "repair_actions": exc.repair_actions,
                "structured_decoding": structured_decoding,
                "attempt_metrics": [
                    {
                        "task_id": metric.task_id,
                        "attempt_number": metric.attempt_number,
                        "raw_length": metric.raw_length,
                        "repairs_applied": metric.repairs_applied,
                        "keys_removed": metric.keys_removed,
                        "enums_normalized": metric.enums_normalized,
                        "json_parse_success": metric.json_parse_success,
                        "schema_validation_success": metric.schema_validation_success,
                        "validation_error": metric.validation_error,
                        "overall_success": metric.overall_success,
                    }
                    for metric in exc.attempt_metrics
                ],
            }
            if exc.last_error_kind == "json":
                parse_error = "JSONDecodeError"
            else:
                validation_error = exc.attempts[-1].validation_error if exc.attempts else "structured_validation_failed"
        except Exception as exc:  # noqa: BLE001
            raw_generated_text = ""
            generated_text = ""
            parse_error = type(exc).__name__
        metrics_collector.record_sample_outcome(sample_success)
        samples.append(
            {
                "task": row.get("task"),
                "source_split": row.get("source_split"),
                "prompt_messages": prompt_messages,
                "expected_assistant": row["messages"][-1]["content"],
                "generated_assistant": generated_text,
                "raw_generated_assistant": raw_generated_text,
                "json_parse_error": parse_error,
                "structured_validation_error": validation_error,
                "structured_validation_metadata": structured_validation_metadata,
                "structured_attempt_count": structured_attempt_count,
                "structured_attempts": structured_attempts,
                "generation_max_new_tokens": max_new_tokens,
                "generation_normalization": normalization,
                "generation_normalization_details": normalization_details,
                "structured_repair_applied": structured_repair_applied,
                "structured_repair_actions": structured_repair_actions,
                "structured_decoding": structured_decoding,
            }
        )

    write_jsonl(output_path, samples)
    return samples, metrics_collector.summary()


def preview_metrics(output_dir: Path | str) -> dict[str, Any]:
    metrics_path = Path(output_dir) / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_json_artifact(output_dir: Path | str, artifact_name: str) -> dict[str, Any]:
    artifact_path = Path(output_dir) / artifact_name
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def load_optional_json_artifact(output_dir: Path | str, artifact_name: str) -> dict[str, Any] | None:
    artifact_path = Path(output_dir) / artifact_name
    if not artifact_path.exists():
        return None
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def load_sample_generations(output_dir: Path | str) -> list[dict]:
    sample_path = Path(output_dir) / "sample_generations.jsonl"
    return read_jsonl(sample_path) if sample_path.exists() else []


def load_model_registry(path: Path | str = MODEL_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"runs": []}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def register_baseline_run(
    registry_path: Path | str = MODEL_REGISTRY_PATH,
    *,
    config: Mapping[str, Any],
    result: SmokeRunResult | Mapping[str, Any],
    analysis_report: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any] | None:
    result_payload = result.to_dict() if isinstance(result, SmokeRunResult) else dict(result)
    if str(result_payload.get("status", "")) != "ok" or not result_payload.get("adapter_dir"):
        return None

    registry_file = Path(registry_path)
    registry = load_model_registry(registry_file)
    entry = {
        "run_id": str(config.get("run_id", "")),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "mode": "baseline",
        "status": str(result_payload.get("status", "")),
        "model_name": str(config.get("model_name", "")),
        "dataset": str(config.get("dataset", BASELINE_DATASET_ID)),
        "steps": int(config.get("max_steps", 0) or 0),
        "output_dir": str(result_payload.get("output_dir", "")),
        "adapter_dir": str(result_payload.get("adapter_dir", "")),
        "used_true_qlora": bool(result_payload.get("used_true_qlora")),
        "train_loss": result_payload.get("train_loss"),
        "eval_loss": result_payload.get("eval_loss"),
        "analysis_report_path": str(config.get("analysis_report_path", "")) or None,
        "sample_path": str(result_payload.get("sample_path", "")) or None,
        "analyzer_overall_status": str(analysis_report.get("overall_status", "")) if analysis_report else None,
        "metrics": {
            "train_loss": result_payload.get("train_loss"),
            "eval_loss": result_payload.get("eval_loss"),
            "semantic_low_quality": analysis_report.get("semantic_low_quality_count") if analysis_report else None,
            "malformed_json": analysis_report.get("malformed_json_count") if analysis_report else None,
            "retry_rate": metrics.get("retry_rate") if metrics else None,
        },
    }
    registry["runs"] = [run for run in registry.get("runs", []) if run.get("run_id") != entry["run_id"]]
    registry["runs"].append(entry)
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    registry_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def select_best_adapter_run(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = [
        run
        for run in registry.get("runs", [])
        if str(run.get("status", "")) == "ok" and run.get("adapter_dir")
    ]
    if not candidates:
        return None

    analyzer_priority = {
        "structurally_usable": 0,
        "semantic_quality_issue": 1,
        "enum_instability": 2,
        "prompt_leakage_issue": 3,
        "structure_failure": 4,
    }

    def score(run: Mapping[str, Any]) -> tuple[Any, ...]:
        metrics = run.get("metrics", {}) if isinstance(run.get("metrics"), Mapping) else {}
        semantic_low_quality = metrics.get("semantic_low_quality")
        eval_loss = metrics.get("eval_loss")
        return (
            analyzer_priority.get(str(run.get("analyzer_overall_status", "")), 99),
            semantic_low_quality if semantic_low_quality is not None else math.inf,
            eval_loss if eval_loss is not None else math.inf,
            str(run.get("created_at", "")),
            str(run.get("run_id", "")),
        )

    return min(candidates, key=score)


def update_best_adapter_pointer(pointer_path: Path | str = BEST_ADAPTER_POINTER_PATH, best_run: Mapping[str, Any] | None = None) -> str | None:
    if best_run is None or not best_run.get("adapter_dir"):
        return None

    pointer_file = Path(pointer_path)
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    adapter_dir = str(best_run["adapter_dir"])
    pointer_file.write_text(adapter_dir, encoding="utf-8")
    return adapter_dir


def count_parseable_json_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = len(samples)
    parseable = sum(1 for row in samples if not row.get("json_parse_error"))
    return {
        "total": total,
        "parseable_json": parseable,
        "failed_json": total - parseable,
    }


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return stripped

    body_lines = lines[1:]
    if body_lines and body_lines[-1].strip() == "```":
        body_lines = body_lines[:-1]
    return "\n".join(body_lines).strip()


def _parse_json_payload(text: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(text), None
    except Exception as exc:  # noqa: BLE001
        return None, type(exc).__name__


def _trim_trivial_json_tail(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text, None

    try:
        _, end_index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return text, None

    trailing = stripped[end_index:]
    if trailing and not trailing.strip(" \t\r\n,"):
        return stripped[:end_index].strip(), "trim_trailing_comma"
    return text, None


def _trim_follow_on_json_object(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return text, None

    try:
        _, end_index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return text, None

    trailing = stripped[end_index:]
    trailing_lstripped = trailing.lstrip()
    if trailing_lstripped.startswith(","):
        trailing_lstripped = trailing_lstripped[1:].lstrip()

    if trailing_lstripped.startswith(("{", "[")):
        return stripped[:end_index].strip(), "trim_follow_on_json_object"
    return text, None


def _normalize_enum_candidate(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _normalize_known_enum_values(task: str, payload: Any) -> tuple[Any, list[dict[str, str]]]:
    if not isinstance(payload, dict):
        return payload, []

    valid_fields: dict[str, set[str]]
    if task in {"B", "C"}:
        valid_fields = {
            "register": VALID_REGISTERS,
            "emotion_expressed": VALID_EMOTIONS,
        }
    elif task == "F":
        valid_fields = {
            "emotion": VALID_EMOTIONS,
            "previous_emotion": VALID_EMOTIONS,
            "transition_type": VALID_TRANSITION_TYPES,
        }
    elif task == "G":
        valid_fields = {
            "register": VALID_REGISTERS,
            "action_tendency": VALID_ACTION_TENDENCIES,
            "misinterpretation_type": VALID_MISINTERPRETATION_TYPES,
        }
    else:
        return payload, []

    normalized_payload = dict(payload)
    details: list[dict[str, str]] = []
    for field_name, valid_values in valid_fields.items():
        current = normalized_payload.get(field_name)
        if not isinstance(current, str):
            continue
        candidate = _normalize_enum_candidate(current)
        if candidate != current and candidate in valid_values:
            normalized_payload[field_name] = candidate
            details.append({"field": field_name, "from": current, "to": candidate})

    return normalized_payload, details


def _has_trailing_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        _, index = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    return bool(stripped[index:].strip())


def _categorize_generation_failure(
    *,
    raw_text: str,
    stripped_text: str,
    raw_parseable: bool,
    stripped_parseable: bool,
    fenced_json: bool,
    enum_drift_total: int,
) -> str:
    if raw_parseable and enum_drift_total == 0:
        return "ok"
    if raw_parseable and enum_drift_total > 0:
        return "enum_drift"
    if fenced_json and stripped_parseable and enum_drift_total == 0:
        return "fenced_json"
    if fenced_json and stripped_parseable and enum_drift_total > 0:
        return "fenced_json_with_enum_drift"

    normalized = stripped_text.strip()
    if not normalized:
        return "non_json_answer"
    if normalized[0] not in "{[":
        return "non_json_answer"
    if _has_trailing_text(normalized):
        return "trailing_text"

    unbalanced_object = normalized.count("{") > normalized.count("}")
    unbalanced_array = normalized.count("[") > normalized.count("]")
    missing_terminal = not normalized.endswith(("}", "]"))
    if not stripped_parseable and (unbalanced_object or unbalanced_array or missing_terminal):
        return "truncation"
    if not stripped_parseable:
        return "invalid_syntax"
    return "ok"


def _enum_drift_issues(task: str, payload: Any) -> list[tuple[str, str]]:
    if not isinstance(payload, dict):
        return []

    def invalid_issue(field_name: str, value: Any, valid_values: set[str]) -> tuple[str, str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return None if value in valid_values else (field_name, value)
        if isinstance(value, (list, dict)):
            return (field_name, json.dumps(value, ensure_ascii=False))
        return (field_name, str(value))

    issues: list[tuple[str, str]] = []
    if task in {"B", "C"}:
        emotion_value = payload.get("emotion_expressed")
        register_value = payload.get("register")
        issue = invalid_issue("emotion_expressed", emotion_value, VALID_EMOTIONS)
        if issue:
            issues.append(issue)
        issue = invalid_issue("register", register_value, VALID_REGISTERS)
        if issue:
            issues.append(issue)
    elif task == "F":
        emotion_value = payload.get("emotion")
        transition_type = payload.get("transition_type")
        issue = invalid_issue("emotion", emotion_value, VALID_EMOTIONS)
        if issue:
            issues.append(issue)
        issue = invalid_issue("transition_type", transition_type, VALID_TRANSITION_TYPES)
        if issue:
            issues.append(issue)
    elif task == "G":
        action_tendency = payload.get("action_tendency")
        register_value = payload.get("register")
        misinterpretation_type = payload.get("misinterpretation_type")
        issue = invalid_issue("action_tendency", action_tendency, VALID_ACTION_TENDENCIES)
        if issue:
            issues.append(issue)
        issue = invalid_issue("register", register_value, VALID_REGISTERS)
        if issue:
            issues.append(issue)
        issue = invalid_issue("misinterpretation_type", misinterpretation_type, VALID_MISINTERPRETATION_TYPES)
        if issue:
            issues.append(issue)
    return issues


def _hangul_ratio(text: str) -> float:
    significant = [char for char in text if not char.isspace() and (char.isalpha() or ("\uac00" <= char <= "\ud7a3"))]
    if not significant:
        return 0.0
    hangul = sum(1 for char in significant if "\uac00" <= char <= "\ud7a3")
    return hangul / len(significant)


def validate_g_semantics(sample: Mapping[str, Any]) -> dict[str, Any]:
    raw_text = str(sample.get("generated_assistant", "")).strip()
    stripped_text = strip_json_fence(raw_text)
    payload, _ = _parse_json_payload(stripped_text)

    if not isinstance(payload, dict):
        return {
            "semantic_status": "LOW_QUALITY",
            "score": 0.0,
            "reason": "generated_assistant_not_parseable",
        }

    interpretation = str(payload.get("interpretation_ko", "")).strip()
    if not interpretation:
        return {
            "semantic_status": "LOW_QUALITY",
            "score": 0.0,
            "reason": "missing_interpretation_ko",
        }

    hangul_ratio = _hangul_ratio(interpretation)
    if hangul_ratio < 0.6:
        return {
            "semantic_status": "LANGUAGE_DRIFT",
            "score": round(hangul_ratio, 3),
            "reason": f"low_hangul_ratio:{hangul_ratio:.3f}",
        }

    reasons: list[str] = []
    score = 1.0

    if not 12 <= len(interpretation) <= 200:
        reasons.append("length_out_of_range")
        score -= 0.25

    if not any(verb in interpretation for verb in INTERPRETATION_VERBS):
        reasons.append("missing_interpretation_verb")
        score -= 0.25

    misinterpretation_type = str(payload.get("misinterpretation_type", "")).strip()
    hint_words = SEMANTIC_HINTS.get(misinterpretation_type)
    if hint_words and not any(hint in interpretation for hint in hint_words):
        return {
            "semantic_status": "SEMANTIC_DRIFT",
            "score": max(0.0, round(score - 0.35, 3)),
            "reason": f"missing_semantic_hint:{misinterpretation_type}",
        }

    if reasons:
        return {
            "semantic_status": "LOW_QUALITY",
            "score": max(0.0, round(score, 3)),
            "reason": ",".join(reasons),
        }

    return {
        "semantic_status": "VALID",
        "score": round(score, 3),
        "reason": "ok",
    }


def analyze_sample_generation(sample: Mapping[str, Any]) -> dict[str, Any]:
    task = str(sample.get("task", "unknown"))
    raw_text = str(sample.get("generated_assistant", "")).strip()
    stripped_text = strip_json_fence(raw_text)
    fenced_json = stripped_text != raw_text

    raw_payload, raw_error = _parse_json_payload(raw_text)
    stripped_payload, stripped_error = _parse_json_payload(stripped_text)
    payload = raw_payload if raw_payload is not None else stripped_payload
    issues = _enum_drift_issues(task, payload)

    raw_parseable = raw_payload is not None
    stripped_parseable = stripped_payload is not None
    recoverable_fenced = fenced_json and not raw_parseable and stripped_parseable
    malformed_json = not stripped_parseable
    failure_category = _categorize_generation_failure(
        raw_text=raw_text,
        stripped_text=stripped_text,
        raw_parseable=raw_parseable,
        stripped_parseable=stripped_parseable,
        fenced_json=fenced_json,
        enum_drift_total=len(issues),
    )

    if malformed_json:
        classification = "malformed_json"
    elif recoverable_fenced and issues:
        classification = "fenced_recoverable_with_enum_drift"
    elif recoverable_fenced:
        classification = "fenced_recoverable"
    elif issues:
        classification = "enum_drift"
    else:
        classification = "raw_parseable"

    semantic_result = validate_g_semantics(sample) if task == "G" else None

    return {
        "task": task,
        "raw_parseable_json": raw_parseable,
        "fenced_json": fenced_json,
        "fence_stripped_parseable_json": stripped_parseable,
        "recoverable_fenced_json": recoverable_fenced,
        "malformed_json": malformed_json,
        "enum_drift_total": len(issues),
        "enum_drift_fields": [field_name for field_name, _ in issues],
        "classification": classification,
        "failure_category": failure_category,
        "raw_json_parse_error": str(sample.get("json_parse_error") or raw_error) if (sample.get("json_parse_error") or raw_error) else None,
        "stripped_json_parse_error": stripped_error,
        "generated_assistant": raw_text,
        "stripped_assistant": stripped_text,
        "semantic_status": semantic_result["semantic_status"] if semantic_result else None,
        "semantic_score": semantic_result["score"] if semantic_result else None,
        "semantic_reason": semantic_result["reason"] if semantic_result else None,
    }


def summarize_sample_generations(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    json_parse_errors = Counter(str(row.get("json_parse_error") or "ok") for row in samples)
    enum_drift_fields: Counter[str] = Counter()
    enum_drift_examples: list[dict[str, Any]] = []
    recoverable_examples: list[dict[str, Any]] = []
    analyses = [analyze_sample_generation(sample) for sample in samples]
    classifications = Counter(analysis["classification"] for analysis in analyses)
    failure_categories = Counter(analysis["failure_category"] for analysis in analyses)
    semantic_statuses = Counter(
        analysis["semantic_status"]
        for analysis in analyses
        if analysis["task"] == "G" and analysis["semantic_status"] is not None
    )
    retry_count = sum(1 for sample in samples if int(sample.get("structured_attempt_count", 1) or 1) > 1)
    repair_applied_count = sum(1 for sample in samples if bool(sample.get("structured_repair_applied")))
    constrained_decoding_used_count = sum(
        1
        for sample in samples
        if isinstance(sample.get("structured_decoding"), Mapping) and bool(sample["structured_decoding"].get("enabled"))
    )

    for analysis in analyses:
        task = str(analysis["task"])
        for field_name in analysis["enum_drift_fields"]:
            enum_drift_fields[field_name] += 1
            if len(enum_drift_examples) < 5:
                enum_drift_examples.append(
                    {
                        "task": task,
                        "field": field_name,
                        "generated_assistant": analysis["generated_assistant"],
                    }
                )
        if analysis["recoverable_fenced_json"] and len(recoverable_examples) < 3:
            recoverable_examples.append(
                {
                    "task": task,
                    "before": analysis["generated_assistant"],
                    "after": analysis["stripped_assistant"],
                }
            )

    return {
        "total": len(samples),
        "raw_parseable_json": sum(1 for analysis in analyses if analysis["raw_parseable_json"]),
        "fenced_json": sum(1 for analysis in analyses if analysis["fenced_json"]),
        "fence_stripped_parseable_json": sum(1 for analysis in analyses if analysis["fence_stripped_parseable_json"]),
        "recoverable_fenced_json": sum(1 for analysis in analyses if analysis["recoverable_fenced_json"]),
        "malformed_json": sum(1 for analysis in analyses if analysis["malformed_json"]),
        "json_parse_error_types": dict(sorted(json_parse_errors.items())),
        "enum_drift_total": sum(enum_drift_fields.values()),
        "enum_drift_samples": sum(1 for analysis in analyses if analysis["enum_drift_total"]),
        "enum_drift_fields": dict(sorted(enum_drift_fields.items())),
        "enum_drift_examples": enum_drift_examples,
        "recoverable_examples": recoverable_examples,
        "classifications": dict(sorted(classifications.items())),
        "failure_categories": dict(sorted(failure_categories.items())),
        "retry_rate": (retry_count / len(samples)) if samples else 0.0,
        "repair_applied_rate": (repair_applied_count / len(samples)) if samples else 0.0,
        "constrained_decoding_used_rate": (constrained_decoding_used_count / len(samples)) if samples else 0.0,
        "semantic_valid": semantic_statuses.get("VALID", 0),
        "semantic_low_quality": semantic_statuses.get("LOW_QUALITY", 0),
        "semantic_drift": semantic_statuses.get("SEMANTIC_DRIFT", 0),
        "language_drift": semantic_statuses.get("LANGUAGE_DRIFT", 0),
    }


def build_operational_judgment(
    summary: Mapping[str, Any],
    sample_summary: Mapping[str, Any],
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    raw_parseable_json = int(sample_summary.get("raw_parseable_json", 0) or 0)
    fence_stripped_parseable_json = int(sample_summary.get("fence_stripped_parseable_json", 0) or 0)
    recoverable_fenced_json = int(sample_summary.get("recoverable_fenced_json", 0) or 0)
    malformed_json = int(sample_summary.get("malformed_json", 0) or 0)
    enum_drift_total = int(sample_summary.get("enum_drift_total", 0) or 0)
    used_true_qlora = bool(summary.get("used_true_qlora"))

    if raw_parseable_json == 0 and recoverable_fenced_json > 0 and malformed_json == 0 and enum_drift_total == 0:
        operational_issue = "markdown_fencing_only"
        recommended_next_action = "Fix generation formatting or decoding to suppress markdown fences; do not redesign dataset or trainer yet."
    elif malformed_json > 0:
        operational_issue = "malformed_json_present"
        recommended_next_action = "Investigate generation formatting and decoding deeper; malformed JSON remains after fence stripping."
    elif enum_drift_total > 0:
        operational_issue = "enum_drift_present"
        recommended_next_action = "Investigate post-train structural quality and enum compliance before broader smoke conclusions."
    elif raw_parseable_json > 0:
        operational_issue = "structurally_usable"
        recommended_next_action = "Generation structure looks usable; move to the next smoke-run check or a slightly longer run."
    else:
        operational_issue = "raw_json_parse_failed"
        recommended_next_action = "Inspect generation formatting first; samples were not parseable and no fenced recovery path was detected."

    return {
        "true_qlora_passed": used_true_qlora,
        "raw_json_parse_failed": raw_parseable_json == 0,
        "raw_parseable_json": raw_parseable_json,
        "fence_stripped_parseable_json": fence_stripped_parseable_json,
        "recoverable_fenced_json": recoverable_fenced_json,
        "malformed_json": malformed_json,
        "enum_drift_total": enum_drift_total,
        "operational_issue": operational_issue,
        "recommended_next_action": recommended_next_action,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }


def build_baseline_candidate_judgment(
    result: SmokeRunResult | Mapping[str, Any],
    analysis_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result_payload = result.to_dict() if isinstance(result, SmokeRunResult) else dict(result)
    analysis_payload = dict(analysis_report or {})

    malformed_json = int(analysis_payload.get("malformed_json_count", 0) or 0)
    fenced_json = int(analysis_payload.get("fenced_json_count", 0) or 0)
    truncation = int(analysis_payload.get("truncation_count", 0) or 0)
    trailing_text = int(analysis_payload.get("trailing_text_count", 0) or 0)
    enum_drift = int(analysis_payload.get("enum_drift_count", 0) or 0)
    semantic_low_quality = int(analysis_payload.get("semantic_low_quality_count", 0) or 0)
    semantic_drift = int(analysis_payload.get("semantic_drift_count", 0) or 0)
    language_drift = int(analysis_payload.get("language_drift_count", 0) or 0)

    used_true_qlora = bool(result_payload.get("used_true_qlora"))
    training_completed_successfully = str(result_payload.get("status", "")) == "ok"
    finite_losses = bool(result_payload.get("finite_losses"))
    adapter_dir = result_payload.get("adapter_dir")
    adapter_exists = False
    if adapter_dir:
        adapter_path = Path(str(adapter_dir))
        adapter_exists = adapter_path.exists() and any(adapter_path.iterdir()) if adapter_path.exists() else False
    structure_stable = all(count == 0 for count in (malformed_json, fenced_json, truncation, trailing_text, enum_drift))
    analyzer_overall_status = str(analysis_payload.get("overall_status", "")) or None
    semantic_quality_primary_issue = structure_stable and (
        analyzer_overall_status == "semantic_quality_issue" or any(count > 0 for count in (semantic_low_quality, semantic_drift, language_drift))
    )
    is_baseline_candidate = used_true_qlora and training_completed_successfully and finite_losses and adapter_exists and structure_stable

    if not training_completed_successfully and str(result_payload.get("status", "")) == "blocked":
        verdict = "FAIL_BLOCKED_RUNTIME"
        recommended_next_action = "Do not register this run as a baseline candidate; inspect the blocker and rerun only after the environment and runtime are healthy."
    elif not training_completed_successfully:
        verdict = "FAIL_TRAINING_INCOMPLETE"
        recommended_next_action = "Do not register this run as a baseline candidate; inspect the blocker and rerun only after the environment and runtime are healthy."
    elif not used_true_qlora:
        verdict = "FAIL_BLOCKED_RUNTIME"
        recommended_next_action = "Reject this run for baseline comparison; true QLoRA was not active."
    elif not adapter_exists or not finite_losses:
        verdict = "FAIL_ARTIFACT_INVALID"
        recommended_next_action = "Treat this run as invalid for baseline comparison until adapter artifacts exist and losses are finite."
    elif not structure_stable:
        verdict = "FAIL_ARTIFACT_INVALID"
        recommended_next_action = "Treat this run as structurally unstable; inspect malformed/truncation/enum failures before considering it as a baseline candidate."
    elif semantic_quality_primary_issue:
        verdict = "PASS_STRUCTURAL_BUT_SEMANTIC_WEAK"
        recommended_next_action = "Treat this run as a structurally healthy baseline candidate, but keep semantic quality review on the shortlist before selecting it as best adapter."
    else:
        verdict = "PASS_BASELINE_CANDIDATE"
        recommended_next_action = "This run is a valid baseline candidate for comparison and possible promotion to best adapter."

    return {
        "used_true_qlora": used_true_qlora,
        "training_completed_successfully": training_completed_successfully,
        "adapter_exists": adapter_exists,
        "losses_finite": finite_losses,
        "structure_stable": structure_stable,
        "semantic_quality_is_primary_remaining_issue": semantic_quality_primary_issue,
        "analyzer_overall_status": analyzer_overall_status,
        "is_baseline_candidate": is_baseline_candidate,
        "verdict": verdict,
        "train_loss": result_payload.get("train_loss"),
        "eval_loss": result_payload.get("eval_loss"),
        "output_dir": result_payload.get("output_dir"),
        "adapter_dir": adapter_dir,
        "recommended_next_action": recommended_next_action,
        "blocker_reason": result_payload.get("blocker_reason"),
    }


def recommended_next_smoke_config() -> dict[str, int]:
    return {
        "max_steps": 50,
        "max_train_samples": 512,
        "max_eval_samples": 128,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_blocked_result(
    *,
    output_dir: Path,
    summary_path: Path,
    config_snapshot_path: Path | None,
    metrics_path: Path | None,
    sample_path: Path | None,
    adapter_dir: Path | None,
    environment: dict[str, Any],
    runtime: RuntimeConfig | None,
    train_rows: list[dict],
    eval_rows: list[dict],
    blocker_reason: str,
) -> SmokeRunResult:
    return SmokeRunResult(
        success=False,
        status="blocked",
        used_true_qlora=bool(runtime and runtime.use_qlora),
        runtime=asdict(runtime) if runtime else None,
        environment=environment,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        config_snapshot=str(config_snapshot_path) if config_snapshot_path else None,
        metrics_path=str(metrics_path) if metrics_path else None,
        sample_path=str(sample_path) if sample_path else None,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        train_task_counts=_count_tasks(train_rows),
        eval_task_counts=_count_tasks(eval_rows),
        train_loss=None,
        eval_loss=None,
        finite_losses=None,
        blocker_reason=blocker_reason,
    )


def run_smoke(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    config = coerce_smoke_config(config_input, default_run_mode="smoke")
    output_dir = _resolve_output_dir(config.output_dir, config.run_mode)
    config_snapshot_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "sample_generations.jsonl"
    metrics_path = output_dir / "metrics.json"
    adapter_dir = output_dir / "adapter"
    environment = get_environment_summary()
    runtime: RuntimeConfig | None = None
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    try:
        train_rows = pick_rows(load_message_rows(config.train_file), config.max_train_samples, config.seed)
        eval_rows = pick_rows(load_message_rows(config.dev_file), config.max_eval_samples, config.seed + 1)
        runtime = detect_runtime(prefer_qlora=not config.disable_qlora, require_qlora=config.require_qlora)

        _write_json(
            config_snapshot_path,
            {
                **config.to_dict(),
                "generated_at": datetime.now(UTC).isoformat(),
                "runtime": asdict(runtime),
                "environment": environment,
                "max_train_samples": len(train_rows),
                "max_eval_samples": len(eval_rows),
            },
        )

        if config.dry_run:
            result = SmokeRunResult(
                success=True,
                status="dry_run",
                used_true_qlora=runtime.use_qlora,
                runtime=asdict(runtime),
                environment=environment,
                output_dir=str(output_dir),
                summary_path=str(summary_path),
                config_snapshot=str(config_snapshot_path),
                metrics_path=None,
                sample_path=None,
                adapter_dir=None,
                train_rows=len(train_rows),
                eval_rows=len(eval_rows),
                train_task_counts=_count_tasks(train_rows),
                eval_task_counts=_count_tasks(eval_rows),
                train_loss=None,
                eval_loss=None,
                finite_losses=None,
            )
            _write_json(summary_path, result.to_dict())
            return result

        libs = _load_training_libraries()
        torch = libs["torch"]
        set_seed = libs["set_seed"]
        Dataset = libs["Dataset"]
        DataCollatorForLanguageModeling = libs["DataCollatorForLanguageModeling"]
        TrainingArguments = libs["TrainingArguments"]
        Trainer = libs["Trainer"]

        set_seed(config.seed)
        model, tokenizer = _load_model_and_tokenizer(config, runtime, libs)
        train_dataset = _tokenize_rows(train_rows, tokenizer, config.max_length, Dataset)
        eval_dataset = _tokenize_rows(eval_rows, tokenizer, config.max_length, Dataset)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args_kwargs = build_training_arguments_kwargs(
            runtime,
            available_parameters=set(inspect.signature(TrainingArguments.__init__).parameters),
            output_dir=str(output_dir / "checkpoints"),
            max_steps=config.max_steps,
            train_batch_size=config.per_device_train_batch_size,
            eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            seed=config.seed,
            logging_steps=config.logging_steps,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            save_total_limit=config.save_total_limit,
        )
        training_args = TrainingArguments(**training_args_kwargs)

        trainer = Trainer(
            **build_trainer_kwargs(
                available_parameters=set(inspect.signature(Trainer.__init__).parameters),
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
                tokenizer=tokenizer,
            )
        )

        train_result = trainer.train()
        eval_metrics = trainer.evaluate()

        ensure_directory(adapter_dir)
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        sample_rows = _select_generation_rows(train_rows, eval_rows)
        samples, structured_metrics = _generate_samples(model, tokenizer, sample_rows, sample_path, runtime, torch)
        sample_summary = summarize_sample_generations(samples)

        train_loss = float(train_result.training_loss)
        eval_loss = float(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None
        finite_losses = math.isfinite(train_loss) and (eval_loss is None or math.isfinite(eval_loss))

        _write_json(
            metrics_path,
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "finite_losses": finite_losses,
                "retry_rate": sample_summary["retry_rate"],
                "repair_applied_rate": sample_summary["repair_applied_rate"],
                "constrained_decoding_used_rate": sample_summary["constrained_decoding_used_rate"],
                "structured_metrics": structured_metrics,
            },
        )

        result = SmokeRunResult(
            success=True,
            status="ok",
            used_true_qlora=runtime.use_qlora,
            runtime=asdict(runtime),
            environment=environment,
            output_dir=str(output_dir),
            summary_path=str(summary_path),
            config_snapshot=str(config_snapshot_path),
            metrics_path=str(metrics_path),
            sample_path=str(sample_path),
            adapter_dir=str(adapter_dir),
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            train_task_counts=_count_tasks(train_rows),
            eval_task_counts=_count_tasks(eval_rows),
            train_loss=train_loss,
            eval_loss=eval_loss,
            finite_losses=finite_losses,
        )
        _write_json(summary_path, result.to_dict())
        return result
    except Exception as exc:  # noqa: BLE001
        blocker_reason = f"{type(exc).__name__}: {exc}"
        if not config_snapshot_path.exists():
            _write_json(
                config_snapshot_path,
                {
                    **config.to_dict(),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "runtime": asdict(runtime) if runtime else None,
                    "environment": environment,
                    "max_train_samples": len(train_rows),
                    "max_eval_samples": len(eval_rows),
                },
            )
        result = _build_blocked_result(
            output_dir=output_dir,
            summary_path=summary_path,
            config_snapshot_path=config_snapshot_path,
            metrics_path=metrics_path if metrics_path.exists() else None,
            sample_path=sample_path if sample_path.exists() else None,
            adapter_dir=adapter_dir if adapter_dir.exists() else None,
            environment=environment,
            runtime=runtime,
            train_rows=train_rows,
            eval_rows=eval_rows,
            blocker_reason=blocker_reason,
        )
        _write_json(summary_path, result.to_dict())
        return result


def run_smoke_or_raise(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    result = run_smoke(config_input)
    if not result.success:
        raise SmokeRunBlockedError(result.blocker_reason or "Smoke run failed")
    return result


def run_baseline(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    result = run_smoke(coerce_smoke_config(config_input, default_run_mode="baseline"))
    return result


def run_baseline_or_raise(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    result = run_baseline(config_input)
    if not result.success:
        raise SmokeRunBlockedError(result.blocker_reason or "Baseline run failed")
    return result


def _build_arg_parser(*, run_mode: str) -> argparse.ArgumentParser:
    defaults = _run_mode_defaults(run_mode)
    if run_mode == "baseline":
        description = "Run a reusable WorldSim QLoRA baseline training job."
        model_help = "Base model to adapt. Defaults to the WorldSim baseline target Qwen3.5 0.8B base model."
        output_help = "Optional explicit output directory. Defaults to outputs/baseline/worldsim-v31-mix-v1/<timestamp>."
    else:
        description = "Run a minimal WorldSim QLoRA smoke training job."
        model_help = "Base model to adapt. Defaults to a small public Qwen instruct model for smoke validation."
        output_help = "Optional explicit output directory. Defaults to outputs/smoke/worldsim-v31-mix-v1/<timestamp>."

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run-mode", default=run_mode, choices=sorted(RUN_MODE_DEFAULTS), help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=defaults["model_name"], help=model_help)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--dev-file", type=Path, default=DEFAULT_DEV_FILE)
    parser.add_argument("--output-dir", type=Path, default=None, help=output_help)
    parser.add_argument("--max-steps", type=int, default=defaults["max_steps"])
    parser.add_argument("--max-train-samples", type=int, default=defaults["max_train_samples"], help="0 means use the full training split.")
    parser.add_argument("--max-eval-samples", type=int, default=defaults["max_eval_samples"], help="0 means use the full eval split.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=defaults["per_device_train_batch_size"])
    parser.add_argument("--per-device-eval-batch-size", type=int, default=defaults["per_device_eval_batch_size"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults["gradient_accumulation_steps"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--logging-steps", type=int, default=defaults["logging_steps"])
    parser.add_argument("--eval-steps", type=int, default=defaults["eval_steps"], help="0 disables periodic evaluation.")
    parser.add_argument("--save-steps", type=int, default=defaults["save_steps"], help="0 disables periodic checkpoint saves.")
    parser.add_argument("--save-total-limit", type=int, default=defaults["save_total_limit"])
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-qlora", action="store_true", help="Force plain LoRA even if CUDA + bitsandbytes are available.")
    parser.add_argument("--require-qlora", action="store_true", help="Fail instead of falling back when true 4-bit QLoRA is unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and runtime selection without loading model weights.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser(run_mode="smoke")
    return parser.parse_args(argv)


def parse_baseline_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser(run_mode="baseline")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_smoke(parse_args(argv))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1


def main_baseline(argv: Sequence[str] | None = None) -> int:
    result = run_baseline(parse_baseline_args(argv))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1
