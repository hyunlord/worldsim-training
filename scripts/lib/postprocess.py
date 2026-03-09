from __future__ import annotations

import json
import hashlib
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.common import AttrDict, ensure_directory, load_yaml, read_jsonl, write_jsonl
from scripts.lib.normalize import CANONICAL_EMOTIONS, CANONICAL_REGISTERS, normalize_emotion, normalize_register, normalize_text, strip_outer_quotes
from scripts.validate_data import AUTO_REPLACEMENTS, auto_repair

NORMALIZATION_VERSION = "worldsim-postprocess-normalize-v2"
VALIDATOR_VERSION = "worldsim-postprocess-validator-v2"
CANONICAL_TASKS = ("A", "B", "C", "E", "F", "G", "H")
CANONICAL_TCI_TRAITS = ("novelty_seeking", "harm_avoidance", "reward_dependence", "persistence")
LEGACY_HEXACO_TRAITS = (
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
)
CANONICAL_TEMPERAMENTS = ("choleric", "melancholic", "sanguine", "phlegmatic", "mixed")
DEFAULT_SPEAKER_ROLES = ("elder", "hunter", "shaman", "warrior", "healer", "gatherer", "craftsman", "chief", "scout", "observer")
DEFAULT_REQUIRED_FIELDS = {
    "A": ("text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"),
    "B": ("text_ko", "text_en", "register", "emotion_expressed", "intensity", "mimetics", "temperament_influence"),
    "C": ("speech_ko", "speech_en", "register", "emotion_expressed", "speaker_role", "temperament_tone"),
    "E": ("action_id", "confidence", "hint_ko", "hint_en", "personality_reasoning", "temperament_factor"),
    "F": ("emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type", "temperament_amplifier"),
    "G": ("interpretation_ko", "interpretation_en", "action_tendency", "confidence", "register", "misinterpretation_type", "temperament_bias"),
    "H": ("name", "description_en", "resource_modifiers", "special_zones", "special_resources", "agent_modifiers"),
}
DEFAULT_TASK_LIMITS = {
    "A": {"min_chars": 20, "max_chars": 40, "sentences": 1},
    "B": {"min_chars": 30, "max_chars": 60, "sentences": 2},
    "C": {"min_chars": 15, "max_chars": 30, "sentences": 1},
    "E": {"min_chars": 10, "max_chars": 30, "sentences": 1},
    "F": {"min_chars": 10, "max_chars": 25, "sentences": 1},
    "G": {"min_chars": 15, "max_chars": 40, "sentences": 1},
    "H": {},
}
DEFAULT_REASONING_AXES = (
    "high_honesty_humility",
    "high_emotionality",
    "high_extraversion",
    "high_agreeableness",
    "high_conscientiousness",
    "high_openness",
    "high_novelty_seeking",
    "high_harm_avoidance",
    "high_reward_dependence",
    "high_persistence",
)
DEFAULT_TRANSITION_TYPES = ("gradual", "sudden", "sustained")
DEFAULT_ORACLE_ACTION_TENDENCIES = ("mobilize", "defend", "wait", "retreat", "celebrate", "mourn")
DEFAULT_ORACLE_MISINTERPRETATIONS = (
    "overconfident_literal",
    "cautious_reversal",
    "optimistic_expansion",
    "passive_deferral",
    "symbolic_abstraction",
)
DEFAULT_REGISTER_PATTERNS = {
    "haera": [r"다[.\s!?]?$", r"는다[.\s!?]?$", r"했다[.\s!?]?$", r"쳤다[.\s!?]?$", r"라[.\s!?]?$"],
    "hao": [r"오[.\s!?]?$", r"소[.\s!?]?$", r"시오[.\s!?]?$"],
    "hae": [r"해[.\s!?]?$", r"야[.\s!?]?$", r"지[.\s!?]?$", r"어[.\s!?]?$"],
}
NEGATIVE_SITUATIONS = {"predator", "storm", "injury", "kin_death", "theft"}
EMOTION_CUES = {
    "joy": ("웃", "방긋", "싱글벙글", "기쁘", "smile", "grin", "cheer"),
    "sadness": ("울", "슬프", "눈물", "mourn", "weep", "sorrow"),
    "fear": ("겁", "두려", "떨", "오들", "벌벌", "숨", "물러", "trembl", "fear", "terrif", "cower"),
    "anger": ("노려", "으드득", "이를", "악물", "분노", "glare", "anger", "rage", "clench"),
    "trust": ("믿", "기대", "의지", "trust", "rely"),
    "disgust": ("역겨", "혐오", "찌푸", "disgust", "recoil"),
    "surprise": ("깜짝", "놀라", "surpris", "startl"),
    "anticipation": ("때를", "재", "기회", "덤빌", "준비", "anticip", "prepare", "size up", "watch for"),
}
CONTRADICTION_CUES = {
    "joy": ("겁", "두려", "떨", "terrif", "fear", "trembl", "겁에 질"),
    "trust": ("겁", "두려", "노려", "trembl", "terror"),
}
LOW_INFORMATION_MARKERS = {"...", "lorem ipsum", "todo", "tbd"}
PRIMARY_TEXT_FIELDS = {
    "A": "text_ko",
    "B": "text_ko",
    "C": "speech_ko",
    "E": "hint_ko",
    "F": "cause_ko",
    "G": "interpretation_ko",
}
TIC_TRAIT_ALIASES = {
    "ns": "novelty_seeking",
    "novelty_seeking": "novelty_seeking",
    "novelty seeking": "novelty_seeking",
    "ha": "harm_avoidance",
    "harm avoidance": "harm_avoidance",
    "harm_avoidance": "harm_avoidance",
    "rd": "reward_dependence",
    "reward dependence": "reward_dependence",
    "reward_dependence": "reward_dependence",
    "p": "persistence",
    "persistence": "persistence",
}


@dataclass(slots=True)
class PostprocessPolicy:
    required_fields: dict[str, tuple[str, ...]]
    task_limits: dict[str, dict[str, int]]
    register_patterns: dict[str, tuple[str, ...]]
    forbidden_words: tuple[str, ...]
    canonical_tci_traits: tuple[str, ...]
    legacy_hexaco_traits: tuple[str, ...]
    temperaments: tuple[str, ...]
    speaker_roles: tuple[str, ...]
    canonical_emotions: tuple[str, ...]
    canonical_registers: tuple[str, ...]
    reasoning_axes: tuple[str, ...]
    transition_types: tuple[str, ...]
    oracle_action_tendencies: tuple[str, ...]
    oracle_misinterpretations: tuple[str, ...]
    auto_replacements: dict[str, str]


@dataclass(slots=True)
class PostprocessResult:
    disposition: str
    normalized_output: dict[str, Any] | None
    structural_issues: list[str] = field(default_factory=list)
    semantic_issues: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_postprocess_policy(config_dir: Path) -> PostprocessPolicy:
    settings = load_yaml(config_dir / "generation.yaml") if (config_dir / "generation.yaml").exists() else {}
    tasks = settings.get("tasks", {})
    validation = settings.get("validation", {})
    required_fields = {
        task: tuple(tasks.get(task, {}).get("output_fields", {}).get("required", DEFAULT_REQUIRED_FIELDS[task]))
        for task in CANONICAL_TASKS
    }
    task_limits = {
        task: {**DEFAULT_TASK_LIMITS.get(task, {}), **validation.get("task_limits", {}).get(task, {})}
        for task in CANONICAL_TASKS
    }
    configured_patterns = validation.get("register_endings", {})
    register_patterns = {}
    for register in CANONICAL_REGISTERS:
        merged_patterns = list(DEFAULT_REGISTER_PATTERNS[register])
        for pattern in configured_patterns.get(register, ()):
            if pattern not in merged_patterns:
                merged_patterns.append(pattern)
        register_patterns[register] = tuple(merged_patterns)
    speaker_roles = tuple(validation.get("speaker_roles", DEFAULT_SPEAKER_ROLES))
    temperaments = tuple(item.get("id") for item in settings.get("temperaments", []) if item.get("id")) or CANONICAL_TEMPERAMENTS
    return PostprocessPolicy(
        required_fields=required_fields,
        task_limits=task_limits,
        register_patterns=register_patterns,
        forbidden_words=tuple(validation.get("forbidden_words", ())),
        canonical_tci_traits=CANONICAL_TCI_TRAITS,
        legacy_hexaco_traits=LEGACY_HEXACO_TRAITS,
        temperaments=tuple(temperaments),
        speaker_roles=tuple(speaker_roles),
        canonical_emotions=CANONICAL_EMOTIONS,
        canonical_registers=CANONICAL_REGISTERS,
        reasoning_axes=tuple(validation.get("reasoning_axes", DEFAULT_REASONING_AXES)),
        transition_types=tuple(validation.get("transition_types", DEFAULT_TRANSITION_TYPES)),
        oracle_action_tendencies=tuple(validation.get("oracle_action_tendencies", DEFAULT_ORACLE_ACTION_TENDENCIES)),
        oracle_misinterpretations=tuple(validation.get("oracle_misinterpretations", DEFAULT_ORACLE_MISINTERPRETATIONS)),
        auto_replacements=dict(AUTO_REPLACEMENTS),
    )


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _safe_git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    }


def _count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text.strip()) if part.strip()])


def _matches_register(text: str, register: str, patterns: dict[str, tuple[str, ...]]) -> bool:
    endings = patterns.get(register, ())
    if not endings:
        return True
    return any(re.search(pattern, text.strip()) for pattern in endings)


def _parse_output(output: object) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(output, dict):
        return dict(output), []
    if not isinstance(output, str):
        return None, ["not_json"]
    raw = output.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None, ["not_json"]
        else:
            return None, ["not_json"]
    if not isinstance(payload, dict):
        return None, ["json_root_not_object"]
    return dict(payload), []


def _normalize_trait(value: object) -> tuple[object, list[str], list[str]]:
    if not isinstance(value, str):
        return value, [], []
    normalized = normalize_text(value).strip(".,!?").lower().replace("-", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    if normalized in TIC_TRAIT_ALIASES:
        canonical = TIC_TRAIT_ALIASES[normalized]
        actions = ["normalized_dominant_trait"] if canonical != normalized else []
        return canonical, actions, []
    if normalized in LEGACY_HEXACO_TRAITS:
        return normalized, [], ["legacy_trait_schema_mismatch"]
    return value, [], ["invalid_dominant_trait"]


def _normalize_common_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    return normalize_text(value)


def _normalize_output(task: str, payload: dict[str, Any], policy: PostprocessPolicy) -> tuple[dict[str, Any], list[str], list[str]]:
    normalized = {key: _normalize_common_text(value) for key, value in payload.items()}
    actions: list[str] = []
    notes: list[str] = []

    for field in ("emotion_expressed", "emotion", "previous_emotion"):
        if field in normalized and isinstance(normalized[field], str):
            canonical = normalize_emotion(normalized[field])
            if canonical is not None:
                if canonical != normalized[field]:
                    actions.append(f"normalized_{field}")
                normalized[field] = canonical

    if "register" in normalized and isinstance(normalized["register"], str):
        canonical_register = normalize_register(normalized["register"])
        if canonical_register is not None:
            if canonical_register != normalized["register"]:
                actions.append("normalized_register")
            normalized["register"] = canonical_register

    if task == "A" and "dominant_trait" in normalized:
        normalized["dominant_trait"], trait_actions, trait_notes = _normalize_trait(normalized["dominant_trait"])
        actions.extend(trait_actions)
        notes.extend(trait_notes)

    if task == "C":
        for field in ("speech_ko", "speech_en"):
            if field in normalized and isinstance(normalized[field], str):
                stripped, changed = strip_outer_quotes(normalized[field])
                if changed:
                    normalized[field] = stripped
                    actions.append(f"unwrapped_{field}")

    if "mimetics" in normalized and isinstance(normalized["mimetics"], list):
        seen: list[str] = []
        for item in normalized["mimetics"]:
            if not isinstance(item, str):
                continue
            cleaned = normalize_text(item)
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        if seen != normalized["mimetics"]:
            actions.append("normalized_mimetics")
        normalized["mimetics"] = seen

    for field in ("text_ko", "speech_ko", "hint_ko", "cause_ko", "interpretation_ko"):
        if field in normalized and isinstance(normalized[field], str):
            repaired, repair_count = auto_repair(normalized[field], policy.auto_replacements)
            normalized[field] = repaired
            if repair_count:
                actions.append(f"auto_repaired_{field}")

    return normalized, sorted(set(actions)), sorted(set(notes))


def _check_required_fields(task: str, payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    for field in policy.required_fields[task]:
        if field not in payload:
            issues.append(f"missing_{field}")
            continue
        value = payload[field]
        if value is None:
            issues.append(f"empty_{field}")
            continue
        if isinstance(value, str) and not value.strip():
            issues.append(f"empty_{field}")
    return issues


def _check_task_a_structural(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    dominant_trait = payload.get("dominant_trait")
    if dominant_trait not in policy.canonical_tci_traits and dominant_trait not in policy.legacy_hexaco_traits:
        issues.append("invalid_dominant_trait")
    temperament = payload.get("temperament_expressed")
    if temperament not in policy.temperaments:
        issues.append("invalid_temperament_expressed")
    register = payload.get("register")
    if register not in policy.canonical_registers:
        issues.append("invalid_register")
    return issues


def _check_task_b_structural(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    if payload.get("register") not in policy.canonical_registers:
        issues.append("invalid_register")
    if payload.get("emotion_expressed") not in policy.canonical_emotions:
        issues.append("invalid_emotion")
    intensity = payload.get("intensity")
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= float(intensity) <= 1:
        issues.append("invalid_intensity")
    mimetics = payload.get("mimetics")
    if not isinstance(mimetics, list):
        issues.append("invalid_mimetics")
    else:
        for item in mimetics:
            if not isinstance(item, str) or not item.strip():
                issues.append("invalid_mimetic_item")
                break
    return issues


def _check_task_c_structural(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    if payload.get("register") not in policy.canonical_registers:
        issues.append("invalid_register")
    if payload.get("emotion_expressed") not in policy.canonical_emotions:
        issues.append("invalid_emotion")
    if payload.get("speaker_role") not in policy.speaker_roles:
        issues.append("invalid_speaker_role")
    return issues


def _check_task_e_structural(record: dict[str, Any], payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    action_id = payload.get("action_id")
    if isinstance(action_id, bool) or not isinstance(action_id, int):
        issues.append("invalid_action_id")
    else:
        action_options = record.get("action_options")
        if isinstance(action_options, list) and action_options:
            if action_id < 0 or action_id >= len(action_options):
                issues.append("invalid_action_id")
        elif action_id < 0 or action_id > 9:
            issues.append("invalid_action_id")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        issues.append("invalid_confidence")
    reasoning = payload.get("personality_reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        issues.append("invalid_personality_reasoning")
    elif reasoning not in policy.reasoning_axes and record.get("personality_reasoning") and reasoning != record.get("personality_reasoning"):
        issues.append("reasoning_context_mismatch")
    return issues


def _check_task_f_structural(record: dict[str, Any], payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    if payload.get("emotion") not in policy.canonical_emotions:
        issues.append("invalid_emotion")
    intensity = payload.get("intensity")
    if isinstance(intensity, bool) or not isinstance(intensity, (int, float)) or not 0 <= float(intensity) <= 1:
        issues.append("invalid_intensity")
    previous_emotion = payload.get("previous_emotion")
    if previous_emotion not in policy.canonical_emotions:
        issues.append("invalid_previous_emotion")
    elif record.get("current_emotion_id") and previous_emotion != record.get("current_emotion_id"):
        issues.append("previous_emotion_mismatch")
    if payload.get("transition_type") not in policy.transition_types:
        issues.append("invalid_transition_type")
    return issues


def _check_task_g_structural(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    if payload.get("register") not in policy.canonical_registers:
        issues.append("invalid_register")
    if payload.get("action_tendency") not in policy.oracle_action_tendencies:
        issues.append("invalid_action_tendency")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        issues.append("invalid_confidence")
    if payload.get("misinterpretation_type") not in policy.oracle_misinterpretations:
        issues.append("invalid_misinterpretation_type")
    return issues


def _check_task_h_structural(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    name = payload.get("name")
    if not isinstance(name, str) or not re.match(r"^[A-Z][a-zA-Z]+$", name):
        issues.append("invalid_name_format")
    description = payload.get("description_en")
    if not isinstance(description, str) or len(description.strip()) < 10:
        issues.append("short_description")
    for field in ("resource_modifiers", "special_zones", "special_resources", "agent_modifiers"):
        if not isinstance(payload.get(field), list):
            issues.append(f"missing_or_invalid_{field}")
    if isinstance(payload.get("resource_modifiers"), list):
        for modifier in payload["resource_modifiers"]:
            if not isinstance(modifier, dict):
                issues.append("invalid_resource_modifier")
                continue
            multiplier = modifier.get("multiplier", -1)
            if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not 0 <= float(multiplier) <= 5:
                issues.append("invalid_multiplier_range")
                break
    return issues


def _check_text_limits(task: str, payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    field = PRIMARY_TEXT_FIELDS[task]
    value = payload.get(field)
    if not isinstance(value, str):
        return [f"invalid_{field}_type"]
    limits = policy.task_limits[task]
    issues: list[str] = []
    length = len(value.strip())
    if length < int(limits.get("min_chars", 1)):
        issues.append("too_short")
    if length > int(limits.get("max_chars", 999)):
        issues.append("too_long")
    expected_sentences = limits.get("sentences")
    if expected_sentences is not None and _count_sentences(value) != int(expected_sentences):
        issues.append("sentence_count_mismatch")
    return issues


def _contains_forbidden(value: str, policy: PostprocessPolicy) -> bool:
    return any(word in value for word in policy.forbidden_words)


def _semantic_common(task: str, payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    field = PRIMARY_TEXT_FIELDS[task]
    text = str(payload.get(field, "")).strip()
    lowered = text.lower()
    issues: list[str] = []
    if not text:
        issues.append("empty_text")
    if any(marker in lowered for marker in LOW_INFORMATION_MARKERS):
        issues.append("low_information_output")
    if _contains_forbidden(text, policy):
        issues.append("forbidden_word_remaining")
    return issues


def _semantic_task_a(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("A", payload, policy)
    dominant_trait = payload.get("dominant_trait")
    if dominant_trait in policy.legacy_hexaco_traits:
        issues.append("legacy_trait_schema_mismatch")
    text = str(payload.get("text_ko", ""))
    if payload.get("register") == "haera" and not _matches_register(text, "haera", policy.register_patterns):
        issues.append("register_style_mismatch")
    return issues


def _collect_emotion_cues(text: str) -> dict[str, int]:
    lowered = text.lower()
    scores: dict[str, int] = {}
    for emotion, cues in EMOTION_CUES.items():
        scores[emotion] = sum(1 for cue in cues if cue in lowered)
    return scores


def _semantic_task_b(record: dict[str, Any], payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("B", payload, policy)
    text_ko = str(payload.get("text_ko", ""))
    text_en = str(payload.get("text_en", ""))
    combined = f"{text_ko} {text_en}".lower()
    emotion = payload.get("emotion_expressed")
    situation_id = str(record.get("situation_id", ""))
    scores = _collect_emotion_cues(combined)

    if emotion in CONTRADICTION_CUES:
        contradictions = sum(1 for cue in CONTRADICTION_CUES[emotion] if cue in combined)
        if contradictions >= 2:
            issues.append("emotion_text_contradiction")

    if situation_id in NEGATIVE_SITUATIONS and emotion in {"joy", "trust"} and scores.get("fear", 0) > 0:
        issues.append("emotion_text_contradiction")

    if emotion == "anticipation" and (scores.get("anticipation", 0) > 0 or "덤빌" in text_ko):
        pass
    elif scores.get(str(emotion), 0) == 0 and str(record.get("emotion_id")) != str(emotion):
        issues.append("emotion_context_tension")

    mimetics = payload.get("mimetics", [])
    if isinstance(mimetics, list):
        missing = [item for item in mimetics if isinstance(item, str) and item not in text_ko]
        if missing:
            issues.append("missing_literal_mimetic")
    return issues


def _semantic_task_c(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("C", payload, policy)
    speech_ko = str(payload.get("speech_ko", ""))
    register = payload.get("register")
    if isinstance(register, str) and register in policy.canonical_registers and not _matches_register(speech_ko, register, policy.register_patterns):
        issues.append("register_style_mismatch")
    return issues


def _semantic_task_e(record: dict[str, Any], payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("E", payload, policy)
    hint_ko = str(payload.get("hint_ko", ""))
    if len(hint_ko.split()) <= 1 and len(hint_ko) < 8:
        issues.append("low_information_output")
    return issues


def _semantic_task_f(record: dict[str, Any], payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("F", payload, policy)
    cause_ko = str(payload.get("cause_ko", ""))
    cause_en = str(payload.get("cause_en", ""))
    combined = f"{cause_ko} {cause_en}".lower()
    emotion = payload.get("emotion")
    situation_id = str(record.get("situation_id", ""))
    scores = _collect_emotion_cues(combined)

    if emotion in CONTRADICTION_CUES:
        contradictions = sum(1 for cue in CONTRADICTION_CUES[emotion] if cue in combined)
        if contradictions >= 2:
            issues.append("emotion_text_contradiction")

    if situation_id in NEGATIVE_SITUATIONS and emotion in {"joy", "trust"} and scores.get("fear", 0) > 0:
        issues.append("emotion_text_contradiction")
    return issues


def _semantic_task_g(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues = _semantic_common("G", payload, policy)
    interpretation_ko = str(payload.get("interpretation_ko", ""))
    register = payload.get("register")
    if isinstance(register, str) and register in policy.canonical_registers and not _matches_register(interpretation_ko, register, policy.register_patterns):
        issues.append("register_style_mismatch")
    if len(interpretation_ko.split()) <= 1 and len(interpretation_ko) < 12:
        issues.append("low_information_output")
    return issues


def _semantic_task_h(payload: dict[str, Any], policy: PostprocessPolicy) -> list[str]:
    issues: list[str] = []
    description = str(payload.get("description_en", "")).strip().lower()
    if not description:
        issues.append("low_information_output")
    if any(marker in description for marker in LOW_INFORMATION_MARKERS):
        issues.append("low_information_output")
    arrays = [payload.get("resource_modifiers", []), payload.get("special_zones", []), payload.get("special_resources", []), payload.get("agent_modifiers", [])]
    if all(isinstance(value, list) and not value for value in arrays):
        issues.append("low_information_ir")
    return issues


def classify_record(record: dict[str, Any], policy: PostprocessPolicy) -> PostprocessResult:
    task = str(record.get("task", ""))
    if task not in CANONICAL_TASKS:
        return PostprocessResult(disposition="failed", normalized_output=None, structural_issues=["unsupported_task"])

    payload, parse_issues = _parse_output(record.get("output"))
    if payload is None:
        return PostprocessResult(disposition="failed", normalized_output=None, structural_issues=parse_issues)

    normalized, actions, notes = _normalize_output(task, payload, policy)
    structural_issues = _check_required_fields(task, normalized, policy)
    if task == "A":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_a_structural(normalized, policy))
        semantic_issues = _semantic_task_a(normalized, policy)
    elif task == "B":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_b_structural(normalized, policy))
        semantic_issues = _semantic_task_b(record, normalized, policy)
    elif task == "C":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_c_structural(normalized, policy))
        semantic_issues = _semantic_task_c(normalized, policy)
    elif task == "E":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_e_structural(record, normalized, policy))
        semantic_issues = _semantic_task_e(record, normalized, policy)
    elif task == "F":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_f_structural(record, normalized, policy))
        semantic_issues = _semantic_task_f(record, normalized, policy)
    elif task == "G":
        structural_issues.extend(_check_text_limits(task, normalized, policy))
        structural_issues.extend(_check_task_g_structural(normalized, policy))
        semantic_issues = _semantic_task_g(normalized, policy)
    else:
        structural_issues.extend(_check_task_h_structural(normalized, policy))
        semantic_issues = _semantic_task_h(normalized, policy)

    structural_issues = sorted(set(issue for issue in structural_issues if issue))
    semantic_issues = sorted(set(issue for issue in semantic_issues + notes if issue))

    if structural_issues:
        disposition = "failed"
    elif any(
        issue
        in {
            "emotion_text_contradiction",
            "low_information_output",
            "forbidden_word_remaining",
            "missing_literal_mimetic",
            "low_information_ir",
        }
        for issue in semantic_issues
    ):
        disposition = "failed"
    elif semantic_issues:
        disposition = "review"
    elif actions:
        disposition = "recoverable"
    else:
        disposition = "passed"

    return PostprocessResult(
        disposition=disposition,
        normalized_output=normalized,
        structural_issues=structural_issues,
        semantic_issues=semantic_issues,
        recovery_actions=sorted(set(actions)),
    )


def enrich_record(record: dict[str, Any], result: PostprocessResult) -> dict[str, Any]:
    enriched = dict(record)
    if result.normalized_output is not None:
        enriched["output"] = _compact_json(result.normalized_output)
    enriched["postprocess"] = {
        "disposition": result.disposition,
        "structural_issues": result.structural_issues,
        "semantic_issues": result.semantic_issues,
        "recovery_actions": result.recovery_actions,
        "normalization_version": NORMALIZATION_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "processed_at": datetime.now(UTC).isoformat(),
    }
    return enriched


def write_postprocess_categories(records: list[dict[str, Any]], output_dir: Path) -> AttrDict:
    ensure_directory(output_dir)
    buckets = {"passed": [], "recoverable": [], "review": [], "failed": []}
    counts_by_task: Counter[str] = Counter()
    counts_by_reason: Counter[str] = Counter()
    for row in records:
        disposition = row["postprocess"]["disposition"]
        buckets[disposition].append(row)
        counts_by_task[row.get("task", "?")] += 1
        for issue in row["postprocess"]["structural_issues"] + row["postprocess"]["semantic_issues"]:
            counts_by_reason[issue] += 1

    paths = {}
    for name, rows in buckets.items():
        path = output_dir / f"{name}.jsonl"
        write_jsonl(path, rows)
        paths[f"{name}_path"] = path

    report = {
        "counts_by_disposition": {name: len(rows) for name, rows in buckets.items()},
        "counts_by_task": dict(counts_by_task),
        "counts_by_reason": dict(counts_by_reason),
        "normalization_version": NORMALIZATION_VERSION,
        "validator_version": VALIDATOR_VERSION,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return AttrDict(report=report, report_path=report_path, **paths)


def validate_records(records: list[dict[str, Any]], policy: PostprocessPolicy, output_dir: Path | None = None) -> AttrDict:
    enriched = [enrich_record(record, classify_record(record, policy)) for record in records if str(record.get("task", "")) in CANONICAL_TASKS]
    if output_dir is None:
        return AttrDict(records=enriched)
    result = write_postprocess_categories(enriched, output_dir)
    result.records = enriched
    return result


def snapshot_metadata(repo_root: Path, *, source_files: dict[str, str], snapshot_files: dict[str, str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = repo_root / "config" / "generation.yaml"
    prompt_path = repo_root / "prompts" / "teacher" / "system.txt"
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": _safe_git_sha(repo_root),
        "postprocess_version": NORMALIZATION_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "source_files": source_files,
        "snapshot_files": snapshot_files,
        "config_version": _file_metadata(config_path),
        "prompt_version": _file_metadata(prompt_path),
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "repo_root": str(repo_root),
        },
    }
    if extra:
        payload.update(extra)
    return payload
