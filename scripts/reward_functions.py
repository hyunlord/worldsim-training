#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import yaml


FEATURE_DIMENSIONS = ("risk_avoid", "approach", "prosocial", "persist", "passive")
PLACEHOLDER_WORDS = {"str", "string", "sentence", "English", "enum", "number", "해라체", "one of"}
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "but", "or", "not", "no", "so", "if", "then", "than", "that", "this",
    "는", "은", "이", "가", "을", "를", "에", "의", "로", "와", "과",
    "도", "만", "까지", "부터", "에서", "으로", "하다", "이다",
}
REASONING_TO_FEATURE = {
    "novelty_seeking": "approach",
    "novelty": "approach",
    "curiosity": "approach",
    "exploration": "approach",
    "bold": "approach",
    "impulsive": "approach",
    "harm_avoidance": "risk_avoid",
    "caution": "risk_avoid",
    "safety": "risk_avoid",
    "fear": "risk_avoid",
    "anxiety": "risk_avoid",
    "avoidance": "risk_avoid",
    "reward_dependence": "prosocial",
    "social": "prosocial",
    "cooperation": "prosocial",
    "bond": "prosocial",
    "empathy": "prosocial",
    "attachment": "prosocial",
    "persistence": "persist",
    "determination": "persist",
    "endurance": "persist",
    "perseverance": "persist",
    "tenacity": "persist",
}

DEFAULT_WEIGHTS = {
    "personality": 0.25,
    "emotion": 0.15,
    "hint_quality": 0.25,
    "confidence_fit": 0.15,
    "reasoning_fit": 0.10,
    "diversity": 0.10,
}

TASK_WEIGHTS = {
    "E": {"personality": 0.25, "emotion": 0.05, "hint_quality": 0.25, "confidence_fit": 0.20, "reasoning_fit": 0.15, "diversity": 0.10},
    "F": {"personality": 0.10, "emotion": 0.35, "hint_quality": 0.25, "confidence_fit": 0.10, "reasoning_fit": 0.10, "diversity": 0.10},
    "B": {"personality": 0.15, "emotion": 0.10, "hint_quality": 0.40, "confidence_fit": 0.10, "reasoning_fit": 0.10, "diversity": 0.15},
    "C": {"personality": 0.15, "emotion": 0.10, "hint_quality": 0.40, "confidence_fit": 0.10, "reasoning_fit": 0.10, "diversity": 0.15},
    "O": {"personality": 0.20, "emotion": 0.05, "hint_quality": 0.30, "confidence_fit": 0.15, "reasoning_fit": 0.15, "diversity": 0.15},
}

REQUIRED_FIELDS = {
    "A": {"text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"},
    "B": {"text_ko", "text_en", "register", "emotion_expressed", "intensity", "mimetics", "temperament_influence"},
    "C": {"speech_ko", "speech_en", "register", "emotion_expressed", "speaker_role", "temperament_tone"},
    "D": {"text_ko", "text_en", "event_type"},
    "E": {"action_id", "confidence", "hint", "personality_reasoning", "temperament_factor"},
    "F": {"emotion", "intensity", "cause", "previous_emotion", "transition_type", "temperament_amplifier"},
    "G": {"interpretation_ko", "interpretation_en", "action_tendency", "confidence", "register", "misinterpretation_type", "temperament_bias"},
    "H": {"name", "description_en", "resource_modifiers", "special_zones", "special_resources", "agent_modifiers"},
    "I": {"priority_id", "reasoning", "need_addressed", "urgency"},
    "J": {"coping_id", "coping_type", "stress_delta", "hint", "side_effect"},
    "K": {"social_action_id", "trust_delta", "hint", "relationship_intent", "reciprocity_expectation"},
    "L": {"response_id", "trust_delta", "hint", "forgiveness_threshold", "social_memory"},
    "M": {"decision_id", "confidence", "dissent_risk", "reasoning", "resource_commitment", "timeline"},
    "N": {"accept", "counter_offer_give", "counter_offer_want", "hint", "negotiation_stance", "walk_away_threshold"},
    "O": {"public_claim", "private_truth", "deception_style", "lie_degree", "detection_risk", "confidence"},
    "P": {"retold_version", "distortion_type", "added_detail", "dropped_detail", "emotional_charge"},
    "Q": {"trauma_response", "behavioral_change", "trigger_situation", "intensity", "duration", "coping_mechanism"},
    "R": {"action", "counter_give", "counter_want", "reasoning", "emotional_state", "walk_away_threshold"},
    "S": {"action", "modified_practice", "reasoning", "social_pressure", "tradition_conflict"},
    "T": {"decision_id", "confidence", "dissent_risk", "minority_position", "minority_action", "spark_event", "reasoning", "timeline"},
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def parse_tci_from_prompt(prompt: str) -> dict[str, float]:
    match = re.search(r"NS=(\d+\.?\d*)\s+HA=(\d+\.?\d*)\s+RD=(\d+\.?\d*)\s+P=(\d+\.?\d*)(?:\s+type=([A-Za-z_]+))?", prompt)
    if not match:
        return {"NS": 0.5, "HA": 0.5, "RD": 0.5, "P": 0.5, "type": "unknown"}
    return {
        "NS": float(match.group(1)),
        "HA": float(match.group(2)),
        "RD": float(match.group(3)),
        "P": float(match.group(4)),
        "type": match.group(5) or "unknown",
    }


def parse_task_from_prompt(prompt: str) -> str:
    match = re.search(r"\[TASK\]\s*([A-T])\b", prompt)
    return match.group(1) if match else ""


def parse_options_from_prompt(prompt: str) -> list[tuple[int, str]]:
    block_match = re.search(r"\[OPTIONS\]\s*(.*?)(?:\n\[[A-Z가-힣_ ]+\]|\Z)", prompt, flags=re.DOTALL)
    if not block_match:
        return []
    block = block_match.group(1)
    options: list[tuple[int, str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for option_id, option_text in re.findall(r"(\d+)\s*:\s*([^\d\n][^\n]*?)(?=(?:\s+\d+\s*:)|$)", line):
            options.append((int(option_id), option_text.strip()))
    return options


def parse_emotion_from_prompt(prompt: str) -> tuple[str, float] | None:
    match = re.search(
        r"(?:\[EMOTION\]|\[CURRENT EMOTION\]|\[지금 느끼는 것\])\s*([a-zA-Z_가-힣]+)\s*:\s*(-?\d+\.?\d*)",
        prompt,
    )
    if not match:
        return None
    return match.group(1), float(match.group(2))


@lru_cache(maxsize=1)
def _load_situations(config_dir: str) -> list[dict]:
    path = Path(config_dir) / "situations.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload.get("situations", [])


def parse_situation_from_prompt(prompt: str) -> str | None:
    config_dir = Path.cwd() / "config"
    for situation in _load_situations(str(config_dir.resolve())):
        candidates = [str(situation.get("id", "")), str(situation.get("ko", "")), str(situation.get("desc", ""))]
        if any(candidate and candidate in prompt for candidate in candidates):
            return str(situation.get("id"))
    return None


@lru_cache(maxsize=16)
def _load_yaml_cached(path_str: str) -> dict:
    return yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))


def load_action_features(config_dir: Path) -> dict[str, dict[str, float]]:
    payload = _load_yaml_cached(str((config_dir / "action_feature_map.yaml").resolve()))
    return payload.get("action_features", {})


def get_action_features(action_text: str, feature_map: dict) -> dict[str, float]:
    features = feature_map.get(action_text)
    if features is None:
        return {name: 0.0 for name in FEATURE_DIMENSIONS}
    return {name: float(features.get(name, 0.0)) for name in FEATURE_DIMENSIONS}


def tci_to_expected_features(tci: dict[str, float]) -> dict[str, float]:
    ns = float(tci.get("NS", 0.5))
    ha = float(tci.get("HA", 0.5))
    rd = float(tci.get("RD", 0.5))
    persistence = float(tci.get("P", 0.5))
    return {
        "risk_avoid": clamp(ha - ns * 0.5),
        "approach": clamp(ns - ha * 0.5),
        "prosocial": clamp(rd),
        "persist": clamp(persistence),
        "passive": clamp((1 - ns) * (1 - persistence) * 0.5),
    }


def _vector(values: dict[str, float]) -> list[float]:
    return [float(values.get(name, 0.0)) for name in FEATURE_DIMENSIONS]


def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    a = _vector(left)
    b = _vector(right)
    norm_a = math.sqrt(sum(v * v for v in a))
    norm_b = math.sqrt(sum(v * v for v in b))
    if norm_a == 0 or norm_b == 0:
        return 0.5
    dot = sum(x * y for x, y in zip(a, b))
    similarity = dot / (norm_a * norm_b)
    return clamp((similarity + 1.0) / 2.0)


def personality_coherence_reward(
    output: dict,
    tci: dict[str, float],
    options: list[tuple[int, str]],
    feature_map: dict,
) -> float:
    if "action_id" not in output:
        return 0.5
    try:
        action_id = int(output["action_id"])
    except (TypeError, ValueError):
        return 0.0
    option_map = dict(options)
    if action_id not in option_map:
        return 0.0
    action_features = get_action_features(option_map[action_id], feature_map)
    expected_features = tci_to_expected_features(tci)
    similarity = _cosine_similarity(action_features, expected_features)
    return clamp((similarity - 0.5) * 2.0)


def load_emotion_transitions(config_dir: Path) -> tuple[list[dict], dict[str, str]]:
    payload = _load_yaml_cached(str((config_dir / "emotion_transitions.yaml").resolve()))
    return payload.get("emotion_transitions", []), payload.get("situation_triggers", {})


def emotion_transition_reward(
    output: dict,
    situation_type: str | None,
    transition_table: list[dict],
    situation_triggers: dict[str, str],
) -> float:
    previous = output.get("previous_emotion")
    current = output.get("emotion")
    transition_type = output.get("transition_type")
    if not previous or not current:
        return 0.5
    trigger = situation_triggers.get(str(situation_type), str(situation_type or ""))
    for row in transition_table:
        if row.get("prev") == previous and row.get("next") == current and row.get("trigger") == trigger:
            plausibility = float(row.get("plausibility", 0.5))
            expected_type = row.get("type")
            if transition_type and expected_type == transition_type:
                return plausibility
            if transition_type and expected_type != transition_type:
                return clamp(plausibility * 0.6)
            return plausibility
    if previous == current and transition_type == "sustained":
        return 0.8
    if transition_type in {"gradual", "sudden", "sustained"}:
        return 0.5
    return 0.2


def text_richness_reward(output: dict, min_avg_len: int = 15) -> float:
    string_values = [value.strip() for value in output.values() if isinstance(value, str) and value.strip()]
    if not string_values:
        return 0.0
    lowered = {value.lower() for value in string_values}
    if any(value in PLACEHOLDER_WORDS for value in lowered):
        return 0.0
    if any(len(value) < 4 for value in string_values):
        return 0.5
    avg_len = sum(len(value) for value in string_values) / len(string_values)
    if avg_len >= min_avg_len:
        return 1.0
    if avg_len >= max(5, min_avg_len / 2):
        return 0.5
    return 0.0


def _iter_text_tokens(parsed: dict) -> tuple[list[str], list[str]]:
    text_fields = []
    all_words: list[str] = []
    for value in parsed.values():
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        text_fields.append(stripped)
        all_words.extend(re.findall(r"[A-Za-z가-힣']+", stripped.lower()))
    return text_fields, all_words


def _piecewise_length_score(avg_len: float) -> float:
    points = [
        (0.0, 0.0),
        (5.0, 0.0),
        (15.0, 0.2),
        (30.0, 0.5),
        (60.0, 0.8),
        (90.0, 1.0),
    ]
    if avg_len <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if avg_len <= x1:
            if x1 == x0:
                return y1
            ratio = (avg_len - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return 1.0


def hint_quality_reward(parsed: dict) -> float:
    text_fields, all_words = _iter_text_tokens(parsed)
    if not text_fields:
        return 0.0

    avg_len = sum(len(value) for value in text_fields) / len(text_fields)
    length_score = _piecewise_length_score(avg_len)

    vocab_score = 0.0
    if all_words:
        vocab_score = len(set(all_words)) / len(all_words)

    specific_words = [word for word in all_words if word not in STOP_WORDS]
    specificity_score = 0.0
    if all_words:
        specificity_score = min((len(specific_words) / len(all_words)) * 2.0, 1.0)

    placeholder_mult = 0.1 if any(value.lower() in PLACEHOLDER_WORDS for value in text_fields) else 1.0
    score = (length_score * 0.4 + vocab_score * 0.3 + specificity_score * 0.3) * placeholder_mult
    return clamp(score)


def confidence_personality_reward(parsed: dict, tci: dict) -> float:
    actual_conf = parsed.get("confidence", 0.5)
    if not isinstance(actual_conf, (int, float)):
        return 0.5
    expected_conf = 0.5 + 0.3 * (float(tci.get("NS", 0.5)) - 0.5) - 0.2 * (float(tci.get("HA", 0.5)) - 0.5) + 0.1 * (float(tci.get("P", 0.5)) - 0.5)
    expected_conf = clamp(expected_conf, 0.2, 0.95)
    distance = abs(float(actual_conf) - expected_conf)
    return clamp(1.0 - distance)


def reasoning_action_consistency_reward(
    parsed: dict,
    options: list[tuple[int, str]],
    feature_map: dict,
) -> float:
    reasoning_text = " ".join(
        str(parsed.get(key, "")).strip().lower()
        for key in ("personality_reasoning", "reasoning")
        if parsed.get(key)
    ).strip()
    if not reasoning_text:
        return 0.5

    expected_feature = None
    for keyword, feature_name in REASONING_TO_FEATURE.items():
        if keyword in reasoning_text:
            expected_feature = feature_name
            break
    if expected_feature is None:
        return 0.5

    action_name = None
    if "action_id" in parsed:
        try:
            action_id = int(parsed["action_id"])
        except (TypeError, ValueError):
            return 0.2
        action_name = dict(options).get(action_id)
    elif "action" in parsed and isinstance(parsed["action"], str):
        action_name = parsed["action"]
    if not action_name:
        return 0.5

    action_features = get_action_features(action_name, feature_map)
    dominant_feature = max(action_features, key=action_features.get)
    dominant_value = action_features.get(dominant_feature, 0.0)
    expected_value = action_features.get(expected_feature, 0.0)

    if dominant_feature == expected_feature and dominant_value >= 0.5:
        return 1.0
    if expected_value >= 0.5:
        return 0.6
    if dominant_value >= 0.6 and dominant_feature != expected_feature:
        return 0.2
    return 0.5


def numeric_validity_reward(output: dict) -> float:
    ranges = {
        "confidence": (0.0, 1.0),
        "intensity": (0.0, 1.0),
        "lie_degree": (0.0, 1.0),
        "detection_risk": (0.0, 1.0),
        "dissent_risk": (0.0, 1.0),
        "walk_away_threshold": (0.0, 1.0),
        "emotional_charge": (-1.0, 1.0),
        "stress_delta": (-1.0, 1.0),
        "trust_delta": (-1.0, 1.0),
    }
    checked = 0
    valid = 0
    for key, (low, high) in ranges.items():
        if key not in output:
            continue
        value = output[key]
        checked += 1
        if isinstance(value, (int, float)) and low <= float(value) <= high:
            valid += 1
    if checked == 0:
        return 1.0
    return valid / checked


def _extract_diversity_token(output: dict, task: str) -> str:
    for key in ("action_id", "priority_id", "coping_id", "social_action_id", "response_id", "decision_id", "minority_position"):
        if key in output:
            return str(output[key])
    if "action" in output:
        return str(output["action"])
    if "emotion" in output:
        return str(output["emotion"])
    if "emotion_expressed" in output:
        return str(output["emotion_expressed"])
    for value in output.values():
        if isinstance(value, str) and value.strip():
            return value.strip().split()[0].lower()
    return ""


def diversity_reward(outputs: list[dict], task: str) -> float:
    if not outputs:
        return 0.0
    tokens = [_extract_diversity_token(output, task) for output in outputs]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if task in {"E", "I", "J", "K", "L", "M", "R", "T"}:
        total = len(tokens)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        max_entropy = math.log(max(len(counts), 1))
        if max_entropy == 0:
            return 0.0
        return clamp(entropy / max_entropy)
    unique_count = len({token for token in tokens if token})
    return clamp(unique_count / len(tokens))


def gate_check(output_str: str, task: str) -> tuple[dict | None, float]:
    try:
        parsed = json.loads(output_str)
    except json.JSONDecodeError:
        return None, 0.0
    if not isinstance(parsed, dict):
        return None, 0.0
    required = REQUIRED_FIELDS.get(task, set())
    if required and not required.issubset(parsed.keys()):
        return None, 0.0
    return parsed, 1.0


def _infer_trigger_from_prompt(prompt: str) -> str | None:
    lowered = prompt.lower()
    keyword_map = {
        "betrayal": ["theft", "steal", "stole", "betray", "rival", "apologize"],
        "threat": ["predator", "threat", "storm", "danger", "attack", "ambush", "flood"],
        "loss": ["injury", "death", "died", "wound", "lost", "ruined"],
        "grief": ["passed away", "funeral", "buried", "grief"],
        "success": ["found food", "massive stag", "gift", "first steps", "healthy newborn", "recovered"],
        "reconciliation": ["apologize", "forgave", "made peace", "medicine"],
        "injustice": ["ignored your warning", "unfair", "blamed"],
    }
    for trigger, keywords in keyword_map.items():
        if any(keyword in lowered for keyword in keywords):
            return trigger
    return None


def combined_reward(
    output_str: str,
    prompt: str,
    *,
    config_dir: Path,
    weights: dict[str, float] | None = None,
    group_outputs: list[str] | None = None,
) -> dict:
    task = parse_task_from_prompt(prompt)
    parsed, gate = gate_check(output_str, task)
    if gate == 0.0 or parsed is None:
        return {
            "total": 0.0,
            "gate": gate,
            "personality": 0.0,
            "emotion": 0.0,
            "plausibility": 0.0,
            "diversity": 0.0,
            "task": task,
            "details": {"error": "invalid_json_or_missing_fields"},
        }

    tci = parse_tci_from_prompt(prompt)
    options = parse_options_from_prompt(prompt)
    feature_map = load_action_features(config_dir)
    transition_table, situation_triggers = load_emotion_transitions(config_dir)
    situation_type = parse_situation_from_prompt(prompt) or _infer_trigger_from_prompt(prompt)

    personality = personality_coherence_reward(parsed, tci, options, feature_map) if task == "E" else 0.5
    emotion = emotion_transition_reward(parsed, situation_type, transition_table, situation_triggers) if task == "F" else 0.5
    text_score = text_richness_reward(parsed)
    hint_qual = hint_quality_reward(parsed)
    conf_fit = confidence_personality_reward(parsed, tci) if tci.get("type") != "unknown" else 0.5
    reason_fit = reasoning_action_consistency_reward(parsed, options, feature_map) if options or "action" in parsed else 0.5
    numeric_score = numeric_validity_reward(parsed)
    plausibility = (hint_qual + numeric_score) / 2.0

    diversity = 0.0
    if group_outputs:
        group_parsed = [json.loads(candidate) for candidate in group_outputs if gate_check(candidate, task)[1] == 1.0]
        diversity = diversity_reward(group_parsed, task)

    task_weights = dict(DEFAULT_WEIGHTS)
    task_weights.update(TASK_WEIGHTS.get(task, {}))
    components = {
        "personality": personality,
        "emotion": emotion,
        "hint_quality": hint_qual,
        "confidence_fit": conf_fit,
        "reasoning_fit": reason_fit,
    }
    if group_outputs:
        components["diversity"] = diversity

    active_weights = {name: task_weights.get(name, 0.0) for name in components}
    total_weight = sum(active_weights.values()) or 1.0
    total = gate * sum(components[name] * active_weights[name] for name in components) / total_weight

    return {
        "total": clamp(total),
        "gate": gate,
        "personality": personality,
        "emotion": emotion,
        "plausibility": plausibility,
        "diversity": diversity,
        "task": task,
        "details": {
            "tci": tci,
            "options": options,
            "situation_type": situation_type,
            "text_richness": text_score,
            "hint_quality": hint_qual,
            "confidence_fit": conf_fit,
            "reasoning_fit": reason_fit,
            "numeric_validity": numeric_score,
            "weights": active_weights,
        },
    }


def score_best_of_n(
    prompt: str,
    outputs: list[str],
    config_dir: Path,
) -> list[dict]:
    scored = []
    for output in outputs:
        reward = combined_reward(output, prompt, config_dir=config_dir)
        scored.append({"output": output, "reward": reward})
    scored.sort(key=lambda item: item["reward"]["total"], reverse=True)
    for index, item in enumerate(scored, start=1):
        item["rank"] = index
    return scored


def select_dpo_pair(scored: list[dict], min_gap: float = 0.05) -> tuple[str, str] | None:
    if len(scored) < 2:
        return None
    chosen = scored[0]
    rejected = scored[-1]
    if chosen["reward"]["total"] - rejected["reward"]["total"] < min_gap:
        return None
    return chosen["output"], rejected["output"]
