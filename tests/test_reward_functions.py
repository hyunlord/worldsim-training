from __future__ import annotations

import json
from pathlib import Path

from scripts.reward_functions import (
    combined_reward,
    diversity_reward,
    gate_check,
    load_action_features,
    load_emotion_transitions,
    parse_options_from_prompt,
    parse_tci_from_prompt,
    personality_coherence_reward,
    score_best_of_n,
    select_dpo_pair,
    tci_to_expected_features,
    text_richness_reward,
    emotion_transition_reward,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_parse_tci_from_prompt_extracts_axes() -> None:
    tci = parse_tci_from_prompt("[TEMP]\nNS=0.8 HA=0.2 RD=0.5 P=0.7 type=choleric")
    assert tci == {"NS": 0.8, "HA": 0.2, "RD": 0.5, "P": 0.7, "type": "choleric"}


def test_parse_tci_fallback_on_missing() -> None:
    tci = parse_tci_from_prompt("[TASK] E")
    assert tci == {"NS": 0.5, "HA": 0.5, "RD": 0.5, "P": 0.5, "type": "unknown"}


def test_parse_options_korean() -> None:
    options = parse_options_from_prompt("[OPTIONS]\n0:도망 1:숨기 2:맞서기")
    assert options == [(0, "도망"), (1, "숨기"), (2, "맞서기")]


def test_parse_options_english() -> None:
    options = parse_options_from_prompt("[OPTIONS]\n0:confront 1:observe 2:retreat")
    assert options == [(0, "confront"), (1, "observe"), (2, "retreat")]


def test_tci_to_expected_features_high_ns() -> None:
    features = tci_to_expected_features({"NS": 0.9, "HA": 0.1, "RD": 0.5, "P": 0.5})
    assert features["approach"] > features["risk_avoid"]


def test_tci_to_expected_features_high_ha() -> None:
    features = tci_to_expected_features({"NS": 0.1, "HA": 0.9, "RD": 0.5, "P": 0.5})
    assert features["risk_avoid"] > features["approach"]


def test_personality_coherence_high_ns_confront() -> None:
    feature_map = load_action_features(CONFIG_DIR)
    tci = {"NS": 0.9, "HA": 0.1, "RD": 0.5, "P": 0.5}
    reward = personality_coherence_reward({"action_id": 0}, tci, [(0, "confront"), (1, "retreat")], feature_map)
    assert reward > 0.7


def test_personality_coherence_high_ha_retreat() -> None:
    feature_map = load_action_features(CONFIG_DIR)
    tci = {"NS": 0.1, "HA": 0.9, "RD": 0.5, "P": 0.5}
    reward = personality_coherence_reward({"action_id": 1}, tci, [(0, "confront"), (1, "retreat")], feature_map)
    assert reward > 0.7


def test_personality_coherence_mismatch() -> None:
    feature_map = load_action_features(CONFIG_DIR)
    tci = {"NS": 0.1, "HA": 0.9, "RD": 0.5, "P": 0.5}
    reward = personality_coherence_reward({"action_id": 0}, tci, [(0, "confront"), (1, "retreat")], feature_map)
    assert reward < 0.6


def test_emotion_transition_joy_to_anger_betrayal() -> None:
    transition_table, triggers = load_emotion_transitions(CONFIG_DIR)
    reward = emotion_transition_reward(
        {"emotion": "anger", "previous_emotion": "joy", "transition_type": "sudden"},
        "theft",
        transition_table,
        triggers,
    )
    assert reward > 0.9


def test_emotion_transition_implausible() -> None:
    transition_table, triggers = load_emotion_transitions(CONFIG_DIR)
    reward = emotion_transition_reward(
        {"emotion": "anger", "previous_emotion": "joy", "transition_type": "gradual"},
        "nothing",
        transition_table,
        triggers,
    )
    assert reward < 0.3


def test_text_richness_placeholder() -> None:
    assert text_richness_reward({"hint": "str", "reasoning": "English"}) == 0.0


def test_text_richness_real_content() -> None:
    reward = text_richness_reward({"hint": "They rush forward to seize the advantage before fear spreads."})
    assert reward == 1.0


def test_gate_check_valid_json() -> None:
    parsed, gate = gate_check(
        '{"action_id":0,"confidence":0.9,"hint":"Charge forward now.","personality_reasoning":"novelty_seeking","temperament_factor":"bold"}',
        "E",
    )
    assert gate == 1.0
    assert parsed is not None


def test_gate_check_invalid_json() -> None:
    parsed, gate = gate_check("{bad json", "E")
    assert parsed is None
    assert gate == 0.0


def test_combined_reward_returns_all_components() -> None:
    result = combined_reward(
        '{"action_id":0,"confidence":0.9,"hint":"Charge forward before hesitation spreads.","personality_reasoning":"novelty_seeking","temperament_factor":"bold"}',
        "[TASK] E\n[TEMP]\nNS=0.8 HA=0.2 RD=0.5 P=0.7 type=choleric\n[SITUATION] 누군가 먹거리를 훔친 것이 드러났다\n[OPTIONS]\n0:confront 1:observe 2:retreat",
        config_dir=CONFIG_DIR,
    )
    assert {"total", "gate", "personality", "emotion", "plausibility", "diversity", "task", "details"} <= set(result)
    assert result["task"] == "E"
    assert result["gate"] == 1.0


def test_score_best_of_n_sorted_by_reward() -> None:
    outputs = [
        '{"action_id":2,"confidence":0.5,"hint":"Retreat quietly.","personality_reasoning":"harm_avoidance","temperament_factor":"timid"}',
        '{"action_id":0,"confidence":0.9,"hint":"Charge forward before hesitation spreads.","personality_reasoning":"novelty_seeking","temperament_factor":"bold"}',
    ]
    scored = score_best_of_n(
        "[TASK] E\n[TEMP]\nNS=0.8 HA=0.2 RD=0.5 P=0.7 type=choleric\n[OPTIONS]\n0:confront 1:observe 2:retreat",
        outputs,
        CONFIG_DIR,
    )
    assert scored[0]["reward"]["total"] >= scored[1]["reward"]["total"]


def test_select_dpo_pair_returns_best_worst() -> None:
    scored = [
        {"output": "best", "reward": {"total": 0.9}},
        {"output": "mid", "reward": {"total": 0.6}},
        {"output": "worst", "reward": {"total": 0.3}},
    ]
    assert select_dpo_pair(scored) == ("best", "worst")


def test_select_dpo_pair_none_if_gap_small() -> None:
    scored = [
        {"output": "a", "reward": {"total": 0.61}},
        {"output": "b", "reward": {"total": 0.55}},
    ]
    assert select_dpo_pair(scored) is None


def test_diversity_reward_all_same() -> None:
    outputs = [{"action_id": 0} for _ in range(8)]
    assert diversity_reward(outputs, "E") == 0.0


def test_diversity_reward_all_different() -> None:
    outputs = [{"action_id": index} for index in range(8)]
    assert diversity_reward(outputs, "E") == 1.0
