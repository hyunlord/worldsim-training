from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.convert_mixed_final_to_training_format import SUPPORTED_TASKS
from scripts.prepare_dataset import _row_to_training_example, _training_system_prompts


def _compact(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def test_training_system_prompts_include_l3_en_default() -> None:
    prompts = _training_system_prompts(None, {})
    assert "L3_EN" in prompts
    assert "Use English for all text fields." in prompts["L3_EN"]


def test_row_to_training_example_routes_v3_english_tasks_to_l3_en() -> None:
    prompts = _training_system_prompts(None, {})
    row = {
        "task": "O",
        "prompt": "[TASK] O",
        "output": _compact(
            {
                "public_claim": "We found nothing.",
                "private_truth": "I hid extra food for myself.",
                "deception_style": "omission",
                "lie_degree": 0.7,
                "detection_risk": 0.3,
                "confidence": 0.8,
            }
        ),
        "layer": "L3",
        "schema_version": 3,
    }

    result = _row_to_training_example(row, prompts)

    assert result["task"] == "O"
    assert result["messages"][0]["content"] == prompts["L3_EN"]


def test_row_to_training_example_keeps_v2_korean_task_routing() -> None:
    prompts = _training_system_prompts(None, {})
    row = {
        "task": "B",
        "prompt": "[TASK] B",
        "output": _compact(
            {
                "text_ko": "곧바로 몸을 낮추고 숨을 고른다.",
                "text_en": "They lower their body and steady their breath.",
                "register": "haera",
                "emotion_expressed": "fear",
                "intensity": 0.8,
                "mimetics": [],
                "temperament_influence": "조심성이 반응을 늦추지 않았다.",
            }
        ),
        "layer": "L4",
    }

    result = _row_to_training_example(row, prompts)

    assert result["messages"][0]["content"] == prompts["L4"]


def test_row_to_training_example_supports_structured_tasks_o_through_t() -> None:
    prompts = _training_system_prompts(None, {})
    sample_outputs = {
        "O": {"public_claim": "We are starving too.", "private_truth": "I hid meat.", "deception_style": "half_truth", "lie_degree": 0.5, "detection_risk": 0.4, "confidence": 0.7},
        "P": {"retold_version": "People say the wolf nearly killed him.", "distortion_type": "exaggeration", "added_detail": "It was foaming at the mouth.", "dropped_detail": "The hunter drove it off.", "emotional_charge": 0.6},
        "Q": {"trauma_response": "hypervigilance", "behavioral_change": "They scan every riverbank before letting anyone approach.", "trigger_situation": "Children going near deep water.", "intensity": 0.9, "duration": "long_term", "coping_mechanism": "Constant supervision."},
        "R": {"action": "counter_offer", "counter_give": "1 knife", "counter_want": "8 dried fish", "reasoning": "The exchange is too weak on our side.", "emotional_state": "anticipation", "walk_away_threshold": 0.6},
        "S": {"action": "modify", "modified_practice": "Adopt the ritual only at winter festivals.", "reasoning": "It preserves our identity while borrowing the useful part.", "social_pressure": 0.4, "tradition_conflict": True},
        "T": {"decision_id": 1, "confidence": 0.7, "dissent_risk": 0.5, "minority_position": 3, "minority_action": "grumble", "spark_event": "food_shortage", "reasoning": "Some hunters will accept the ruling but resent the ration cuts.", "timeline": "conditional"},
    }

    for task, output in sample_outputs.items():
        row = {
            "task": task,
            "prompt": f"[TASK] {task}",
            "output": _compact(output),
            "layer": "L3",
            "schema_version": 3,
        }
        result = _row_to_training_example(row, prompts)
        assert result["task"] == task
        assert result["messages"][0]["content"] == prompts["L3_EN"]


def test_supported_tasks_includes_o_through_t() -> None:
    for task in "OPQRST":
        assert task in SUPPORTED_TASKS


def test_new_batch_configs_parse_correctly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    batch_f = yaml.safe_load((repo_root / "config" / "batches" / "batch_v31_04_task_f_reinforce.yaml").read_text(encoding="utf-8"))
    batch_pairs = yaml.safe_load((repo_root / "config" / "batches" / "batch_v31_05_personality_pairs.yaml").read_text(encoding="utf-8"))

    assert batch_f["schema_version"] == 3
    assert batch_f["tasks"] == {"F": 200}
    assert batch_pairs["schema_version"] == 3
    assert batch_pairs["tasks"] == {"E": 100, "M": 50, "O": 50}


def test_generation_config_exposes_layer3_en_prompt_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / "config" / "generation.yaml").read_text(encoding="utf-8"))
    assert config["prompts"]["training"]["layer3_en_system"] == "prompts/training/layer3_en_system.txt"
