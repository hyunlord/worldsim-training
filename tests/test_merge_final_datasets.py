from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def make_row(task: str, output: dict, **extra: object) -> dict:
    return {
        "task": task,
        "output": compact_json(output),
        **extra,
    }


def test_merge_final_datasets_applies_task_caps_and_preserves_provenance(tmp_path: Path) -> None:
    batch1_dir = tmp_path / "data" / "final" / "batch_v31_01_abc"
    batch2_dir = tmp_path / "data" / "final" / "batch_v31_02_gefhc"
    output_dir = tmp_path / "data" / "final" / "worldsim-v31-mix-v1"

    shared_c = {
        "speech_ko": "다들 앞으로 나서라.",
        "speech_en": "Everyone, step forward.",
        "register": "haera",
        "emotion_expressed": "anger",
        "speaker_role": "chief",
        "temperament_tone": "direct_command",
    }

    write_jsonl(
        batch1_dir / "train.jsonl",
        [
            make_row("A", {"text_ko": "뒤를 살피며 조심조심 걷는다.", "text_en": "They walk with care.", "register": "haera", "dominant_trait": "harm_avoidance", "temperament_expressed": "melancholic"}, personality_id="a1"),
            make_row("B", {"text_ko": "짐승 그림자에 오들오들 떨었다. 곧 물러섰다.", "text_en": "They trembled at the beast's shadow and stepped back.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.8, "mimetics": ["오들오들"], "temperament_influence": "fear_sharpens_retreat"}, personality_id="b1"),
            make_row("B", {"text_ko": "비바람에 이를 악물고 버텼다. 어깨를 세웠다.", "text_en": "They gritted their teeth in the storm and held firm.", "register": "haera", "emotion_expressed": "anger", "intensity": 0.6, "mimetics": [], "temperament_influence": "anger_holds_the_line"}, personality_id="b2"),
            make_row("B", {"text_ko": "낯선 이를 보며 숨을 고르고 살폈다. 틈을 재었다.", "text_en": "They steadied their breath when they saw the stranger and measured the opening.", "register": "haera", "emotion_expressed": "anticipation", "intensity": 0.5, "mimetics": [], "temperament_influence": "anticipation_drives_patience"}, personality_id="b3"),
            make_row("C", shared_c, personality_id="c1"),
        ],
    )
    write_jsonl(
        batch1_dir / "dev.jsonl",
        [
            make_row("A", {"text_ko": "발걸음은 느려도 끝까지 살핀다.", "text_en": "They inspect things to the end.", "register": "haera", "dominant_trait": "persistence", "temperament_expressed": "phlegmatic"}, personality_id="a2"),
            make_row("B", {"text_ko": "먹거리를 보자 방긋 웃었다. 손을 서둘러 놀렸다.", "text_en": "They smiled when they saw food and moved quickly.", "register": "haera", "emotion_expressed": "joy", "intensity": 0.7, "mimetics": ["방긋"], "temperament_influence": "joy_speeds_action"}, personality_id="b4"),
            make_row("B", {"text_ko": "불씨를 움켜쥐고 둘레를 지켜보았다. 다가올 때를 쟀다.", "text_en": "They gripped the ember and watched around them, timing what would come next.", "register": "haera", "emotion_expressed": "anticipation", "intensity": 0.6, "mimetics": [], "temperament_influence": "anticipation_holds_focus"}, personality_id="b5"),
        ],
    )

    write_jsonl(
        batch2_dir / "train.jsonl",
        [
            make_row("C", {"speech_ko": "다들 앞으로 나서라. ", "speech_en": "Everyone, step forward.", "register": "haera", "emotion_expressed": "anger", "speaker_role": "chief", "temperament_tone": "direct_command"}, personality_id="c2"),
            make_row("C", {"speech_ko": "숨을 죽이고 둘레를 살펴라.", "speech_en": "Hold your breath and scan the surroundings.", "register": "haera", "emotion_expressed": "fear", "speaker_role": "scout", "temperament_tone": "guarded_warning"}, personality_id="c4"),
            make_row("E", {"action_id": 0, "confidence": 0.9, "hint_ko": "겁이 치밀어 곧장 달아났다", "hint_en": "Fear surged, so they fled.", "personality_reasoning": "high_harm_avoidance", "temperament_factor": "harm_avoidance_dominant"}, personality_id="e1"),
            make_row("F", {"emotion": "fear", "intensity": 0.88, "cause_ko": "날랜 짐승이 앞을 막아 겁이 솟았다", "cause_en": "A swift beast blocked the path and fear surged.", "previous_emotion": "trust", "transition_type": "sudden", "temperament_amplifier": "high_HA_intensifies_fear"}, personality_id="f1"),
            make_row("G", {"interpretation_ko": "북멧등만 넘으면 먹거리가 넉넉히 기다리오", "interpretation_en": "Food waits beyond the northern ridge.", "action_tendency": "mobilize", "confidence": 0.74, "register": "hao", "misinterpretation_type": "overconfident_literal", "temperament_bias": "scarcity_driven_decisive_literalism"}, personality_id="g1", oracle_id="oracle_01"),
            make_row("H", {"name": "WitchCursedWasteland", "description_en": "The surface is desolate and labyrinths hold the resources.", "resource_modifiers": [], "special_zones": [{"kind": "labyrinth", "spawn_count_min": 1, "spawn_count_max": 10}], "special_resources": [{"name": "mana_stone", "tags": ["currency"]}], "agent_modifiers": [{"system": "temperament", "trigger": "essence_equipped", "effect": "change_temperament"}]}, worldbuilding_id="wb_01"),
        ],
    )
    write_jsonl(
        batch2_dir / "dev.jsonl",
        [
            make_row("C", shared_c, personality_id="c3"),
            make_row("G", {"interpretation_ko": "불쥔 이가 앞에 서니 내가 곧 무리끌이야", "interpretation_en": "The one with fire stands first, so I should lead.", "action_tendency": "mobilize", "confidence": 0.92, "register": "hae", "misinterpretation_type": "overconfident_literal", "temperament_bias": "dominance_driven_self_selection"}, personality_id="g2", oracle_id="oracle_07"),
        ],
    )

    manifest_payload = {
        "dataset_name": "source",
        "counts": {"train": 0, "dev": 0, "included_total": 0, "excluded_total": 0},
        "included_task_counts": {},
    }
    (batch1_dir / "dataset_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    (batch2_dir / "dataset_manifest.json").write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")

    from scripts.merge_final_datasets import merge_final_datasets

    result = merge_final_datasets(
        batch1_train=batch1_dir / "train.jsonl",
        batch1_dev=batch1_dir / "dev.jsonl",
        batch1_manifest=batch1_dir / "dataset_manifest.json",
        batch2_train=batch2_dir / "train.jsonl",
        batch2_dev=batch2_dir / "dev.jsonl",
        batch2_manifest=batch2_dir / "dataset_manifest.json",
        output_dir=output_dir,
        dataset_id="worldsim-v31-mix-v1",
        b_train_cap=2,
        b_dev_cap=1,
        seed=7,
    )

    train_rows = [json.loads(line) for line in result.train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in result.dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    excluded_rows = [json.loads(line) for line in result.excluded_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert len([row for row in train_rows if row["task"] == "B"]) == 2
    assert len([row for row in dev_rows if row["task"] == "B"]) == 1
    assert len([row for row in dev_rows if row["task"] == "C"]) == 1
    assert len([row for row in train_rows if row["task"] == "C"]) == 1
    assert {row["task"] for row in train_rows} >= {"A", "B", "C", "E", "F", "G", "H"}
    assert {row["merge_source_batch"] for row in train_rows + dev_rows} == {"batch_v31_01_abc", "batch_v31_02_gefhc"}
    assert {row["merge_source_split"] for row in train_rows + dev_rows} <= {"train", "dev"}

    exclusion_reasons = {row["merge_exclusion_reason"] for row in excluded_rows}
    assert "task_cap" in exclusion_reasons
    assert "duplicate_content" in exclusion_reasons
    assert manifest["b_cap"]["train"] == 2
    assert manifest["b_cap"]["dev"] == 1
    assert manifest["included_counts_by_task"]["B"]["train"] == 2
    assert manifest["included_counts_by_task"]["B"]["dev"] == 1
    assert manifest["duplicate_filtering"]["excluded_total"] >= 1
