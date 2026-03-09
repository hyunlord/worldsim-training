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


def test_convert_mixed_final_to_training_format_builds_messages_and_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path
    prompts_dir = repo_root / "prompts" / "training"
    config_dir = repo_root / "config"
    source_dir = repo_root / "data" / "final" / "worldsim-v31-mix-v1"
    output_dir = repo_root / "data" / "training" / "worldsim-v31-mix-v1"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "generation.yaml").write_text(
        """
prompts:
  training:
    layer0_system: prompts/training/layer0_system.txt
    layer3_system: prompts/training/layer3_system.txt
    layer4_system: prompts/training/layer4_system.txt
    layer5_system: prompts/training/layer5_system.txt
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (prompts_dir / "layer0_system.txt").write_text("L0 system", encoding="utf-8")
    (prompts_dir / "layer3_system.txt").write_text("L3 system", encoding="utf-8")
    (prompts_dir / "layer4_system.txt").write_text("L4 system", encoding="utf-8")
    (prompts_dir / "layer5_system.txt").write_text("L5 system", encoding="utf-8")

    write_jsonl(
        source_dir / "train.jsonl",
        [
            {
                "task": "A",
                "layer": "L4",
                "prompt": "[TASK] A",
                "output": compact_json(
                    {
                        "text_ko": "뒤를 살피며 조심조심 걷는다.",
                        "text_en": "They walk with care.",
                        "register": "haera",
                        "dominant_trait": "harm_avoidance",
                        "temperament_expressed": "melancholic",
                    }
                ),
                "merge_source_batch": "batch_v31_01_abc",
                "merge_source_split": "train",
                "merge_dataset_id": "worldsim-v31-mix-v1",
            },
            {
                "task": "H",
                "layer": "L0",
                "prompt": "[TASK] H",
                "output": compact_json(
                    {
                        "name": "DungeonEconomy",
                        "description_en": "Dungeon rules govern survival.",
                        "resource_modifiers": [{"target": "dungeon_loot", "multiplier": 3.0}],
                        "special_zones": [{"kind": "dungeon_node", "spawn_count_min": 3, "spawn_count_max": 7}],
                        "special_resources": [{"name": "magic_stone", "tags": ["currency"]}],
                        "agent_modifiers": [{"system": "temperament", "trigger": "essence_equip", "effect": "shift_random_axis"}],
                    }
                ),
                "merge_source_batch": "batch_v31_02_gefhc",
                "merge_source_split": "train",
                "merge_dataset_id": "worldsim-v31-mix-v1",
            },
        ],
    )
    write_jsonl(
        source_dir / "dev.jsonl",
        [
            {
                "task": "G",
                "layer": "L5",
                "prompt": "[TASK] G",
                "output": compact_json(
                    {
                        "interpretation_ko": "북멧등만 넘으면 먹거리가 넉넉히 기다리오",
                        "interpretation_en": "Food waits beyond the northern ridge.",
                        "action_tendency": "mobilize",
                        "confidence": 0.74,
                        "register": "hao",
                        "misinterpretation_type": "overconfident_literal",
                        "temperament_bias": "scarcity_driven_decisive_literalism",
                    }
                ),
                "merge_source_batch": "batch_v31_02_gefhc",
                "merge_source_split": "dev",
                "merge_dataset_id": "worldsim-v31-mix-v1",
            }
        ],
    )
    (source_dir / "merge_manifest.json").write_text(
        json.dumps({"dataset_id": "worldsim-v31-mix-v1"}, ensure_ascii=False),
        encoding="utf-8",
    )

    from scripts.convert_mixed_final_to_training_format import convert_mixed_final_to_training_format

    result = convert_mixed_final_to_training_format(
        repo_root=repo_root,
        input_train=source_dir / "train.jsonl",
        input_dev=source_dir / "dev.jsonl",
        source_manifest=source_dir / "merge_manifest.json",
        output_dir=output_dir,
        dataset_id="worldsim-v31-mix-v1",
    )

    train_rows = [json.loads(line) for line in result.train_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in result.dev_output.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert len(train_rows) == 2
    assert len(dev_rows) == 1
    assert train_rows[0]["messages"][0]["content"] == "L4 system"
    assert train_rows[1]["messages"][0]["content"] == "L0 system"
    assert dev_rows[0]["messages"][0]["content"] == "L5 system"
    assert json.loads(train_rows[0]["messages"][2]["content"])["dominant_trait"] == "harm_avoidance"
    assert json.loads(train_rows[1]["messages"][2]["content"])["name"] == "DungeonEconomy"
    assert train_rows[0]["source_batch"] == "batch_v31_01_abc"
    assert train_rows[1]["source_dataset_id"] == "worldsim-v31-mix-v1"
    assert manifest["counts"]["train_converted"] == 2
    assert manifest["counts"]["dev_converted"] == 1
    assert manifest["counts"]["excluded_total"] == 0
    assert manifest["detected_training_format"] == "messages"


def test_convert_mixed_final_to_training_format_rejects_missing_prompt(tmp_path: Path) -> None:
    repo_root = tmp_path
    prompts_dir = repo_root / "prompts" / "training"
    config_dir = repo_root / "config"
    source_dir = repo_root / "data" / "final" / "worldsim-v31-mix-v1"
    output_dir = repo_root / "data" / "training" / "worldsim-v31-mix-v1"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "generation.yaml").write_text(
        """
prompts:
  training:
    layer4_system: prompts/training/layer4_system.txt
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (prompts_dir / "layer4_system.txt").write_text("L4 system", encoding="utf-8")

    write_jsonl(
        source_dir / "train.jsonl",
        [
            {
                "task": "A",
                "layer": "L4",
                "output": compact_json(
                    {
                        "text_ko": "뒤를 살피며 조심조심 걷는다.",
                        "text_en": "They walk with care.",
                        "register": "haera",
                        "dominant_trait": "harm_avoidance",
                        "temperament_expressed": "melancholic",
                    }
                ),
            }
        ],
    )
    write_jsonl(source_dir / "dev.jsonl", [])

    from scripts.convert_mixed_final_to_training_format import convert_mixed_final_to_training_format

    try:
        convert_mixed_final_to_training_format(
            repo_root=repo_root,
            input_train=source_dir / "train.jsonl",
            input_dev=source_dir / "dev.jsonl",
            source_manifest=None,
            output_dir=output_dir,
            dataset_id="worldsim-v31-mix-v1",
        )
    except ValueError as exc:
        assert "missing prompt/output" in str(exc)
    else:
        raise AssertionError("Expected conversion to fail when prompt is missing")
