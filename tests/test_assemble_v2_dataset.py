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


def test_assemble_v2_dataset_merges_sources_and_writes_manifest(tmp_path: Path) -> None:
    repo_root = tmp_path
    write_jsonl(
        repo_root / "data" / "final" / "worldsim-v31-mix-v1" / "train.jsonl",
        [
            {"task": "A", "prompt": "[TASK] A", "output": compact_json({"text_ko": "살핀다.", "text_en": "They scan.", "register": "haera", "dominant_trait": "harm_avoidance", "temperament_expressed": "melancholic"})},
            {"task": "B", "prompt": "[TASK] B", "output": compact_json({"text_ko": "겁이 솟아 물러선다.", "text_en": "Fear rises and they step back.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.7, "mimetics": [], "temperament_influence": "fear_sharpens_retreat"})},
        ],
    )
    write_jsonl(
        repo_root / "data" / "final" / "worldsim-v31-mix-v1" / "dev.jsonl",
        [
            {"task": "G", "prompt": "[TASK] G", "output": compact_json({"interpretation_ko": "이 말은 북멧등을 넘으라 한다.", "interpretation_en": "This means cross the northern ridge.", "action_tendency": "mobilize", "confidence": 0.6, "register": "hao", "misinterpretation_type": "overconfident_literal", "temperament_bias": "decisive"})}
        ],
    )
    write_jsonl(
        repo_root / "data" / "validated" / "batch_v2_01_tasks_in" / "passed.jsonl",
        [
            {"task": "I", "prompt": "[TASK] I", "output": compact_json({"priority_id": 0, "reasoning_ko": "먼저 먹거리를 찾는다.", "reasoning_en": "Food comes first.", "need_addressed": "hunger", "urgency": 0.9})},
            {"task": "J", "prompt": "[TASK] J", "output": compact_json({"coping_id": 1, "coping_type": "acceptance", "stress_delta": -0.3, "hint_ko": "숨을 고르며 받아들인다.", "hint_en": "They steady their breath and accept it.", "side_effect": "none"})},
        ],
    )
    write_jsonl(
        repo_root / "data" / "validated" / "batch_v2_02_task_g_fix" / "passed.jsonl",
        [
            {"task": "G", "prompt": "[TASK] G", "output": compact_json({"interpretation_ko": "이 말은 불씨를 지키라 한다.", "interpretation_en": "This means guard the ember.", "action_tendency": "defend", "confidence": 0.7, "register": "hao", "misinterpretation_type": "fearful_symbolic", "temperament_bias": "guarded"})},
            {"task": "G", "prompt": "[TASK] G", "output": compact_json({"interpretation_ko": "이 말은 불씨를 지키라 한다.", "interpretation_en": "This means guard the ember.", "action_tendency": "defend", "confidence": 0.7, "register": "hao", "misinterpretation_type": "fearful_symbolic", "temperament_bias": "guarded"})},
        ],
    )
    write_jsonl(
        repo_root / "data" / "samples" / "negative_examples.jsonl",
        [{"task": "NEG", "label": "reject", "output": "꼼꼼하게 꼼꼼하게 꼼꼼하게", "reason": "repetition_loop"}],
    )
    write_jsonl(
        repo_root / "data" / "samples" / "general_korean.jsonl",
        [{"task": "GEN", "label": "retain", "prompt": "하늘을 그려줘.", "output": "맑은 하늘에 흰 구름이 느리게 흘러간다."}],
    )

    from scripts.assemble_v2_dataset import assemble_v2_dataset

    result = assemble_v2_dataset(repo_root, output_dir=repo_root / "data" / "final" / "worldsim-v2-mix", seed=7)
    train_rows = [json.loads(line) for line in result.train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev_rows = [json.loads(line) for line in result.dev_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["sources"]["batch1_total"] == 2
    assert manifest["sources"]["batch2_total"] == 2
    assert manifest["sources"]["negatives"] == 1
    assert manifest["sources"]["general"] == 1
    assert manifest["deduplication"]["removed"] >= 1
    assert manifest["output"]["total"] == len(train_rows) + len(dev_rows)
    assert {"A", "B", "I", "J", "NEG", "GEN"} <= {row["task"] for row in train_rows + dev_rows}
    assert "G" in manifest["task_counts"]["train"] or "G" in manifest["task_counts"]["dev"]

