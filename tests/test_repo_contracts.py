from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_data import generate_dataset
from scripts.prepare_dataset import prepare_dataset
from scripts.validate_data import validate_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_bootstrap_assets(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "teacher").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "training").mkdir(parents=True, exist_ok=True)
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "validated").mkdir(parents=True, exist_ok=True)
    (root / "data" / "final").mkdir(parents=True, exist_ok=True)
    (root / "data" / "samples").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "manifests").mkdir(parents=True, exist_ok=True)

    (root / "config" / "generation.yaml").write_text(
        """
paths:
  situations: config/situations.yaml
  personalities: config/personalities.yaml
  emotions: config/emotions.yaml
  teacher_system_prompt: prompts/teacher/system.txt
  raw_dir: data/raw
  validated_dir: data/validated
  final_dir: data/final
  samples_dir: data/samples
  manifests_dir: artifacts/manifests
prompts:
  training:
    layer3_system: prompts/training/layer3_system.txt
    layer4_system: prompts/training/layer4_system.txt
task_variants:
  A: 1
  B: 1
  C: 1
  D: 1
  E: 0
  F: 0
validation:
  forbidden_words:
    - 식량
    - 마을
    - 전투
  trait_axes: [honesty_humility, emotionality, extraversion, agreeableness, conscientiousness, openness]
  reasoning_axes: [high_honesty_humility, high_emotionality, high_extraversion, high_agreeableness, high_conscientiousness, high_openness]
  speaker_roles: [elder, hunter, shaman, warrior, healer, gatherer, craftsman, chief, scout, observer]
  transition_types: [gradual, sudden, sustained]
  register_endings:
    haera: ['다[.\\s]?$', '는다[.\\s]?$']
    hao: ['오[.\\s!?]?$', '소[.\\s!?]?$']
    hae: ['해[.\\s!?]?$', '야[.\\s!?]?$']
  task_limits:
    A: {min_chars: 20, max_chars: 40, sentences: 1}
    B: {min_chars: 30, max_chars: 60, sentences: 2}
    C: {min_chars: 15, max_chars: 30, sentences: 1}
    D: {min_chars: 10, max_chars: 25, sentences: 1}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    (root / "config" / "situations.yaml").write_text(
        """
situations:
  - id: predator
    ko: 짐승발견
    desc: 날랜 짐승이 무리 곁에 나타났다
    action_options: [도망, 숨기, 맞서기, 경고, 얼어붙기]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "config" / "personalities.yaml").write_text(
        """
personalities:
  - id: cautious_elder
    ko: 신중한원로
    keywords:
      - 겁많음
      - 꼼꼼함
      - 조용함
    default_register: hao
    desc: 위험을 경계하는 어른
    dominant_trait: conscientiousness
    speaker_role: elder
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "config" / "emotions.yaml").write_text(
        """
emotions:
  - id: fear
    ko: 두려움
    intensities:
      - 0.9
    mimetics:
      - 오들오들
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "prompts" / "teacher" / "system.txt").write_text(
        "너는 석기시대 서사 데이터를 만든다.\n",
        encoding="utf-8",
    )
    (root / "prompts" / "teacher" / "task_a.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "register":"haera", "dominant_trait":"{dominant_trait}", "temperament_expressed":"mixed"}\n',
        encoding="utf-8",
    )
    (root / "prompts" / "teacher" / "task_b.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "register":"haera", "emotion_expressed":"{emotion_id}", "intensity":0.9, "mimetics":["{mimetic}"], "temperament_influence":"mixed_temperament_restrained_fear"}\n',
        encoding="utf-8",
    )
    (root / "prompts" / "teacher" / "task_c.txt").write_text(
        '{"speech_ko":"...", "speech_en":"...", "register":"{register}", "emotion_expressed":"{emotion_id}", "speaker_role":"{speaker_role}"}\n',
        encoding="utf-8",
    )
    (root / "prompts" / "teacher" / "task_d.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "event_type":"{situation_id}"}\n',
        encoding="utf-8",
    )
    (root / "prompts" / "training" / "layer3_system.txt").write_text("JSON only bilingual output.", encoding="utf-8")
    (root / "prompts" / "training" / "layer4_system.txt").write_text("JSON only bilingual output.", encoding="utf-8")


def test_generate_data_loads_repo_assets_and_writes_to_raw(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)

    def fake_generator(job: dict, system_prompt: str) -> str:
        assert "석기시대" in system_prompt
        return compact_json(
            {
                "text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.",
                "text_en": "Fearless and always sees things through.",
                "register": "haera",
                "dominant_trait": "conscientiousness",
                "temperament_expressed": "mixed",
            }
        )

    result = generate_dataset(tmp_path, generator=fake_generator, limit=1)

    assert result.output_path.parent == tmp_path / "data" / "raw"
    assert result.output_path.exists()
    rows = read_jsonl(result.output_path)
    assert len(rows) == 1
    assert json.loads(rows[0]["output"])["text_en"] == "Fearless and always sees things through."


def test_validate_data_reads_raw_and_writes_pass_fail_files(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)
    raw_path = tmp_path / "data" / "raw" / "batch.jsonl"
    write_jsonl(
        raw_path,
        [
            {"task": "D", "situation_id": "predator", "output": compact_json({"text_ko": "돌이가 강가에서 물고기를 잡았다.", "text_en": "Dol-i caught a fish by the river.", "event_type": "predator"})},
            {"task": "D", "situation_id": "predator", "output": compact_json({"text_ko": "식량을 발견했다.", "text_en": "Found food.", "event_type": "predator"})},
        ],
    )

    result = validate_dataset(tmp_path, raw_path)

    assert result.passed_path.parent == tmp_path / "data" / "validated"
    assert result.failed_path.parent == tmp_path / "data" / "validated"
    passed_rows = read_jsonl(result.passed_path)
    assert len(passed_rows) == 2
    assert json.loads(passed_rows[1]["output"])["text_ko"] == "먹거리를 발견했다."
    failed_rows = read_jsonl(result.failed_path)
    assert len(failed_rows) == 0


def test_prepare_dataset_merges_validated_samples_and_writes_manifest(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)
    passed_path = tmp_path / "data" / "validated" / "passed.jsonl"
    negative_path = tmp_path / "data" / "samples" / "negative_examples.jsonl"
    general_path = tmp_path / "data" / "samples" / "general_korean.jsonl"

    write_jsonl(passed_path, [{"task": "A", "layer": "L4", "prompt": "[TASK] A", "output": compact_json({"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness", "temperament_expressed": "mixed"})}])
    write_jsonl(negative_path, [{"task": "NEG", "output": "이 사람은 이다. 이 사람은 이다."}])
    write_jsonl(general_path, [{"task": "GEN", "output": "바람이 강물 위를 스쳐 간다."}])

    result = prepare_dataset(tmp_path)

    assert result.dataset_path.parent == tmp_path / "data/final"
    assert result.manifest_path.parent == tmp_path / "artifacts/manifests"
    dataset_rows = read_jsonl(result.dataset_path)
    assert len(dataset_rows) == 3
    assert json.loads(dataset_rows[0]["messages"][2]["content"])["dominant_trait"] == "conscientiousness"
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"validated": 1, "negative": 1, "general": 1, "total": 3}
