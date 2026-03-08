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


def write_bootstrap_assets(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "teacher").mkdir(parents=True, exist_ok=True)
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
generation:
  variants:
    task_a: 1
    task_b: 1
    task_c: 1
    task_d: 1
defaults:
  register: haera
  validation:
    forbidden_words:
      - 식량
      - 마을
      - 전투
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


def test_generate_data_loads_repo_assets_and_writes_to_raw(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)

    def fake_generator(job: dict, system_prompt: str) -> str:
        assert "석기시대" in system_prompt
        return f"{job['task']}:{job.get('situation_id', 'none')}"

    result = generate_dataset(tmp_path, generator=fake_generator, limit=3)

    assert result.output_path.parent == tmp_path / "data" / "raw"
    assert result.output_path.exists()
    rows = read_jsonl(result.output_path)
    assert len(rows) == 3
    assert {row["task"] for row in rows} <= {"A", "B", "C", "D"}


def test_validate_data_reads_raw_and_writes_pass_fail_files(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)
    raw_path = tmp_path / "data" / "raw" / "batch.jsonl"
    write_jsonl(
        raw_path,
        [
            {"task": "D", "register": "haera", "output": "돌이가 강가에서 물고기를 잡았다."},
            {"task": "D", "register": "haera", "output": "식량을 발견했다."},
        ],
    )

    result = validate_dataset(tmp_path, raw_path)

    assert result.passed_path.parent == tmp_path / "data" / "validated"
    assert result.failed_path.parent == tmp_path / "data" / "validated"
    passed_rows = read_jsonl(result.passed_path)
    assert len(passed_rows) == 2
    assert passed_rows[1]["output"] == "먹거리를 발견했다."
    failed_rows = read_jsonl(result.failed_path)
    assert len(failed_rows) == 0


def test_prepare_dataset_merges_validated_samples_and_writes_manifest(tmp_path: Path) -> None:
    write_bootstrap_assets(tmp_path)
    passed_path = tmp_path / "data" / "validated" / "passed.jsonl"
    negative_path = tmp_path / "data" / "samples" / "negative_examples.jsonl"
    general_path = tmp_path / "data" / "samples" / "general_korean.jsonl"

    write_jsonl(passed_path, [{"task": "A", "output": "곧은 마음으로 끝을 보는 이다."}])
    write_jsonl(negative_path, [{"task": "NEG", "output": "이 사람은 이다. 이 사람은 이다."}])
    write_jsonl(general_path, [{"task": "GEN", "output": "바람이 강물 위를 스쳐 간다."}])

    result = prepare_dataset(tmp_path)

    assert result.dataset_path.parent == tmp_path / "data" / "final"
    assert result.manifest_path.parent == tmp_path / "artifacts" / "manifests"
    dataset_rows = read_jsonl(result.dataset_path)
    assert len(dataset_rows) == 3
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"validated": 1, "negative": 1, "general": 1, "total": 3}
