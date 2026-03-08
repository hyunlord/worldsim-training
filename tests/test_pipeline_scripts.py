from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import yaml


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def bootstrap_repo(tmp_path: Path) -> Path:
    for rel in [
        "config",
        "prompts/teacher",
        "prompts/validation",
        "prompts/training",
        "data/raw",
        "data/validated",
        "data/final",
        "data/samples",
        "artifacts/manifests",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    (tmp_path / "config/situations.yaml").write_text(
        """
situations:
  - id: predator
    ko: 짐승발견
    desc: 날랜 짐승이 나타났다
    action_options: [도망, 숨기, 맞서기, 경고, 얼어붙기]
  - id: storm
    ko: 비바람
    desc: 거센 비바람이 몰아친다
    action_options: [피하기, 움막보강, 불지키기, 견디기, 도망]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "config/personalities.yaml").write_text(
        """
personalities:
  - id: cautious_elder
    ko: 신중한원로
    keywords: [겁많음, 꼼꼼함]
    desc: 위험을 경계하는 어른
    default_register: hao
    dominant_trait: conscientiousness
    speaker_role: elder
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "config/emotions.yaml").write_text(
        """
emotions:
  - id: fear
    ko: 두려움
    intensities: [0.3, 0.9]
    mimetics: [오들오들]
  - id: joy
    ko: 기쁨
    intensities: [0.6]
    mimetics: [방긋]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "config/generation.yaml").write_text(
        """
paths:
  raw_dir: data/raw
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
prompts:
  teacher:
    system: prompts/teacher/system.txt
    tasks:
      A: prompts/teacher/task_a.txt
      B: prompts/teacher/task_b.txt
      C: prompts/teacher/task_c.txt
      D: prompts/teacher/task_d.txt
      E: prompts/teacher/task_e.txt
      F: prompts/teacher/task_f.txt
  training:
    layer3_system: prompts/training/layer3_system.txt
    layer4_system: prompts/training/layer4_system.txt
task_variants:
  A: 2
  B: 1
  C: 1
  D: 1
  E: 1
  F: 1
validation:
  forbidden_words: [식량, 전투]
  meta_patterns: [WorldSim]
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
    E: {min_chars: 10, max_chars: 30, sentences: 1}
    F: {min_chars: 10, max_chars: 25, sentences: 1}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/system.txt").write_text(
        "JSON only bilingual output.\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_a.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "register":"haera", "dominant_trait":"{dominant_trait}"}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_b.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "register":"haera", "emotion_expressed":"{emotion_id}", "intensity":0.9, "mimetics":["{mimetic}"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_c.txt").write_text(
        '{"speech_ko":"...", "speech_en":"...", "register":"{register}", "emotion_expressed":"{emotion_id}", "speaker_role":"{speaker_role}"}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_d.txt").write_text(
        '{"text_ko":"...", "text_en":"...", "event_type":"{situation_id}"}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_e.txt").write_text(
        '{"action_id":0, "confidence":0.9, "hint_ko":"...", "hint_en":"...", "personality_reasoning":"{personality_reasoning}"}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/teacher/task_f.txt").write_text(
        '{"emotion":"fear", "intensity":0.9, "cause_ko":"...", "cause_en":"...", "previous_emotion":"{current_emotion_id}", "transition_type":"sudden"}\n',
        encoding="utf-8",
    )
    (tmp_path / "prompts/training/layer3_system.txt").write_text("JSON only bilingual output.", encoding="utf-8")
    (tmp_path / "prompts/training/layer4_system.txt").write_text("JSON only bilingual output.", encoding="utf-8")
    write_jsonl(tmp_path / "data/samples/negative_examples.jsonl", [{"task": "NEG", "output": "이 사람은 이다."}])
    write_jsonl(tmp_path / "data/samples/general_korean.jsonl", [{"task": "GEN", "output": "강가에 안개가 내렸다."}])
    return tmp_path


def test_generate_data_builds_jobs_from_config_and_prompts(tmp_path: Path) -> None:
    repo_root = bootstrap_repo(tmp_path)
    module = load_module("generate_data", Path.cwd() / "scripts/generate_data.py")

    jobs = module.build_jobs(repo_root, seed=7)
    counts = {}
    for job in jobs:
        counts[job["task"]] = counts.get(job["task"], 0) + 1

    assert counts == {"A": 2, "B": 4, "C": 2, "D": 2, "E": 2, "F": 4}
    assert jobs[0]["system_prompt"] == "JSON only bilingual output."
    assert jobs[0]["expected_format"] == "json"


def test_validate_data_splits_passed_and_failed_records(tmp_path: Path) -> None:
    repo_root = bootstrap_repo(tmp_path)
    raw_path = repo_root / "data/raw/generated_test.jsonl"
    write_jsonl(
        raw_path,
        [
            {
                "task": "A",
                "register": "haera",
                "dominant_trait": "conscientiousness",
                "output": compact_json({"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness"}),
            },
            {
                "task": "D",
                "situation_id": "predator",
                "output": compact_json({"text_ko": "돌이가 먹거리를 찾았다.", "text_en": "Dol-i found food.", "event_type": "predator"}),
            },
        ],
    )

    module = load_module("validate_data", Path.cwd() / "scripts/validate_data.py")
    summary = module.validate_file(repo_root=repo_root, input_path=raw_path)

    passed = read_jsonl(repo_root / "data/validated/passed.jsonl")
    failed = read_jsonl(repo_root / "data/validated/failed.jsonl")

    assert summary["passed"] == 2
    assert summary["failed"] == 0
    assert json.loads(passed[0]["output"])["text_en"] == "Fearless and always sees things through."
    assert failed == []


def test_prepare_dataset_combines_validated_and_sample_streams(tmp_path: Path) -> None:
    repo_root = bootstrap_repo(tmp_path)
    write_jsonl(
        repo_root / "data/validated/passed.jsonl",
        [{"task": "A", "layer": "L4", "prompt": "[TASK] A", "output": compact_json({"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness"})}],
    )

    module = load_module("prepare_dataset", Path.cwd() / "scripts/prepare_dataset.py")
    result = module.prepare_dataset(repo_root=repo_root, dataset_name="worldsim-v0")

    dataset_rows = read_jsonl(result["dataset_path"])
    manifest = yaml.safe_load(result["manifest_path"].read_text(encoding="utf-8"))

    assert len(dataset_rows) == 3
    assert json.loads(dataset_rows[0]["messages"][2]["content"])["dominant_trait"] == "conscientiousness"
    assert result["dataset_path"].parent == repo_root / "data/final"
    assert result["manifest_path"].parent == repo_root / "artifacts/manifests"
    assert manifest["dataset_name"] == "worldsim-v0"
    assert manifest["counts"] == {"validated": 1, "negative": 1, "general": 1, "total": 3}


def test_prepare_dataset_respects_dataset_mix_flags(tmp_path: Path) -> None:
    repo_root = bootstrap_repo(tmp_path)
    config_path = repo_root / "config/generation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset_mix"] = {
        "include_negative_samples": False,
        "include_general_samples": True,
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_jsonl(
        repo_root / "data/validated/passed.jsonl",
        [{"task": "A", "layer": "L4", "prompt": "[TASK] A", "output": compact_json({"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness"})}],
    )

    module = load_module("prepare_dataset", Path.cwd() / "scripts/prepare_dataset.py")
    result = module.prepare_dataset(repo_root=repo_root, dataset_name="worldsim-v1")

    dataset_rows = read_jsonl(result["dataset_path"])
    manifest = yaml.safe_load(result["manifest_path"].read_text(encoding="utf-8"))

    assert len(dataset_rows) == 2
    assert [row["source_split"] for row in dataset_rows] == ["validated", "general"]
    assert manifest["counts"] == {"validated": 1, "negative": 0, "general": 1, "total": 2}


def test_load_local_env_reads_repo_env_file(tmp_path: Path, monkeypatch) -> None:
    repo_root = bootstrap_repo(tmp_path)
    (repo_root / ".env").write_text("OPENROUTER_API_KEY=sk-test-value\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    module = load_module("generate_data", Path.cwd() / "scripts/generate_data.py")
    module.load_local_env(repo_root)

    assert os.environ["OPENROUTER_API_KEY"] == "sk-test-value"
