from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.curriculum_order_v3 import curriculum_order_v3
from scripts.generate_data import (
    batch_task_counts,
    build_jobs,
    build_response_format,
    load_generation_config,
    load_prompt_assets,
)
from scripts.validate_data import repair_and_validate_json_output


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_v3_prompt_loader_prefers_v3_file(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    teacher_dir = prompts_dir / "teacher"
    teacher_dir.mkdir(parents=True)
    (teacher_dir / "system.txt").write_text("system", encoding="utf-8")
    (teacher_dir / "task_e.txt").write_text("v2-e", encoding="utf-8")
    (teacher_dir / "task_e_v3.txt").write_text("v3-e", encoding="utf-8")
    (teacher_dir / "task_o.txt").write_text("task-o", encoding="utf-8")

    assets_v2 = load_prompt_assets(prompts_dir, schema_version=2)
    assets_v3 = load_prompt_assets(prompts_dir, schema_version=3)

    assert assets_v2["tasks"]["E"] == "v2-e"
    assert assets_v3["tasks"]["E"] == "v3-e"
    assert assets_v3["tasks"]["O"] == "task-o"


def test_build_jobs_generates_new_tasks_o_through_t() -> None:
    repo_root = _repo_root()
    jobs = build_jobs(repo_root, task_filter={"O", "P", "Q", "R", "S", "T"}, schema_version=3)
    counts = {}
    for job in jobs:
        counts[job["task"]] = counts.get(job["task"], 0) + 1

    for task in "OPQRST":
        assert counts.get(task, 0) > 0

    task_o = next(job for job in jobs if job["task"] == "O")
    task_t = next(job for job in jobs if job["task"] == "T")
    assert task_o["true_state"]
    assert task_o["public_goal"]
    assert task_t["faction_hint"]
    assert task_t["action_options"]


def test_v3_response_format_uses_english_logic_fields() -> None:
    repo_root = _repo_root()
    settings = load_generation_config(repo_root / "config")
    task_e = next(job for job in build_jobs(repo_root, task_filter={"E"}, schema_version=3) if job["task"] == "E")
    task_o = next(job for job in build_jobs(repo_root, task_filter={"O"}, schema_version=3) if job["task"] == "O")
    task_t = next(job for job in build_jobs(repo_root, task_filter={"T"}, schema_version=3) if job["task"] == "T")

    response_e, _ = build_response_format(task_e, settings, schema_version=3)
    response_o, _ = build_response_format(task_o, settings, schema_version=3)
    response_t, _ = build_response_format(task_t, settings, schema_version=3)

    schema_e = response_e["json_schema"]["schema"]
    assert schema_e["required"] == ["action_id", "confidence", "hint", "personality_reasoning", "temperament_factor"]
    assert "hint_ko" not in schema_e["properties"]
    assert "hint" in schema_e["properties"]

    schema_o = response_o["json_schema"]["schema"]
    assert schema_o["required"] == ["public_claim", "private_truth", "deception_style", "lie_degree", "detection_risk", "confidence"]

    schema_t = response_t["json_schema"]["schema"]
    assert schema_t["required"] == [
        "decision_id",
        "confidence",
        "dissent_risk",
        "minority_position",
        "minority_action",
        "spark_event",
        "reasoning",
        "timeline",
    ]


def test_batch_v3_configs_parse_and_expose_schema_version() -> None:
    repo_root = _repo_root()
    for batch_name in ("batch_v3_01_english_logic", "batch_v3_02_new_tasks"):
        path = repo_root / "config" / "batches" / f"{batch_name}.yaml"
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 3
        assert batch_task_counts(payload)


def test_validate_data_accepts_v3_new_task_record() -> None:
    repo_root = _repo_root()
    rules = load_generation_config(repo_root / "config")["validation"]
    record = {
        "task": "O",
        "schema_version": 3,
        "output": json.dumps(
            {
                "public_claim": "I found nothing beyond the ridge.",
                "private_truth": "I hid the berry grove for myself.",
                "deception_style": "omission",
                "lie_degree": 0.62,
                "detection_risk": 0.48,
                "confidence": 0.71,
            },
            ensure_ascii=False,
        ),
    }

    repaired_output, violations, repair_count = repair_and_validate_json_output(record, rules)
    parsed = json.loads(repaired_output)
    assert repair_count == 0
    assert violations == []
    assert parsed["deception_style"] == "omission"


def test_curriculum_order_v3_prioritizes_early_stage_tasks() -> None:
    rows = [
        {"task": "T", "id": 1},
        {"task": "E", "id": 2},
        {"task": "NEG", "id": 3},
        {"task": "K", "id": 4},
        {"task": "A", "id": 5},
    ]

    ordered = curriculum_order_v3(rows, seed=7)
    ordered_tasks = [row["task"] for row in ordered]

    assert ordered_tasks.index("E") < ordered_tasks.index("K")
    assert ordered_tasks.index("K") < ordered_tasks.index("A")
    assert ordered_tasks.index("A") < ordered_tasks.index("T")
