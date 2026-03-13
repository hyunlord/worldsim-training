from __future__ import annotations

from collections import Counter
from pathlib import Path

from scripts.generate_data import build_jobs, build_response_format, load_catalogs, load_generation_config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_catalogs_includes_new_task_inputs() -> None:
    config_dir = _repo_root() / "config"
    catalogs = load_catalogs(config_dir)

    assert len(catalogs["needs_situations"]) >= 8
    assert len(catalogs["stress_situations"]) >= 8
    assert len(catalogs["social_situations"]) >= 8
    assert len(catalogs["group_situations"]) >= 8
    assert len(catalogs["trade_scenarios"]) >= 8


def test_build_jobs_creates_examples_for_tasks_i_through_n() -> None:
    repo_root = _repo_root()
    jobs = build_jobs(repo_root, task_filter={"I", "J", "K", "L", "M", "N"})
    counts = Counter(job["task"] for job in jobs)

    for task in ("I", "J", "K", "L", "M", "N"):
        assert counts[task] > 0

    task_i = next(job for job in jobs if job["task"] == "I")
    task_j = next(job for job in jobs if job["task"] == "J")
    task_k = next(job for job in jobs if job["task"] == "K")
    task_l = next(job for job in jobs if job["task"] == "L")
    task_m = next(job for job in jobs if job["task"] == "M")
    task_n = next(job for job in jobs if job["task"] == "N")

    assert "{needs_state}" not in task_i["prompt"]
    assert "{action_options}" not in task_i["prompt"]
    assert task_j["scenario_desc"]
    assert task_k["relationship_context"]
    assert task_l["social_memory"]
    assert task_m["group_context"]
    assert task_n["trade_offer"]


def test_build_response_format_supports_tasks_i_through_n() -> None:
    repo_root = _repo_root()
    settings = load_generation_config(repo_root / "config")
    jobs = build_jobs(repo_root, task_filter={"I", "J", "K", "L", "M", "N"})

    required_fields = {
        "I": ["priority_id", "reasoning_ko", "reasoning_en", "need_addressed", "urgency"],
        "J": ["coping_id", "coping_type", "stress_delta", "hint_ko", "hint_en", "side_effect"],
        "K": ["social_action_id", "trust_delta", "hint_ko", "hint_en", "relationship_intent", "reciprocity_expectation"],
        "L": ["response_id", "trust_delta", "hint_ko", "hint_en", "forgiveness_threshold", "social_memory"],
        "M": ["decision_id", "confidence", "dissent_risk", "reasoning_ko", "reasoning_en", "resource_commitment", "timeline"],
        "N": ["accept", "counter_offer_give", "counter_offer_want", "hint_ko", "hint_en", "negotiation_stance", "walk_away_threshold"],
    }

    for task, expected in required_fields.items():
        job = next(candidate for candidate in jobs if candidate["task"] == task)
        response_format, _ = build_response_format(job, settings)
        assert response_format is not None
        assert response_format["type"] == "json_schema"
        schema = response_format["json_schema"]["schema"]
        assert schema["required"] == expected
