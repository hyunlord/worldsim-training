import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_data import (
    build_jobs,
    build_response_format,
    build_output_path,
    generate_dataset,
    load_catalogs,
    load_generation_config,
    load_prompt_assets,
    render_prompt,
)


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_generate_data_builds_jobs_from_config_and_prompt_assets(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()

    write_yaml(
        config_dir / "situations.yaml",
        {
            "situations": [
                {"id": "predator", "ko": "짐승발견", "desc": "날랜 짐승이 가까이 나타났다", "typical_actions": ["도망"]},
            ]
        },
    )
    write_yaml(
        config_dir / "personalities.yaml",
        {
            "personalities": [
                {
                    "id": "cautious_elder",
                    "ko": "신중한원로",
                    "keywords": ["겁많음", "꼼꼼함"],
                    "desc": "위험을 먼저 살핀다",
                    "default_register": "hao",
                }
            ]
        },
    )
    write_yaml(
        config_dir / "emotions.yaml",
        {
            "emotions": [
                {"id": "fear", "ko": "두려움", "intensities": [0.9], "mimetics": ["오들오들"]},
            ]
        },
    )
    write_yaml(
        config_dir / "generation.yaml",
        {
            "seed": 7,
            "task_variants": {"A": 1, "B": 2, "C": 1, "D": 1},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
        },
    )

    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_a.txt").write_text(
        "[성격]\n{personality_desc}\n[키워드]\n{personality_keywords}\n[변형]\n{variant}",
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_b.txt").write_text(
        "[상황]\n{scenario_desc}\n[감정]\n{emotion_name}:{emotion_intensity}\n[의성어]\n{mimetic}",
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_c.txt").write_text(
        "[말투]\n{register_instruction}\n[상황]\n{scenario_desc}",
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_d.txt").write_text("[누가]\n{name}\n[상황]\n{scenario_desc}", encoding="utf-8")

    settings = load_generation_config(config_dir)
    catalogs = load_catalogs(config_dir)
    prompt_assets = load_prompt_assets(prompts_dir)

    jobs = build_jobs(catalogs, settings)

    assert len(jobs) == 5
    assert {job["task"] for job in jobs} == {"A", "B", "C", "D"}

    task_b = next(job for job in jobs if job["task"] == "B")
    rendered = render_prompt(task_b, prompt_assets)

    assert "날랜 짐승이 가까이 나타났다" in rendered
    assert "오들오들" in rendered
    assert settings["paths"]["raw_dir"] == "data/raw"


def test_generate_data_builds_layer3_jobs_and_filters_requested_tasks(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()

    write_yaml(
        config_dir / "situations.yaml",
        {
            "situations": [
                {
                    "id": "predator",
                    "ko": "짐승발견",
                    "desc": "날랜 짐승이 가까이 나타났다",
                    "action_options": ["도망", "숨기", "맞서기", "경고", "얼어붙기"],
                },
            ]
        },
    )
    write_yaml(
        config_dir / "personalities.yaml",
        {
            "personalities": [
                {
                    "id": "cautious_elder",
                    "ko": "신중한원로",
                    "keywords": ["겁많음", "꼼꼼함"],
                    "desc": "위험을 먼저 살핀다",
                    "default_register": "hao",
                }
            ]
        },
    )
    write_yaml(
        config_dir / "emotions.yaml",
        {
            "emotions": [
                {"id": "fear", "ko": "두려움", "intensities": [0.9], "mimetics": ["오들오들"]},
                {"id": "trust", "ko": "믿음", "intensities": [0.4], "mimetics": ["든든히"]},
            ]
        },
    )
    write_yaml(
        config_dir / "generation.yaml",
        {
            "seed": 7,
            "task_variants": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 2, "F": 1},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
        },
    )

    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_e.txt").write_text(
        "[TASK] E\n[PERS] {personality_keywords}\n[EMOT] {emotion_name}:{emotion_intensity}\n"
        "[SITU] {scenario_name}\n[OPTIONS] {options_line}",
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_f.txt").write_text(
        "[TASK] F\n[PERS] {personality_keywords}\n[CURRENT_EMOT] {current_emotion_name}:{current_emotion_intensity}\n"
        "[SITU] {scenario_name}",
        encoding="utf-8",
    )

    jobs = build_jobs(tmp_path, task_filter={"E", "F"})

    assert len(jobs) == 4
    assert [job["task"] for job in jobs].count("E") == 2
    assert [job["task"] for job in jobs].count("F") == 2
    assert jobs[0]["layer"] == "L3"
    assert jobs[0]["expected_format"] == "json"
    assert jobs[0]["action_options"] == ["도망", "숨기", "맞서기", "경고", "얼어붙기"]
    assert "[OPTIONS] 0:도망 1:숨기 2:맞서기 3:경고 4:얼어붙기" in jobs[0]["prompt"]
    assert '"action_id"' not in jobs[0]["prompt"]


def test_generate_dataset_prints_progress_and_final_usage_summary(tmp_path: Path, capsys) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()

    write_yaml(
        config_dir / "situations.yaml",
        {
            "situations": [
                {"id": "predator", "ko": "짐승발견", "desc": "날랜 짐승이 가까이 나타났다"},
            ]
        },
    )
    write_yaml(
        config_dir / "personalities.yaml",
        {
            "personalities": [
                {
                    "id": "cautious_elder",
                    "ko": "신중한원로",
                    "keywords": ["겁많음", "꼼꼼함"],
                    "desc": "위험을 먼저 살핀다",
                    "default_register": "hao",
                }
            ]
        },
    )
    write_yaml(
        config_dir / "emotions.yaml",
        {
            "emotions": [
                {"id": "fear", "ko": "두려움", "intensities": [0.9], "mimetics": ["오들오들"]},
            ]
        },
    )
    write_yaml(
        config_dir / "generation.yaml",
        {
            "seed": 7,
            "task_variants": {"A": 1, "B": 1, "C": 0, "D": 0},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
            "provider": {
                "pricing": {
                    "input_per_million_tokens_usd": 3.0,
                    "output_per_million_tokens_usd": 15.0,
                }
            },
            "reporting": {"progress_every": 1},
        },
    )

    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_a.txt").write_text("[성격]\n{personality_desc}", encoding="utf-8")
    (prompts_dir / "teacher" / "task_b.txt").write_text("[상황]\n{scenario_desc}", encoding="utf-8")

    def fake_generator(job: dict, system_prompt: str) -> dict:
        assert system_prompt == "teacher system"
        return {
            "output": f"{job['task']} output",
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
            "model": "anthropic/claude-sonnet-4-20250514",
        }

    result = generate_dataset(tmp_path, generator=fake_generator, limit=2)

    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    captured = capsys.readouterr().out

    assert result.prompt_tokens == 200
    assert result.completion_tokens == 50
    assert result.total_tokens == 250
    assert result.estimated_cost_usd == pytest.approx(0.00135)
    assert result.rows_per_second > 0
    assert result.tokens_per_second > 0
    assert rows[0]["prompt_tokens"] == 100
    assert rows[0]["completion_tokens"] == 25
    assert rows[0]["estimated_cost_usd"] == pytest.approx(0.000675)
    assert "[1/2]" in captured
    assert "Generation summary" in captured
    assert "estimated_cost_usd" in captured


def test_build_response_format_uses_structured_json_constraints_for_layer3() -> None:
    job = {
        "task": "E",
        "action_options": ["도망", "숨기", "맞서기"],
    }
    settings = {
        "provider": {"require_parameters": True},
        "validation": {
            "task_limits": {
                "E": {"min_chars": 10, "max_chars": 30},
                "F": {"min_chars": 10, "max_chars": 25},
            },
            "layer3_json": {
                "task_f": {
                    "valid_emotions": [
                        "joy",
                        "sadness",
                        "fear",
                        "anger",
                        "trust",
                        "disgust",
                        "surprise",
                        "anticipation",
                    ]
                }
            },
        },
    }

    response_format, extra_body = build_response_format(job, settings)

    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["action_id"]["enum"] == [0, 1, 2]
    assert schema["properties"]["hint"]["minLength"] == 10
    assert schema["properties"]["hint"]["maxLength"] == 30
    assert extra_body["provider"]["require_parameters"] is True


def test_generate_dataset_balances_limit_across_requested_tasks(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()

    write_yaml(
        config_dir / "situations.yaml",
        {
            "situations": [
                {
                    "id": "predator",
                    "ko": "짐승발견",
                    "desc": "날랜 짐승이 가까이 나타났다",
                    "action_options": ["도망", "숨기", "맞서기", "경고", "얼어붙기"],
                },
            ]
        },
    )
    write_yaml(
        config_dir / "personalities.yaml",
        {
            "personalities": [
                {
                    "id": "cautious_elder",
                    "ko": "신중한원로",
                    "keywords": ["겁많음", "꼼꼼함"],
                    "desc": "위험을 먼저 살핀다",
                }
            ]
        },
    )
    write_yaml(
        config_dir / "emotions.yaml",
        {
            "emotions": [
                {"id": "fear", "ko": "두려움", "intensities": [0.9]},
                {"id": "trust", "ko": "믿음", "intensities": [0.4]},
            ]
        },
    )
    write_yaml(
        config_dir / "generation.yaml",
        {
            "seed": 7,
            "task_variants": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 2, "F": 2},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
            "provider": {
                "pricing": {
                    "input_per_million_tokens_usd": 0.0,
                    "output_per_million_tokens_usd": 0.0,
                }
            },
        },
    )
    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_e.txt").write_text("[TASK] E\n[OPTIONS] {options_line}", encoding="utf-8")
    (prompts_dir / "teacher" / "task_f.txt").write_text("[TASK] F\n[CURRENT] {current_emotion_name}", encoding="utf-8")

    def fake_generator(job: dict, system_prompt: str) -> dict:
        return {
            "output": job["task"],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    result = generate_dataset(tmp_path, generator=fake_generator, limit=2, task_filter={"E", "F"}, verbose=False)
    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert [row["task"] for row in rows] == ["E", "F"]


def test_generate_dataset_retries_transient_failures_and_checkpoints_completed_rows(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()

    write_yaml(config_dir / "situations.yaml", {"situations": [{"id": "predator", "ko": "짐승발견", "desc": "날랜 짐승"}]})
    write_yaml(
        config_dir / "personalities.yaml",
        {"personalities": [{"id": "cautious_elder", "ko": "신중한원로", "keywords": ["겁많음"], "desc": "조심스럽다"}]},
    )
    write_yaml(config_dir / "emotions.yaml", {"emotions": [{"id": "fear", "ko": "두려움", "intensities": [0.9]}]})
    write_yaml(
        config_dir / "generation.yaml",
        {
            "task_variants": {"A": 2, "B": 0, "C": 0, "D": 0},
            "paths": {"raw_dir": "data/raw"},
            "provider": {
                "retry_attempts": 2,
                "retry_backoff_seconds": 0,
                "pricing": {
                    "input_per_million_tokens_usd": 0.0,
                    "output_per_million_tokens_usd": 0.0,
                },
            },
        },
    )
    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_a.txt").write_text("[TASK] A\n[PERS] {personality_keywords}", encoding="utf-8")

    attempts = {"count": 0}

    def flaky_generator(job: dict, system_prompt: str) -> dict:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        if job["variant"] == 1:
            raise RuntimeError("persistent")
        return {
            "output": "살금살금 둘레를 살폈다.",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    output_path = tmp_path / "data" / "raw" / "checkpoint.jsonl"
    with pytest.raises(RuntimeError, match="persistent"):
        generate_dataset(tmp_path, generator=flaky_generator, limit=2, output_path=output_path, verbose=False)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert attempts["count"] == 4
    assert len(rows) == 1
    assert rows[0]["task"] == "A"


def test_build_output_path_is_unique_within_same_second(tmp_path: Path) -> None:
    first = build_output_path(tmp_path / "data" / "raw")
    second = build_output_path(tmp_path / "data" / "raw")

    assert first != second
