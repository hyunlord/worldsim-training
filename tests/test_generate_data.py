import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_data import (
    _resolve_cli_output_path,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def bootstrap_bilingual_assets(tmp_path: Path) -> tuple[Path, Path]:
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
                    "dominant_trait": "conscientiousness",
                    "speaker_role": "elder",
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
            "task_variants": {"A": 1, "B": 2, "C": 1, "D": 1, "E": 2, "F": 1},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
            "provider": {
                "require_parameters": True,
                "pricing": {
                    "input_per_million_tokens_usd": 3.0,
                    "output_per_million_tokens_usd": 15.0,
                }
            },
            "validation": {
                "trait_axes": [
                    "honesty_humility",
                    "emotionality",
                    "extraversion",
                    "agreeableness",
                    "conscientiousness",
                    "openness",
                ],
                "reasoning_axes": [
                    "high_honesty_humility",
                    "high_emotionality",
                    "high_extraversion",
                    "high_agreeableness",
                    "high_conscientiousness",
                    "high_openness",
                ],
                "speaker_roles": ["elder", "hunter", "shaman", "warrior", "healer", "gatherer", "craftsman", "chief", "scout", "observer"],
                "transition_types": ["gradual", "sudden", "sustained"],
                "task_limits": {
                    "A": {"min_chars": 20, "max_chars": 40, "sentences": 1},
                    "B": {"min_chars": 30, "max_chars": 60, "sentences": 2},
                    "C": {"min_chars": 15, "max_chars": 30, "sentences": 1},
                    "D": {"min_chars": 10, "max_chars": 25, "sentences": 1},
                    "E": {"min_chars": 10, "max_chars": 30, "sentences": 1},
                    "F": {"min_chars": 10, "max_chars": 25, "sentences": 1},
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
            "reporting": {"progress_every": 1},
        },
    )

    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_a.txt").write_text(
        '[TASK] A\n[PERS] {personality_keywords}\n[TRAIT] {dominant_trait}\n'
        '{"text_ko":"...", "text_en":"...", "register":"haera", "dominant_trait":"{dominant_trait}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_b.txt").write_text(
        '[TASK] B\n[PERS] {personality_keywords}\n[EMOT] {emotion_name}:{emotion_intensity}\n'
        '[MIMETIC] {mimetic}\n[SITU] {scenario_desc}\n'
        '{"text_ko":"...", "text_en":"...", "register":"haera", "emotion_expressed":"{emotion_id}", "intensity":0.9, "mimetics":["{mimetic}"]}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_c.txt").write_text(
        '[TASK] C\n[ROLE] {speaker_role}\n[REG] {register}\n'
        '{"speech_ko":"...", "speech_en":"...", "register":"{register}", "emotion_expressed":"{emotion_id}", "speaker_role":"{speaker_role}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_d.txt").write_text(
        '[TASK] D\n[NAME] {name}\n[SITU] {scenario_desc}\n'
        '{"text_ko":"...", "text_en":"...", "event_type":"{situation_id}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_e.txt").write_text(
        '[TASK] E\n[PERS] {personality_keywords}\n[OPTIONS] {options_line}\n'
        '{"action_id":0, "confidence":0.9, "hint_ko":"...", "hint_en":"...", "personality_reasoning":"{personality_reasoning}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_f.txt").write_text(
        '[TASK] F\n[PERS] {personality_keywords}\n[CURRENT_EMOT] {current_emotion_id}\n'
        '{"emotion":"fear", "intensity":0.9, "cause_ko":"...", "cause_en":"...", "previous_emotion":"{current_emotion_id}", "transition_type":"sudden"}',
        encoding="utf-8",
    )
    return config_dir, prompts_dir


def test_generate_data_builds_jobs_from_config_and_prompt_assets(tmp_path: Path) -> None:
    config_dir, prompts_dir = bootstrap_bilingual_assets(tmp_path)

    settings = load_generation_config(config_dir)
    catalogs = load_catalogs(config_dir)
    prompt_assets = load_prompt_assets(prompts_dir)

    jobs = build_jobs(catalogs, settings)

    assert len(jobs) == 11
    assert {job["task"] for job in jobs} == {"A", "B", "C", "D", "E", "F"}
    assert all(job["expected_format"] == "json" for job in jobs)
    assert next(job for job in jobs if job["task"] == "A")["dominant_trait"] == "conscientiousness"
    assert next(job for job in jobs if job["task"] == "C")["speaker_role"] == "elder"

    task_b = next(job for job in jobs if job["task"] == "B")
    rendered = render_prompt(task_b, prompt_assets)

    assert "날랜 짐승이 가까이 나타났다" in rendered
    assert '"text_ko"' in rendered
    assert '"text_en"' in rendered


def test_generate_data_builds_bilingual_layer3_jobs_and_filters_requested_tasks(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)

    jobs = build_jobs(tmp_path, task_filter={"E", "F"})

    assert len(jobs) == 4
    assert [job["task"] for job in jobs].count("E") == 2
    assert [job["task"] for job in jobs].count("F") == 2
    assert jobs[0]["layer"] == "L3"
    assert jobs[0]["expected_format"] == "json"
    assert jobs[0]["personality_reasoning"] == "high_conscientiousness"
    assert "[OPTIONS] 0:도망 1:숨기 2:맞서기 3:경고 4:얼어붙기" in jobs[0]["prompt"]
    assert '"hint_ko"' in jobs[0]["prompt"]
    assert '"hint_en"' in jobs[0]["prompt"]


def test_build_response_format_uses_structured_json_constraints_for_l4_and_l3(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)
    jobs = build_jobs(tmp_path)
    settings = load_generation_config(tmp_path / "config")

    task_a = next(job for job in jobs if job["task"] == "A")
    response_format_a, _ = build_response_format(task_a, settings)
    task_f = next(job for job in jobs if job["task"] == "F")
    response_format_f, extra_body = build_response_format(task_f, settings)

    assert response_format_a["type"] == "json_schema"
    schema_a = response_format_a["json_schema"]["schema"]
    assert schema_a["required"] == ["text_ko", "text_en", "register", "dominant_trait"]
    assert schema_a["properties"]["dominant_trait"]["enum"] == ["conscientiousness"]
    assert schema_a["properties"]["register"]["enum"] == ["haera"]

    schema_f = response_format_f["json_schema"]["schema"]
    assert schema_f["required"] == ["emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type"]
    assert schema_f["properties"]["previous_emotion"]["enum"] == ["fear", "trust"]
    assert schema_f["properties"]["transition_type"]["enum"] == ["gradual", "sudden", "sustained"]
    assert extra_body["provider"]["require_parameters"] is True


def test_generate_dataset_prints_progress_and_final_usage_summary(tmp_path: Path, capsys) -> None:
    bootstrap_bilingual_assets(tmp_path)

    def fake_generator(job: dict, system_prompt: str) -> dict:
        assert system_prompt == "teacher system"
        payloads = {
            "A": {"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness"},
            "B": {"text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.", "text_en": "The bushes shook hard. Trembling all over, they backed away.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.9, "mimetics": ["오들오들"]},
        }
        return {
            "output": compact_json(payloads[job["task"]]),
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
    assert json.loads(rows[0]["output"])["text_en"]
    assert "[1/2]" in captured
    assert "Generation summary" in captured
    assert "estimated_cost_usd" in captured


def test_generate_dataset_balances_limit_across_requested_tasks(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)

    def fake_generator(job: dict, system_prompt: str) -> dict:
        payload = (
            {"action_id": 0, "confidence": 0.9, "hint_ko": "겁이 치밀어 곧바로 달아났다", "hint_en": "Fear surged, so they fled at once.", "personality_reasoning": "high_conscientiousness"}
            if job["task"] == "E"
            else {"emotion": "fear", "intensity": 0.9, "cause_ko": "날랜 짐승이 바로 눈앞에 덮쳤다", "cause_en": "A fierce beast lunged right in front of them.", "previous_emotion": job["current_emotion_id"], "transition_type": "sudden"}
        )
        return {
            "output": compact_json(payload),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    result = generate_dataset(tmp_path, generator=fake_generator, limit=2, task_filter={"E", "F"}, verbose=False)
    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert [row["task"] for row in rows] == ["E", "F"]


def test_generate_dataset_retries_transient_failures_and_checkpoints_completed_rows(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")
    settings["task_variants"] = {"A": 2, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
    settings["provider"]["retry_attempts"] = 2
    settings["provider"]["retry_backoff_seconds"] = 0
    write_yaml(tmp_path / "config" / "generation.yaml", settings)

    attempts = {"count": 0}

    def flaky_generator(job: dict, system_prompt: str) -> dict:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient")
        if job["variant"] == 1:
            raise RuntimeError("persistent")
        return {
            "output": compact_json(
                {
                    "text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.",
                    "text_en": "Fearless and always sees things through.",
                    "register": "haera",
                    "dominant_trait": "conscientiousness",
                }
            ),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    output_path = tmp_path / "data" / "raw" / "checkpoint.jsonl"
    with pytest.raises(RuntimeError, match="persistent"):
        generate_dataset(tmp_path, generator=flaky_generator, limit=2, output_path=output_path, verbose=False)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert attempts["count"] == 4
    assert len(rows) == 1
    assert json.loads(rows[0]["output"])["dominant_trait"] == "conscientiousness"


def test_render_prompt_preserves_literal_placeholder_tokens_inside_values() -> None:
    prompt_assets = {"tasks": {"A": "[DESC] {personality_desc}\n[REG] {register}"}}
    job = {"task": "A", "personality_desc": "곧은 마음 {register} 그대로", "register": "haera"}

    rendered = render_prompt(job, prompt_assets)

    assert rendered == "[DESC] 곧은 마음 {register} 그대로\n[REG] haera"


def test_resolve_cli_output_path_rejects_paths_outside_raw_dir(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")

    with pytest.raises(ValueError, match="raw_dir"):
        _resolve_cli_output_path(tmp_path, settings, tmp_path / "escape.jsonl")


def test_build_output_path_is_unique_within_same_second(tmp_path: Path) -> None:
    first = build_output_path(tmp_path / "data" / "raw")
    second = build_output_path(tmp_path / "data" / "raw")

    assert first != second
