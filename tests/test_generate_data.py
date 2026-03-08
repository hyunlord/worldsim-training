from pathlib import Path

import yaml

from scripts.generate_data import build_jobs, load_catalogs, load_generation_config, load_prompt_assets, render_prompt


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

