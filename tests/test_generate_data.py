import json
from collections import Counter
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
    parse_and_validate,
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


def bootstrap_v31_assets(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts"
    grammars_dir = tmp_path / "grammars"
    (prompts_dir / "teacher").mkdir(parents=True)
    config_dir.mkdir()
    grammars_dir.mkdir()

    write_yaml(
        config_dir / "situations.yaml",
        {
            "situations": [
                {
                    "id": "predator",
                    "ko": "짐승발견",
                    "desc": "날랜 짐승이 무리 곁에 나타났다",
                    "action_options": ["도망", "숨기", "맞서기", "경고", "얼어붙기"],
                }
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
                    "personality_reasoning": "high_conscientiousness",
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
            "seed": 11,
            "task_variants": {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1, "H": 1},
            "paths": {"raw_dir": "data/raw"},
            "names": ["돌이"],
            "provider": {
                "require_parameters": False,
                "pricing": {
                    "input_per_million_tokens_usd": 3.0,
                    "output_per_million_tokens_usd": 15.0,
                },
            },
            "prompts": {
                "teacher": {
                    "system": "prompts/teacher/system.txt",
                    "tasks": {
                        "A": "prompts/teacher/task_a.txt",
                        "B": "prompts/teacher/task_b.txt",
                        "C": "prompts/teacher/task_c.txt",
                        "D": "prompts/teacher/task_d.txt",
                        "E": "prompts/teacher/task_e.txt",
                        "F": "prompts/teacher/task_f.txt",
                        "G": "prompts/teacher/task_g.txt",
                        "H": "prompts/teacher/task_h.txt",
                    },
                },
                "grammars": {
                    "task_e_action": "grammars/task_e_action.gbnf",
                    "task_f_emotion": "grammars/task_f_emotion.gbnf",
                    "task_g_oracle": "grammars/task_g_oracle.gbnf",
                    "task_h_worldruleset": "grammars/task_h_worldruleset.gbnf",
                },
            },
            "temperaments": [
                {
                    "id": "choleric",
                    "ko": "담즙질",
                    "tci": {"NS": 0.8, "HA": 0.2, "RD": 0.5, "P": 0.7},
                    "keywords": ["당당함", "충동적"],
                    "bias": "action_oriented",
                },
                {
                    "id": "melancholic",
                    "ko": "우울질",
                    "tci": {"NS": 0.2, "HA": 0.8, "RD": 0.6, "P": 0.4},
                    "keywords": ["신중함", "비관적"],
                    "bias": "cautious_conservative",
                },
            ],
            "worlds": [
                {"id": "default", "ko": "기본세계", "desc": "석기시대 기본 생태계", "vocab_additions": []},
                {"id": "winter", "ko": "겨울세계", "desc": "눈보라와 불씨가 귀한 세계", "vocab_additions": ["눈보라", "불씨"]},
            ],
            "oracles": [
                {
                    "id": "oracle_01",
                    "text_ko": "북쪽 산 너머에 풍요가 있다",
                    "text_en": "Abundance lies beyond the northern mountain",
                    "ambiguity": "high",
                }
            ],
            "worldbuilding_texts": [
                {
                    "id": "wb_01",
                    "text": "이 세계는 마녀의 저주로 지상이 황폐해졌고, 미궁에서만 자원이 나오며, 마석이 화폐이다.",
                    "expected_world_type": "dungeon",
                }
            ],
            "tasks": {
                "H": {"teacher_model": "openai/gpt-4.1"},
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
                "temperament_ids": ["choleric", "melancholic"],
                "temperament_biases": ["action_oriented", "cautious_conservative"],
                "world_ids": ["default", "winter"],
                "oracle_action_tendencies": ["mobilize", "defend", "wait", "retreat", "celebrate", "mourn"],
                "oracle_misinterpretations": [
                    "overconfident_literal",
                    "cautious_reversal",
                    "optimistic_expansion",
                    "passive_deferral",
                    "symbolic_abstraction",
                ],
                "task_limits": {
                    "A": {"min_chars": 20, "max_chars": 40, "sentences": 1},
                    "B": {"min_chars": 30, "max_chars": 60, "sentences": 2},
                    "C": {"min_chars": 15, "max_chars": 30, "sentences": 1},
                    "D": {"min_chars": 10, "max_chars": 25, "sentences": 1},
                    "E": {"min_chars": 10, "max_chars": 30, "sentences": 1},
                    "F": {"min_chars": 10, "max_chars": 25, "sentences": 1},
                    "G": {"min_chars": 15, "max_chars": 40, "sentences": 1},
                },
            },
        },
    )

    (prompts_dir / "teacher" / "system.txt").write_text("teacher system", encoding="utf-8")
    (prompts_dir / "teacher" / "task_a.txt").write_text(
        '[TASK] A\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[PERS] {personality_keywords}\n'
        '{"text_ko":"...", "text_en":"...", "register":"haera", "dominant_trait":"{dominant_trait}", "temperament_expressed":"{temperament_id}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_b.txt").write_text(
        '[TASK] B\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[PERS] {personality_keywords}\n'
        '[RULE] emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation\n'
        '[RULE] register must be exactly one of: haera, hao, hae\n'
        '[ENUMS] joy, sadness, fear, anger, trust, disgust, surprise, anticipation\n'
        '[ENUMS] haera, hao, hae\n'
        '{"text_ko":"...", "text_en":"...", "register":"haera", "emotion_expressed":"{emotion_id}", "intensity":0.9, "mimetics":["{mimetic}"], "temperament_influence":"high_HA_amplified_fear"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_c.txt").write_text(
        '[TASK] C\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[ROLE] {speaker_role}\n'
        '[RULE] emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation\n'
        '[RULE] register must be exactly one of: haera, hao, hae\n'
        '[ENUMS] joy, sadness, fear, anger, trust, disgust, surprise, anticipation\n'
        '[ENUMS] haera, hao, hae\n'
        '{"speech_ko":"...", "speech_en":"...", "register":"{register}", "emotion_expressed":"{emotion_id}", "speaker_role":"{speaker_role}", "temperament_tone":"choleric_directness"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_d.txt").write_text(
        '[TASK] D\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[NAME] {name}\n'
        '{"text_ko":"...", "text_en":"...", "event_type":"{situation_id}"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_e.txt").write_text(
        '[TASK] E\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[OPTIONS] {options_line}\n'
        '{"action_id":0, "confidence":0.9, "hint_ko":"...", "hint_en":"...", "personality_reasoning":"{personality_reasoning}", "temperament_factor":"harm_avoidance_dominant"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_f.txt").write_text(
        '[TASK] F\n[TEMP] {temperament_line}\n[STRESS] {stress}\n[WORLD] {world_id}\n[CURRENT_EMOT] {current_emotion_id}\n'
        '{"emotion":"fear", "intensity":0.9, "cause_ko":"...", "cause_en":"...", "previous_emotion":"{current_emotion_id}", "transition_type":"sudden", "temperament_amplifier":"high_HA_intensifies_fear"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_g.txt").write_text(
        '[TASK] G\n[TEMP] {temperament_line}\n[PERS] {personality_keywords}\n[ORACLE] {oracle_text_ko}\n[WORLD] {world_id}\n'
        '[RULE] register must be exactly one of: haera, hao, hae\n'
        '[ENUMS] haera, hao, hae\n'
        '{"interpretation_ko":"...", "interpretation_en":"...", "action_tendency":"mobilize", "confidence":0.9, "register":"hao", "misinterpretation_type":"overconfident_literal", "temperament_bias":"choleric_action_oriented"}',
        encoding="utf-8",
    )
    (prompts_dir / "teacher" / "task_h.txt").write_text(
        '[TASK] H\n[WORLD] {world_id}\n[WORLDBUILDING] {worldbuilding_text}\n'
        '{"name":"DungeonEconomy", "description_en":"Cursed surface world with dungeon-based economy.", "resource_modifiers":[{"target":"dungeon_loot","multiplier":3.0}], "special_zones":[{"kind":"dungeon_node","spawn_count_min":3,"spawn_count_max":7}], "special_resources":[{"name":"magic_stone","tags":["currency","tradeable"]}], "agent_modifiers":[{"system":"temperament","trigger":"essence_equip","effect":"shift_random_axis"}]}',
        encoding="utf-8",
    )
    (grammars_dir / "task_e_action.gbnf").write_text("root ::= \"{}\"", encoding="utf-8")
    (grammars_dir / "task_f_emotion.gbnf").write_text("root ::= \"{}\"", encoding="utf-8")
    (grammars_dir / "task_g_oracle.gbnf").write_text("root ::= \"{}\"", encoding="utf-8")
    (grammars_dir / "task_h_worldruleset.gbnf").write_text("root ::= \"{}\"", encoding="utf-8")
    return config_dir, prompts_dir


def test_generate_data_builds_jobs_from_config_and_prompt_assets(tmp_path: Path) -> None:
    config_dir, prompts_dir = bootstrap_bilingual_assets(tmp_path)

    settings = load_generation_config(config_dir)
    catalogs = load_catalogs(config_dir)
    prompt_assets = load_prompt_assets(prompts_dir)

    jobs = build_jobs(catalogs, settings)

    assert len(jobs) == 13
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
    assert schema_a["required"] == ["text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"]
    assert schema_a["properties"]["dominant_trait"]["enum"] == ["conscientiousness"]
    assert schema_a["properties"]["register"]["enum"] == ["haera"]
    assert schema_a["properties"]["temperament_expressed"]["enum"] == ["mixed"]

    schema_f = response_format_f["json_schema"]["schema"]
    assert schema_f["required"] == ["emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type", "temperament_amplifier"]
    assert schema_f["properties"]["previous_emotion"]["enum"] == ["fear", "trust"]
    assert schema_f["properties"]["transition_type"]["enum"] == ["gradual", "sudden", "sustained"]
    assert extra_body["provider"]["require_parameters"] is True


def test_generate_data_builds_v31_jobs_with_context_and_new_tasks(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)

    jobs = build_jobs(tmp_path)
    counts = {task: [job["task"] for job in jobs].count(task) for task in {job["task"] for job in jobs}}

    assert counts == {"A": 2, "B": 2, "C": 3, "D": 1, "E": 1, "F": 2, "G": 2, "H": 1}
    task_a = next(job for job in jobs if job["task"] == "A" and job["temperament_id"] == "choleric")
    task_g = next(job for job in jobs if job["task"] == "G")
    task_h = next(job for job in jobs if job["task"] == "H")

    assert task_a["world_id"] in {"default", "winter"}
    assert task_a["temperament_line"] == "NS=0.8 HA=0.2 RD=0.5 P=0.7 type=choleric"
    assert "[TEMP] NS=0.8 HA=0.2 RD=0.5 P=0.7 type=choleric" in task_a["prompt"]
    assert "[STRESS]" in task_a["prompt"]
    assert task_g["layer"] == "L5"
    assert task_g["oracle_id"] == "oracle_01"
    assert task_h["layer"] == "L0"
    assert task_h["teacher_model"] == "openai/gpt-4.1"


def test_build_response_format_uses_v31_required_fields_for_extended_tasks(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    jobs = build_jobs(tmp_path)
    settings = load_generation_config(tmp_path / "config")

    task_a = next(job for job in jobs if job["task"] == "A")
    task_g = next(job for job in jobs if job["task"] == "G")
    task_h = next(job for job in jobs if job["task"] == "H")

    response_format_a, _ = build_response_format(task_a, settings)
    response_format_g, _ = build_response_format(task_g, settings)
    response_format_h, _ = build_response_format(task_h, settings)

    schema_a = response_format_a["json_schema"]["schema"]
    schema_g = response_format_g["json_schema"]["schema"]
    schema_h = response_format_h["json_schema"]["schema"]

    assert schema_a["required"] == ["text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"]
    assert schema_a["properties"]["temperament_expressed"]["enum"] == ["choleric"]
    assert schema_g["required"] == [
        "interpretation_ko",
        "interpretation_en",
        "action_tendency",
        "confidence",
        "register",
        "misinterpretation_type",
        "temperament_bias",
    ]
    assert schema_g["properties"]["action_tendency"]["enum"] == ["mobilize", "defend", "wait", "retreat", "celebrate", "mourn"]
    assert schema_h["required"] == [
        "name",
        "description_en",
        "resource_modifiers",
        "special_zones",
        "special_resources",
        "agent_modifiers",
    ]
    assert schema_h["properties"]["resource_modifiers"]["type"] == "array"


def test_generate_dataset_prints_progress_and_final_usage_summary(tmp_path: Path, capsys) -> None:
    bootstrap_bilingual_assets(tmp_path)

    def fake_generator(job: dict, system_prompt: str) -> dict:
        assert system_prompt == "teacher system"
        payloads = {
            "A": {"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness", "temperament_expressed": "mixed"},
            "B": {"text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.", "text_en": "The bushes shook hard. Trembling all over, they backed away.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.9, "mimetics": ["오들오들"], "temperament_influence": "mixed_temperament_restrained_fear"},
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
            {"action_id": 0, "confidence": 0.9, "hint_ko": "겁이 치밀어 곧바로 달아났다", "hint_en": "Fear surged, so they fled at once.", "personality_reasoning": "high_conscientiousness", "temperament_factor": "mixed_temperament_balanced_choice"}
            if job["task"] == "E"
            else {"emotion": "fear", "intensity": 0.9, "cause_ko": "날랜 짐승이 바로 눈앞에 덮쳤다", "cause_en": "A fierce beast lunged right in front of them.", "previous_emotion": job["current_emotion_id"], "transition_type": "sudden", "temperament_amplifier": "mixed_temperament_balanced_fear"}
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
                        "temperament_expressed": "mixed",
                    }
                ),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    output_path = tmp_path / "data" / "raw" / "checkpoint.jsonl"
    result = generate_dataset(tmp_path, generator=flaky_generator, limit=2, output_path=output_path, verbose=False)
    skipped_path = output_path.parent / "skipped.jsonl"

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skipped_rows = [json.loads(line) for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert attempts["count"] == 4
    assert result.count == 1
    assert result.skipped_count == 1
    assert len(rows) == 1
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["task"] == "A"
    assert skipped_rows[0]["skip_reason"] == "persistent"
    assert json.loads(rows[0]["output"])["dominant_trait"] == "conscientiousness"


def test_generate_dataset_retries_validation_failures_before_checkpointing(tmp_path: Path) -> None:
    bootstrap_bilingual_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")
    settings["task_variants"] = {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0, "F": 0}
    settings["provider"]["retry_attempts"] = 2
    settings["provider"]["retry_backoff_seconds"] = 0
    write_yaml(tmp_path / "config" / "generation.yaml", settings)

    attempts = {"count": 0}

    def validation_flaky_generator(job: dict, system_prompt: str) -> dict:
        attempts["count"] += 1
        payload = (
            {"text_ko": "끝맺음이 흐려 마음이놓", "text_en": "The ending trails off.", "register": "haera", "dominant_trait": "conscientiousness", "temperament_expressed": "mixed"}
            if attempts["count"] == 1
            else {"text_ko": "곧은 마음에 겁 없고 한번 마음먹으면 끝을 본다.", "text_en": "Fearless and always sees things through.", "register": "haera", "dominant_trait": "conscientiousness", "temperament_expressed": "mixed"}
        )
        return {
            "output": compact_json(payload),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    output_path = tmp_path / "data" / "raw" / "validation_retry.jsonl"
    result = generate_dataset(tmp_path, generator=validation_flaky_generator, limit=1, output_path=output_path, verbose=False)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert result.count == 1
    assert attempts["count"] == 2
    assert len(rows) == 1
    assert json.loads(rows[0]["output"])["dominant_trait"] == "conscientiousness"


def test_parse_and_validate_normalizes_emotion_and_register_variants_before_validation(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")
    job = next(job for job in build_jobs(tmp_path, task_filter={"B"}) if job["task"] == "B")

    normalized, validation_error = parse_and_validate(
        compact_json(
            {
                "text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.",
                "text_en": "The bushes shook hard. Trembling all over, they backed away.",
                "register": "해라체",
                "emotion_expressed": "공포",
                "intensity": 0.9,
                "mimetics": ["오들오들"],
                "temperament_influence": "high_HA_amplified_fear",
            }
        ),
        job,
        settings,
    )

    assert validation_error is None
    payload = json.loads(normalized)
    assert payload["register"] == "haera"
    assert payload["emotion_expressed"] == "fear"


def test_generate_dataset_skips_unrecoverable_validation_failures_and_records_them(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")
    settings["task_variants"] = {"A": 0, "B": 1, "C": 0, "D": 0, "E": 0, "F": 0, "G": 0, "H": 0}
    settings["provider"]["retry_attempts"] = 2
    settings["provider"]["retry_backoff_seconds"] = 0
    write_yaml(tmp_path / "config" / "generation.yaml", settings)

    attempts = {"count": 0}

    def invalid_generator(job: dict, system_prompt: str) -> dict:
        attempts["count"] += 1
        return {
            "output": compact_json(
                {
                    "text_ko": "풀숲이 거세게 흔들렸다. 온몸이 오들오들 떨리며 물러섰다.",
                    "text_en": "The bushes shook hard. Trembling all over, they backed away.",
                    "register": "haera",
                    "emotion_expressed": "mood",
                    "intensity": 0.9,
                    "mimetics": ["오들오들"],
                    "temperament_influence": "high_HA_amplified_fear",
                }
            ),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    output_path = tmp_path / "data" / "raw" / "skip_validation.jsonl"
    result = generate_dataset(tmp_path, generator=invalid_generator, limit=1, output_path=output_path, verbose=False)
    skipped_path = output_path.parent / "skipped.jsonl"

    assert result.count == 0
    assert result.skipped_count == 1
    assert attempts["count"] == 2
    assert output_path.read_text(encoding="utf-8") == ""
    skipped_rows = [json.loads(line) for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(skipped_rows) == 1
    assert skipped_rows[0]["task"] == "B"
    assert skipped_rows[0]["skip_reason"] == "generation_validation_failed:invalid_emotion"


def test_rendered_task_b_prompt_repeats_exact_enum_constraints(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    job = next(job for job in build_jobs(tmp_path, task_filter={"B"}) if job["task"] == "B")
    prompt = job["prompt"]

    emotion_line = "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation"
    register_line = "register must be exactly one of: haera, hao, hae"

    assert emotion_line in prompt
    assert register_line in prompt
    assert prompt.count("joy, sadness, fear, anger, trust, disgust, surprise, anticipation") >= 2
    assert prompt.count("haera, hao, hae") >= 2


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


def test_generate_dataset_batch_plan_writes_batch_scoped_artifacts_and_exact_task_counts(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    batch_plan = {
        "batch_id": "batch_test",
        "task_counts": {"G": 2, "H": 1, "E": 1, "F": 1},
        "reporting": {"progress_every": 2},
        "output": {
            "raw_dir": "data/raw/batch_test",
            "generated_file": "generated.jsonl",
            "skipped_file": "skipped.jsonl",
            "progress_file": "progress.json",
            "summary_file": "summary.json",
        },
    }

    def fake_generator(job: dict, system_prompt: str) -> dict:
        if job["task"] == "E":
            payload = {
                "action_id": 0,
                "confidence": 0.9,
                "hint_ko": "겁이 치밀어 곧바로 달아났다",
                "hint_en": "Fear surged, so they fled at once.",
                "personality_reasoning": job["personality_reasoning"],
                "temperament_factor": "mixed_temperament_balanced_choice",
            }
        elif job["task"] == "F":
            payload = {
                "emotion": "fear",
                "intensity": 0.8,
                "cause_ko": "날랜 짐승이 앞을 막아 겁이 솟았다",
                "cause_en": "A fierce beast blocked the way and fear rose.",
                "previous_emotion": job["current_emotion_id"],
                "transition_type": "sudden",
                "temperament_amplifier": "mixed_temperament_balanced_fear",
            }
        elif job["task"] == "G":
            payload = {
                "interpretation_ko": "신이 길을 열었으니 곧 무리를 이끌겠소.",
                "interpretation_en": "The gods opened the way, so I will lead the tribe soon.",
                "action_tendency": "mobilize",
                "confidence": 0.8,
                "register": job["register"],
                "misinterpretation_type": "overconfident_literal",
                "temperament_bias": "action_oriented certainty",
            }
        else:
            payload = {
                "name": "DungeonEconomy",
                "description_en": "A dungeon-focused world with scarce surface resources.",
                "resource_modifiers": [{"target": "surface_foraging", "multiplier": 0.5}],
                "special_zones": [{"kind": "dungeon_node", "spawn_count_min": 3, "spawn_count_max": 7}],
                "special_resources": [{"name": "magic_stone", "tags": ["currency", "tradeable"]}],
                "agent_modifiers": [{"system": "temperament", "trigger": "essence_equip", "effect": "shift_random_axis"}],
            }
        return {
            "output": compact_json(payload),
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "model": "test-model",
        }

    result = generate_dataset(tmp_path, generator=fake_generator, batch_plan=batch_plan, verbose=False)

    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    progress = json.loads(result.progress_path.read_text(encoding="utf-8"))

    assert result.output_path == (tmp_path / "data" / "raw" / "batch_test" / "generated.jsonl").resolve()
    assert result.skipped_path == (tmp_path / "data" / "raw" / "batch_test" / "skipped.jsonl").resolve()
    assert result.progress_path == (tmp_path / "data" / "raw" / "batch_test" / "progress.json").resolve()
    assert result.summary_path == (tmp_path / "data" / "raw" / "batch_test" / "summary.json").resolve()
    assert Counter(row["task"] for row in rows) == Counter({"G": 2, "H": 1, "E": 1, "F": 1})
    assert summary["counts_by_task"]["planned"] == {"G": 2, "H": 1, "E": 1, "F": 1}
    assert summary["counts_by_task"]["successful"] == {"G": 2, "H": 1, "E": 1, "F": 1}
    assert progress["successful_rows"] == 5
    assert progress["skipped_rows"] == 0


def test_generate_dataset_batch_plan_supports_variant_overrides_for_high_h_counts(tmp_path: Path) -> None:
    bootstrap_v31_assets(tmp_path)
    settings = load_generation_config(tmp_path / "config")
    settings["worldbuilding_texts"] = [
        {
            "id": f"wb_{index:02d}",
            "text": f"세계 {index}은 얼음과 불씨가 엇갈리는 곳이다.",
            "expected_world_type": "winter",
        }
        for index in range(10)
    ]
    write_yaml(tmp_path / "config" / "generation.yaml", settings)
    batch_plan = {
        "batch_id": "batch_test_h",
        "task_counts": {"H": 80},
        "task_variant_overrides": {"H": 8},
        "output": {"raw_dir": "data/raw/batch_test_h"},
    }

    def fake_generator(job: dict, system_prompt: str) -> dict:
        return {
            "output": compact_json(
                {
                    "name": "WinterWorld",
                    "description_en": "A frozen world where fire remains precious.",
                    "resource_modifiers": [{"target": "surface_foraging", "multiplier": 0.2}],
                    "special_zones": [{"kind": "frozen_cave", "spawn_count_min": 1, "spawn_count_max": 4}],
                    "special_resources": [{"name": "ember_seed", "tags": ["fuel"]}],
                    "agent_modifiers": [{"system": "temperament", "trigger": "cold_snap", "effect": "raise_caution"}],
                }
            ),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test-model",
        }

    result = generate_dataset(tmp_path, generator=fake_generator, batch_plan=batch_plan, verbose=False)
    rows = [json.loads(line) for line in result.output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result.count == 80
    assert all(row["task"] == "H" for row in rows)
