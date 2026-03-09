#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, ensure_within_directory, load_text, load_yaml, resolve_path

DEFAULT_REGISTERS = {
    "haera": "해라체로 써라. 문장을 -다, -는다, -았다, -었다 로 끝내라.",
    "hao": "하오체로 써라. 문장을 -오, -소, -시오 로 끝내라.",
    "hae": "해체로 써라. 문장을 -해, -야, -지, -어 로 끝내라.",
}
DEFAULT_TRAIT_AXES = [
    "honesty_humility",
    "emotionality",
    "extraversion",
    "agreeableness",
    "conscientiousness",
    "openness",
]
DEFAULT_REASONING_AXES = [f"high_{axis}" for axis in DEFAULT_TRAIT_AXES]
DEFAULT_TEMPERAMENT_IDS = ["choleric", "melancholic", "sanguine", "phlegmatic", "mixed"]
DEFAULT_TEMPERAMENT_BIASES = [
    "action_oriented",
    "cautious_conservative",
    "optimistic_social",
    "passive_steady",
    "context_dependent",
]
DEFAULT_TEMPERAMENTS = [
    {
        "id": "choleric",
        "ko": "담즙질",
        "tci": {"NS": 0.8, "HA": 0.2, "RD": 0.5, "P": 0.7},
        "keywords": ["당당함", "충동적", "결단력", "지배적"],
        "bias": "action_oriented",
    },
    {
        "id": "melancholic",
        "ko": "우울질",
        "tci": {"NS": 0.2, "HA": 0.8, "RD": 0.6, "P": 0.4},
        "keywords": ["신중함", "불안함", "꼼꼼함", "비관적"],
        "bias": "cautious_conservative",
    },
    {
        "id": "sanguine",
        "ko": "다혈질",
        "tci": {"NS": 0.7, "HA": 0.3, "RD": 0.8, "P": 0.3},
        "keywords": ["낙천적", "사교적", "산만함", "열정적"],
        "bias": "optimistic_social",
    },
    {
        "id": "phlegmatic",
        "ko": "점액질",
        "tci": {"NS": 0.3, "HA": 0.4, "RD": 0.3, "P": 0.6},
        "keywords": ["차분함", "느긋함", "관망적", "끈기"],
        "bias": "passive_steady",
    },
    {
        "id": "mixed",
        "ko": "혼합",
        "tci": {"NS": 0.5, "HA": 0.5, "RD": 0.5, "P": 0.5},
        "keywords": ["균형적", "상황적"],
        "bias": "context_dependent",
    },
]
DEFAULT_WORLDS = [
    {"id": "default", "ko": "기본세계", "desc": "석기시대, 지상자원 정상, 일반 생태계", "vocab_additions": []},
]
DEFAULT_SPEAKER_ROLES = [
    "elder",
    "hunter",
    "shaman",
    "warrior",
    "healer",
    "gatherer",
    "craftsman",
    "chief",
    "scout",
    "observer",
]
DEFAULT_TRANSITION_TYPES = ["gradual", "sudden", "sustained"]
DEFAULT_ORACLE_ACTION_TENDENCIES = ["mobilize", "defend", "wait", "retreat", "celebrate", "mourn"]
DEFAULT_ORACLE_MISINTERPRETATIONS = [
    "overconfident_literal",
    "cautious_reversal",
    "optimistic_expansion",
    "passive_deferral",
    "symbolic_abstraction",
]
EMOTION_VALUE_MAP = {
    "joy": "joy",
    "happiness": "joy",
    "happy": "joy",
    "기쁨": "joy",
    "즐거움": "joy",
    "sadness": "sadness",
    "sad": "sadness",
    "sorrow": "sadness",
    "슬픔": "sadness",
    "fear": "fear",
    "afraid": "fear",
    "fright": "fear",
    "공포": "fear",
    "두려움": "fear",
    "겁": "fear",
    "anger": "anger",
    "angry": "anger",
    "rage": "anger",
    "분노": "anger",
    "trust": "trust",
    "belief": "trust",
    "믿음": "trust",
    "신뢰": "trust",
    "disgust": "disgust",
    "혐오": "disgust",
    "역겨움": "disgust",
    "surprise": "surprise",
    "surprised": "surprise",
    "놀람": "surprise",
    "anticipation": "anticipation",
    "expectation": "anticipation",
    "기대": "anticipation",
}
REGISTER_VALUE_MAP = {
    "haera": "haera",
    "해라": "haera",
    "해라체": "haera",
    "hao": "hao",
    "하오": "hao",
    "하오체": "hao",
    "hae": "hae",
    "해": "hae",
    "해체": "hae",
}
DEFAULT_DOMINANT_TRAITS = {
    "cautious_elder": "conscientiousness",
    "reckless_hunter": "extraversion",
    "visionary_shaman": "openness",
    "vengeful_warrior": "emotionality",
    "empathetic_healer": "agreeableness",
    "greedy_gatherer": "honesty_humility",
    "diligent_craftsman": "conscientiousness",
    "charismatic_chief": "extraversion",
    "paranoid_scout": "emotionality",
    "stoic_observer": "openness",
}
DEFAULT_SPEAKER_ROLE_MAP = {
    "cautious_elder": "elder",
    "reckless_hunter": "hunter",
    "visionary_shaman": "shaman",
    "vengeful_warrior": "warrior",
    "empathetic_healer": "healer",
    "greedy_gatherer": "gatherer",
    "diligent_craftsman": "craftsman",
    "charismatic_chief": "chief",
    "paranoid_scout": "scout",
    "stoic_observer": "observer",
}


def load_generation_config(config_dir: Path) -> dict:
    return load_yaml(config_dir / "generation.yaml")


def load_local_env(repo_root: Path) -> None:
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_catalogs(config_dir: Path) -> dict:
    settings = load_generation_config(config_dir)
    return {
        "situations": load_yaml(config_dir / "situations.yaml").get("situations", []),
        "personalities": load_yaml(config_dir / "personalities.yaml").get("personalities", []),
        "emotions": load_yaml(config_dir / "emotions.yaml").get("emotions", []),
        "temperaments": settings.get("temperaments", []),
        "worlds": settings.get("worlds", []),
        "oracles": settings.get("oracles", []),
        "worldbuilding_texts": settings.get("worldbuilding_texts", []),
    }


def load_prompt_assets(prompts_dir: Path) -> dict:
    teacher_dir = prompts_dir / "teacher"
    assets = {"system": load_text(teacher_dir / "system.txt"), "tasks": {}}
    for task in ("A", "B", "C", "D", "E", "F", "G", "H"):
        path = teacher_dir / f"task_{task.lower()}.txt"
        if path.exists():
            assets["tasks"][task] = load_text(path)
    return assets


def _variant_count(settings: dict, task: str) -> int:
    if "task_variants" in settings:
        return int(settings["task_variants"].get(task, 0))
    return int(settings.get("generation", {}).get("variants", {}).get(f"task_{task.lower()}", 0))


def _register_instructions(settings: dict) -> dict:
    return {**DEFAULT_REGISTERS, **settings.get("register_instructions", {})}


def _trait_axes(settings: dict) -> list[str]:
    return list(settings.get("validation", {}).get("trait_axes", DEFAULT_TRAIT_AXES))


def _reasoning_axes(settings: dict) -> list[str]:
    return list(settings.get("validation", {}).get("reasoning_axes", DEFAULT_REASONING_AXES))


def _temperament_ids(settings: dict) -> list[str]:
    configured = settings.get("validation", {}).get("temperament_ids")
    if configured:
        return list(configured)
    temperaments = settings.get("temperaments", [])
    if temperaments:
        return [temperament["id"] for temperament in temperaments]
    return list(DEFAULT_TEMPERAMENT_IDS)


def _temperament_biases(settings: dict) -> list[str]:
    configured = settings.get("validation", {}).get("temperament_biases")
    if configured:
        return list(configured)
    temperaments = settings.get("temperaments", [])
    if temperaments:
        return [temperament.get("bias", temperament["id"]) for temperament in temperaments]
    return list(DEFAULT_TEMPERAMENT_BIASES)


def _oracle_action_tendencies(settings: dict) -> list[str]:
    validation = settings.get("validation", {})
    return list(validation.get("oracle_action_tendencies") or validation.get("action_tendencies") or DEFAULT_ORACLE_ACTION_TENDENCIES)


def _oracle_misinterpretations(settings: dict) -> list[str]:
    validation = settings.get("validation", {})
    return list(validation.get("oracle_misinterpretations") or validation.get("misinterpretation_types") or DEFAULT_ORACLE_MISINTERPRETATIONS)


def _speaker_roles(settings: dict) -> list[str]:
    return list(settings.get("validation", {}).get("speaker_roles", DEFAULT_SPEAKER_ROLES))


def _transition_types(settings: dict) -> list[str]:
    return list(settings.get("validation", {}).get("transition_types", DEFAULT_TRANSITION_TYPES))


def _emotion_ids(catalogs: dict) -> list[str]:
    return [emotion["id"] for emotion in catalogs.get("emotions", [])] or ["joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"]


def _tci_axis_mapping(settings: dict) -> dict[str, str]:
    return settings.get("v31_context", {}).get(
        "temperament_axes",
        {
            "NS": "novelty_seeking",
            "HA": "harm_avoidance",
            "RD": "reward_dependence",
            "P": "persistence",
        },
    )


def _dominant_temperament_axis(temperament: dict, settings: dict) -> tuple[str | None, str | None]:
    mapping = _tci_axis_mapping(settings)
    tci = temperament.get("tci", {})
    scored = [(code, tci.get(code)) for code in ("NS", "HA", "RD", "P") if isinstance(tci.get(code), (int, float))]
    if not scored:
        return None, None
    code, _ = max(scored, key=lambda item: (float(item[1]), item[0]))
    return code, mapping.get(code, code)


def _catalog_temperaments(catalogs: dict) -> list[dict]:
    return catalogs.get("temperaments") or [next(temperament for temperament in DEFAULT_TEMPERAMENTS if temperament["id"] == "mixed")]


def _catalog_worlds(catalogs: dict) -> list[dict]:
    return catalogs.get("worlds") or list(DEFAULT_WORLDS)


def _format_scalar(value: object, default: float = 0.5) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return f"{default:g}"


def _temperament_line(temperament: dict) -> str:
    tci = temperament.get("tci", {})
    return (
        f"NS={_format_scalar(tci.get('NS'))} "
        f"HA={_format_scalar(tci.get('HA'))} "
        f"RD={_format_scalar(tci.get('RD'))} "
        f"P={_format_scalar(tci.get('P'))} "
        f"type={temperament.get('id', 'mixed')}"
    )


def _v31_context(*, rng: random.Random, temperaments: list[dict], worlds: list[dict], temperament: dict | None = None, world: dict | None = None) -> dict:
    chosen_temperament = temperament or rng.choice(temperaments or list(DEFAULT_TEMPERAMENTS))
    chosen_world = world or rng.choice(worlds or list(DEFAULT_WORLDS))
    stress = round(rng.uniform(0.1, 0.9), 1)
    tci = chosen_temperament.get("tci", {})
    return {
        "temperament_id": chosen_temperament.get("id", "mixed"),
        "temperament_name": chosen_temperament.get("ko", chosen_temperament.get("id", "mixed")),
        "temperament_keywords": chosen_temperament.get("keywords", []),
        "temperament_bias": chosen_temperament.get("bias", chosen_temperament.get("id", "mixed")),
        "temperament_line": _temperament_line(chosen_temperament),
        "tci_ns": _format_scalar(tci.get("NS")),
        "tci_ha": _format_scalar(tci.get("HA")),
        "tci_rd": _format_scalar(tci.get("RD")),
        "tci_p": _format_scalar(tci.get("P")),
        "stress": stress,
        "world_id": chosen_world.get("id", "default"),
        "world_name": chosen_world.get("ko", chosen_world.get("id", "default")),
        "world_desc": chosen_world.get("desc", ""),
        "world_vocab": ", ".join(chosen_world.get("vocab_additions", [])),
        "world_vocab_additions": ", ".join(chosen_world.get("vocab_additions", [])),
    }


def _dominant_trait_for_personality(personality: dict, settings: dict) -> str:
    allowed = _trait_axes(settings)
    trait = personality.get("dominant_trait") or DEFAULT_DOMINANT_TRAITS.get(personality["id"])
    if trait and trait in allowed:
        return trait
    return allowed[0]


def _speaker_role_for_personality(personality: dict, settings: dict) -> str:
    role = personality.get("speaker_role") or DEFAULT_SPEAKER_ROLE_MAP.get(personality["id"])
    if role:
        return role
    return _speaker_roles(settings)[0]


def _personality_reasoning_for_personality(personality: dict, settings: dict) -> str:
    allowed = _reasoning_axes(settings)
    reasoning = personality.get("personality_reasoning")
    if reasoning and reasoning in allowed:
        return reasoning
    dominant_trait = _dominant_trait_for_personality(personality, settings)
    candidate = f"high_{dominant_trait}"
    if candidate in allowed:
        return candidate
    return allowed[0]


def _dominant_trait_for_context(personality: dict, settings: dict, temperament: dict | None = None) -> str:
    allowed = _trait_axes(settings)
    explicit = personality.get("dominant_trait")
    if explicit and explicit in allowed:
        return explicit
    if temperament is not None:
        _, trait_name = _dominant_temperament_axis(temperament, settings)
        if trait_name and trait_name in allowed:
            return trait_name
    fallback = DEFAULT_DOMINANT_TRAITS.get(personality["id"])
    if fallback and fallback in allowed:
        return fallback
    return allowed[0]


def _reasoning_for_context(personality: dict, settings: dict, temperament: dict | None = None) -> str:
    allowed = _reasoning_axes(settings)
    explicit = personality.get("personality_reasoning")
    if explicit and explicit in allowed:
        return explicit
    if temperament is not None:
        axis_code, trait_name = _dominant_temperament_axis(temperament, settings)
        for candidate in (f"high_{axis_code}" if axis_code else None, f"high_{trait_name}" if trait_name else None):
            if candidate and candidate in allowed:
                return candidate
    fallback = _personality_reasoning_for_personality(personality, settings)
    if fallback in allowed:
        return fallback
    return allowed[0]


def _task_teacher_model(settings: dict, task: str) -> str | None:
    task_settings = settings.get("tasks", {}).get(task, {})
    return task_settings.get("teacher_model") or settings.get("provider", {}).get("task_model_overrides", {}).get(task)


def _repo_prompt_assets(repo_root: Path, settings: dict) -> dict:
    prompts = settings.get("prompts", {})
    teacher = prompts.get("teacher", {})
    if teacher.get("tasks"):
        return {
            "system": load_text(resolve_path(repo_root, teacher["system"])),
            "tasks": {task: load_text(resolve_path(repo_root, path)) for task, path in teacher["tasks"].items()},
        }

    paths = settings.get("paths", {})
    system_target = prompts.get("teacher_system") or paths.get("teacher_system_prompt") or "prompts/teacher/system.txt"
    assets = {"system": load_text(resolve_path(repo_root, system_target)), "tasks": {}}
    for task in ("A", "B", "C", "D", "E", "F", "G", "H"):
        prompt_target = prompts.get(f"task_{task.lower()}")
        if prompt_target is None:
            candidate = repo_root / "prompts" / "teacher" / f"task_{task.lower()}.txt"
            if candidate.exists():
                prompt_target = str(candidate.relative_to(repo_root))
        if prompt_target:
            assets["tasks"][task] = load_text(resolve_path(repo_root, prompt_target))
    return assets


def _build_jobs_from_catalogs(catalogs: dict, settings: dict, *, system_prompt: str = "", seed: int | None = None) -> list[dict]:
    rng = random.Random(settings.get("seed", 42) if seed is None else seed)
    situations = catalogs.get("situations", [])
    personalities = catalogs.get("personalities", [])
    emotions = catalogs.get("emotions", [])
    temperaments = _catalog_temperaments(catalogs)
    worlds = _catalog_worlds(catalogs)
    oracles = catalogs.get("oracles", [])
    worldbuilding_texts = catalogs.get("worldbuilding_texts", [])
    oracle_temperaments = [temperament for temperament in temperaments if temperament.get("id") != "mixed"] or temperaments
    emotion_ids = _emotion_ids(catalogs)
    names = settings.get("names") or settings.get("generation", {}).get("task_d_names", ["돌이"])
    registers = _register_instructions(settings)
    default_register = settings.get("defaults", {}).get("register", "haera")
    jobs: list[dict] = []

    for personality in personalities:
        for temperament in temperaments:
            for variant in range(_variant_count(settings, "A")):
                context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperament)
                jobs.append(
                    {
                        "task": "A",
                        "layer": "L4",
                        "expected_format": "json",
                        "variant": variant,
                        "personality_id": personality["id"],
                        "personality_name": personality.get("ko", personality["id"]),
                        "personality_desc": personality.get("desc", ""),
                        "personality_keywords": personality.get("keywords", []),
                        "keywords": ", ".join(personality.get("keywords", [])),
                        "register": "haera",
                        "register_instruction": registers["haera"],
                        "dominant_trait": _dominant_trait_for_context(personality, settings, temperament),
                        "teacher_model": _task_teacher_model(settings, "A"),
                        "system_prompt": system_prompt,
                        **context,
                    }
                )

    for situation_index, situation in enumerate(situations):
        for emotion_index, emotion in enumerate(emotions):
            for personality_index, personality in enumerate(personalities):
                for variant in range(_variant_count(settings, "B")):
                    intensity = rng.choice(emotion.get("intensities", [0.6]))
                    context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperaments[(personality_index + variant) % len(temperaments)], world=worlds[(situation_index + emotion_index + variant) % len(worlds)])
                    jobs.append(
                        {
                            "task": "B",
                            "layer": "L4",
                            "expected_format": "json",
                            "variant": variant,
                            "personality_id": personality["id"],
                            "personality_name": personality.get("ko", personality["id"]),
                            "personality_desc": personality.get("desc", ""),
                            "personality_keywords": personality.get("keywords", []),
                            "keywords": ", ".join(personality.get("keywords", [])),
                            "register": "haera",
                            "register_instruction": registers["haera"],
                            "emotion_id": emotion["id"],
                            "emotion_name": emotion.get("ko", emotion["id"]),
                            "emotion": emotion.get("ko", emotion["id"]),
                            "emotion_intensity": intensity,
                            "intensity": intensity,
                            "mimetic": rng.choice(emotion.get("mimetics", [""])) if emotion.get("mimetics") else "",
                            "emotion_options": emotion_ids,
                            "situation_id": situation["id"],
                            "scenario_name": situation.get("ko", situation["id"]),
                            "scenario_desc": situation.get("desc", ""),
                            "situation": situation.get("ko", situation["id"]),
                            "teacher_model": _task_teacher_model(settings, "B"),
                            "system_prompt": system_prompt,
                            **context,
                        }
                    )

    for situation_index, situation in enumerate(situations):
        for personality_index, personality in enumerate(personalities):
            for register_index, register in enumerate(registers):
                for variant in range(_variant_count(settings, "C")):
                    emotion = rng.choice(emotions) if emotions else {"id": "neutral", "ko": "담담함"}
                    context = _v31_context(
                        rng=rng,
                        temperaments=temperaments,
                        worlds=worlds,
                        temperament=temperaments[(personality_index + register_index + variant) % len(temperaments)],
                        world=worlds[(situation_index + register_index + variant) % len(worlds)],
                    )
                    jobs.append(
                        {
                            "task": "C",
                            "layer": "L4",
                            "expected_format": "json",
                            "variant": variant,
                            "personality_id": personality["id"],
                            "personality_name": personality.get("ko", personality["id"]),
                            "personality_desc": personality.get("desc", ""),
                            "personality_keywords": personality.get("keywords", []),
                            "keywords": ", ".join(personality.get("keywords", [])),
                            "register": register,
                            "register_instruction": registers.get(register, registers["haera"]),
                            "emotion_id": emotion["id"],
                            "emotion_name": emotion.get("ko", emotion["id"]),
                            "emotion": emotion.get("ko", emotion["id"]),
                            "emotion_options": emotion_ids,
                            "speaker_role": _speaker_role_for_personality(personality, settings),
                            "situation_id": situation["id"],
                            "scenario_name": situation.get("ko", situation["id"]),
                            "scenario_desc": situation.get("desc", ""),
                            "situation": situation.get("ko", situation["id"]),
                            "teacher_model": _task_teacher_model(settings, "C"),
                            "system_prompt": system_prompt,
                            **context,
                        }
                    )

    for situation_index, situation in enumerate(situations):
        for variant in range(_variant_count(settings, "D")):
            context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperaments[variant % len(temperaments)], world=worlds[situation_index % len(worlds)])
            jobs.append(
                {
                    "task": "D",
                    "layer": "L4",
                    "expected_format": "json",
                    "variant": variant,
                    "name": names[variant % len(names)],
                    "register": "haera",
                    "register_instruction": registers["haera"],
                    "situation_id": situation["id"],
                    "scenario_name": situation.get("ko", situation["id"]),
                    "scenario_desc": situation.get("desc", ""),
                    "situation": situation.get("ko", situation["id"]),
                    "teacher_model": _task_teacher_model(settings, "D"),
                    "system_prompt": system_prompt,
                    **context,
                }
            )

    for situation_index, situation in enumerate(situations):
        action_options = situation.get("action_options") or situation.get("typical_actions") or []
        if not action_options:
            continue
        for personality_index, personality in enumerate(personalities):
            for variant in range(_variant_count(settings, "E")):
                emotion = rng.choice(emotions) if emotions else {"id": "fear", "ko": "두려움", "intensities": [0.7]}
                intensity = rng.choice(emotion.get("intensities", [0.7]))
                options_line = " ".join(f"{idx}:{option}" for idx, option in enumerate(action_options))
                context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperaments[(personality_index + variant) % len(temperaments)], world=worlds[(situation_index + variant) % len(worlds)])
                jobs.append(
                    {
                        "task": "E",
                        "layer": "L3",
                        "expected_format": "json",
                        "variant": variant,
                        "name": names[(personality_index + variant) % len(names)],
                        "personality_id": personality["id"],
                        "personality_name": personality.get("ko", personality["id"]),
                        "personality_desc": personality.get("desc", ""),
                        "personality_keywords": personality.get("keywords", []),
                        "keywords": ", ".join(personality.get("keywords", [])),
                        "emotion_id": emotion["id"],
                        "emotion_name": emotion.get("ko", emotion["id"]),
                        "emotion": emotion.get("ko", emotion["id"]),
                        "emotion_intensity": intensity,
                        "intensity": intensity,
                        "register": "",
                        "register_instruction": "",
                        "personality_reasoning": _reasoning_for_context(personality, settings, context_temperament := temperaments[(personality_index + variant) % len(temperaments)]),
                        "situation_id": situation["id"],
                        "scenario_name": situation.get("ko", situation["id"]),
                        "scenario_desc": situation.get("desc", ""),
                        "situation": situation.get("ko", situation["id"]),
                        "action_options": action_options,
                        "options_line": options_line,
                        "teacher_model": _task_teacher_model(settings, "E"),
                        "system_prompt": system_prompt,
                        **context,
                    }
                )

    for situation_index, situation in enumerate(situations):
        for personality_index, personality in enumerate(personalities):
            for current_emotion in emotions:
                for variant in range(_variant_count(settings, "F")):
                    current_intensity = rng.choice(current_emotion.get("intensities", [0.5]))
                    context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperaments[(personality_index + variant) % len(temperaments)], world=worlds[(situation_index + variant) % len(worlds)])
                    jobs.append(
                        {
                            "task": "F",
                            "layer": "L3",
                            "expected_format": "json",
                            "variant": variant,
                            "name": names[(personality_index + variant) % len(names)],
                            "personality_id": personality["id"],
                            "personality_name": personality.get("ko", personality["id"]),
                            "personality_desc": personality.get("desc", ""),
                            "personality_keywords": personality.get("keywords", []),
                            "keywords": ", ".join(personality.get("keywords", [])),
                            "current_emotion_id": current_emotion["id"],
                            "current_emotion_name": current_emotion.get("ko", current_emotion["id"]),
                            "current_emotion_intensity": current_intensity,
                            "emotion_id": current_emotion["id"],
                            "emotion_name": current_emotion.get("ko", current_emotion["id"]),
                            "emotion": current_emotion.get("ko", current_emotion["id"]),
                            "emotion_intensity": current_intensity,
                            "register": "",
                            "register_instruction": "",
                            "transition_types": _transition_types(settings),
                            "emotion_options": emotion_ids,
                            "situation_id": situation["id"],
                            "scenario_name": situation.get("ko", situation["id"]),
                            "scenario_desc": situation.get("desc", ""),
                            "situation": situation.get("ko", situation["id"]),
                            "teacher_model": _task_teacher_model(settings, "F"),
                            "system_prompt": system_prompt,
                            **context,
                        }
                    )

    for oracle in oracles:
        for personality in personalities:
            for temperament in oracle_temperaments:
                for variant in range(_variant_count(settings, "G")):
                    context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, temperament=temperament, world=worlds[0])
                    register = personality.get("default_register", "hao")
                    jobs.append(
                        {
                            "task": "G",
                            "layer": "L5",
                            "expected_format": "json",
                            "variant": variant,
                            "personality_id": personality["id"],
                            "personality_name": personality.get("ko", personality["id"]),
                            "personality_desc": personality.get("desc", ""),
                            "personality_keywords": personality.get("keywords", []),
                            "keywords": ", ".join(personality.get("keywords", [])),
                            "role": "chief",
                            "speaker_role": "chief",
                            "recent_context": "최근 기근, 무리 줄어듦",
                            "register": register,
                            "register_instruction": registers.get(register, registers["hao"]),
                            "oracle_id": oracle["id"],
                            "oracle_text_ko": oracle.get("text_ko", ""),
                            "oracle_text_en": oracle.get("text_en", ""),
                            "oracle_ambiguity": oracle.get("ambiguity", ""),
                            "teacher_model": _task_teacher_model(settings, "G"),
                            "system_prompt": system_prompt,
                            **context,
                        }
                    )

    for worldbuilding_text in worldbuilding_texts:
        for variant in range(_variant_count(settings, "H")):
            matched_world = next((world for world in worlds if world.get("id") == worldbuilding_text.get("expected_world_type")), worlds[0])
            context = _v31_context(rng=rng, temperaments=temperaments, worlds=worlds, world=matched_world)
            jobs.append(
                {
                    "task": "H",
                    "layer": "L0",
                    "expected_format": "json",
                    "variant": variant,
                    "worldbuilding_id": worldbuilding_text["id"],
                    "worldbuilding_text": worldbuilding_text.get("text", ""),
                    "expected_world_type": worldbuilding_text.get("expected_world_type", ""),
                    "teacher_model": _task_teacher_model(settings, "H"),
                    "system_prompt": system_prompt,
                    **context,
                }
            )

    return jobs


def build_jobs(
    catalogs_or_repo_root: dict | Path,
    settings: dict | None = None,
    seed: int | None = None,
    task_filter: set[str] | None = None,
) -> list[dict]:
    if isinstance(catalogs_or_repo_root, Path):
        repo_root = catalogs_or_repo_root
        repo_settings = load_generation_config(repo_root / "config")
        catalogs = load_catalogs(repo_root / "config")
        prompt_assets = _repo_prompt_assets(repo_root, repo_settings)
        jobs = _build_jobs_from_catalogs(catalogs, repo_settings, system_prompt=prompt_assets["system"], seed=seed)
        if task_filter is not None:
            jobs = [job for job in jobs if job["task"] in task_filter]
        if prompt_assets["tasks"]:
            for job in jobs:
                if job["task"] in prompt_assets["tasks"]:
                    job["prompt"] = render_prompt(job, prompt_assets)
        return jobs
    if settings is None:
        raise ValueError("settings must be provided when building jobs from catalogs")
    jobs = _build_jobs_from_catalogs(catalogs_or_repo_root, settings, seed=seed)
    if task_filter is not None:
        jobs = [job for job in jobs if job["task"] in task_filter]
    return jobs


def render_prompt(job: dict, prompt_assets: dict) -> str:
    template = prompt_assets["tasks"][job["task"]]
    replacements = {
        "personality_name": job.get("personality_name", ""),
        "personality_desc": job.get("personality_desc", ""),
        "personality_keywords": ", ".join(job.get("personality_keywords", [])),
        "keywords": job.get("keywords", ", ".join(job.get("personality_keywords", []))),
        "temperament_id": job.get("temperament_id", ""),
        "temperament_name": job.get("temperament_name", ""),
        "temperament_keywords": ", ".join(job.get("temperament_keywords", [])),
        "temperament_bias": job.get("temperament_bias", ""),
        "temperament_line": job.get("temperament_line", ""),
        "stress": job.get("stress", ""),
        "tci_ns": job.get("tci_ns", ""),
        "tci_ha": job.get("tci_ha", ""),
        "tci_rd": job.get("tci_rd", ""),
        "tci_p": job.get("tci_p", ""),
        "world_id": job.get("world_id", ""),
        "world_name": job.get("world_name", ""),
        "world_desc": job.get("world_desc", ""),
        "world_vocab": job.get("world_vocab", ""),
        "world_vocab_additions": job.get("world_vocab_additions", job.get("world_vocab", "")),
        "emotion_name": job.get("emotion_name", ""),
        "emotion_id": job.get("emotion_id", ""),
        "emotion": job.get("emotion", job.get("emotion_name", "")),
        "emotion_intensity": job.get("emotion_intensity", ""),
        "intensity": job.get("intensity", job.get("emotion_intensity", "")),
        "mimetic": job.get("mimetic", ""),
        "emotion_options": ", ".join(job.get("emotion_options", [])),
        "scenario_name": job.get("scenario_name", ""),
        "scenario_desc": job.get("scenario_desc", ""),
        "situation": job.get("scenario_desc") or job.get("scenario_name", ""),
        "situation_id": job.get("situation_id", ""),
        "register_instruction": job.get("register_instruction", ""),
        "register": job.get("register", ""),
        "action_options": job.get("options_line", job.get("action_options", "")),
        "options_line": job.get("options_line", ""),
        "current_emotion_name": job.get("current_emotion_name", ""),
        "current_emotion_id": job.get("current_emotion_id", ""),
        "current_emotion_intensity": job.get("current_emotion_intensity", ""),
        "dominant_trait": job.get("dominant_trait", ""),
        "personality_reasoning": job.get("personality_reasoning", ""),
        "speaker_role": job.get("speaker_role", ""),
        "transition_types": ", ".join(job.get("transition_types", DEFAULT_TRANSITION_TYPES)),
        "oracle_id": job.get("oracle_id", ""),
        "oracle_text_ko": job.get("oracle_text_ko", ""),
        "oracle_text_en": job.get("oracle_text_en", ""),
        "oracle_ambiguity": job.get("oracle_ambiguity", ""),
        "role": job.get("role", ""),
        "recent_context": job.get("recent_context", ""),
        "worldbuilding_id": job.get("worldbuilding_id", ""),
        "worldbuilding_text": job.get("worldbuilding_text", ""),
        "expected_world_type": job.get("expected_world_type", ""),
        "name": job.get("name", ""),
        "variant": job.get("variant", 0),
    }
    placeholder = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in replacements:
            return match.group(0)
        return str(replacements[key])

    return placeholder.sub(replace, template)


def parse_task_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    parsed = {part.strip().upper() for part in value.split(",") if part.strip()}
    return parsed or None


def select_jobs(jobs: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit >= len(jobs):
        return list(jobs)
    if limit <= 0:
        return []

    task_order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for job in jobs:
        task = job["task"]
        if task not in buckets:
            buckets[task] = []
            task_order.append(task)
        buckets[task].append(job)

    if len(task_order) <= 1:
        return jobs[:limit]

    selected: list[dict] = []
    offsets = {task: 0 for task in task_order}
    while len(selected) < limit:
        progressed = False
        for task in task_order:
            offset = offsets[task]
            bucket = buckets[task]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[task] += 1
            progressed = True
            if len(selected) == limit:
                break
        if not progressed:
            break
    return selected


def build_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return ensure_directory(output_dir) / f"generated_{timestamp}.jsonl"


def _raw_dir(repo_root: Path, settings: dict) -> Path:
    return ensure_directory(resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw")))


def _resolve_cli_output_path(repo_root: Path, settings: dict, output_path: str | Path | None = None) -> Path:
    raw_dir = _raw_dir(repo_root, settings)
    if output_path is None:
        return build_output_path(raw_dir)
    candidate = resolve_path(repo_root, output_path)
    return ensure_within_directory(raw_dir, candidate, label="raw_dir output_path")


def default_raw_output_path(repo_root: Path, stamp: str | None = None) -> Path:
    settings = load_generation_config(repo_root / "config")
    raw_dir = _raw_dir(repo_root, settings)
    if stamp is None:
        return build_output_path(raw_dir)
    return raw_dir / f"generated_{stamp}.jsonl"


def extract_usage(usage: object | None) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        payload = usage.model_dump()
    elif isinstance(usage, dict):
        payload = usage
    else:
        payload = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }

    prompt_tokens = int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0)
    completion_tokens = int(payload.get("completion_tokens") or payload.get("output_tokens") or 0)
    total_tokens = int(payload.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost_usd(usage: dict[str, int], settings: dict) -> float:
    pricing = settings.get("provider", {}).get("pricing", {})
    input_rate = float(pricing.get("input_per_million_tokens_usd", 0.0))
    output_rate = float(pricing.get("output_per_million_tokens_usd", 0.0))
    return round(
        (usage["prompt_tokens"] / 1_000_000 * input_rate)
        + (usage["completion_tokens"] / 1_000_000 * output_rate),
        8,
    )


def normalize_generation_result(result: object, settings: dict, *, fallback_model: str | None = None) -> AttrDict:
    if isinstance(result, dict):
        output_text = result.get("output") or result.get("text") or ""
        if isinstance(output_text, (dict, list)):
            output_text = json.dumps(output_text, ensure_ascii=False, separators=(",", ":"))
        usage = extract_usage(result.get("usage"))
        model = result.get("model") or fallback_model or settings.get("provider", {}).get("default_model")
    else:
        if isinstance(result, (dict, list)):
            output_text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        else:
            output_text = str(result)
        usage = extract_usage(None)
        model = fallback_model or settings.get("provider", {}).get("default_model")

    return AttrDict(
        output=output_text,
        usage=usage,
        model=model,
        estimated_cost_usd=estimate_cost_usd(usage, settings),
    )


def reporting_settings(settings: dict) -> dict:
    return settings.get("reporting", {})


def print_progress(
    *,
    index: int,
    total: int,
    elapsed_seconds: float,
    totals: dict[str, float],
    last_request_seconds: float,
) -> None:
    rows_per_second = index / elapsed_seconds if elapsed_seconds > 0 else 0.0
    tokens_per_second = totals["total_tokens"] / elapsed_seconds if elapsed_seconds > 0 else 0.0
    print(
        f"[{index}/{total}] elapsed={elapsed_seconds:.2f}s "
        f"last={last_request_seconds:.2f}s rows_per_second={rows_per_second:.2f} "
        f"tokens={int(totals['total_tokens'])} estimated_cost_usd=${totals['estimated_cost_usd']:.6f}",
        flush=True,
    )
    print(
        f"  prompt_tokens={int(totals['prompt_tokens'])} completion_tokens={int(totals['completion_tokens'])} "
        f"tokens_per_second={tokens_per_second:.2f}",
        flush=True,
    )


def print_final_summary(*, result: AttrDict) -> None:
    print("Generation summary", flush=True)
    print(f"  output_path={result.output_path}", flush=True)
    print(f"  skipped_path={result.skipped_path}", flush=True)
    print(f"  rows={result.count}", flush=True)
    print(f"  skipped={result.skipped_count}", flush=True)
    print(f"  elapsed_seconds={result.elapsed_seconds:.2f}", flush=True)
    print(f"  prompt_tokens={result.prompt_tokens}", flush=True)
    print(f"  completion_tokens={result.completion_tokens}", flush=True)
    print(f"  total_tokens={result.total_tokens}", flush=True)
    print(f"  estimated_cost_usd=${result.estimated_cost_usd:.6f}", flush=True)
    print(f"  rows_per_second={result.rows_per_second:.2f}", flush=True)
    print(f"  tokens_per_second={result.tokens_per_second:.2f}", flush=True)


def _task_text_limits(settings: dict, task: str) -> tuple[int, int]:
    limits = settings.get("validation", {}).get("task_limits", {}).get(task, {})
    return int(limits.get("min_chars", 1)), int(limits.get("max_chars", 200))


def build_response_format(job: dict, settings: dict) -> tuple[dict | None, dict | None]:
    provider = settings.get("provider", {})
    extra_body = None
    if provider.get("require_parameters") is not None:
        extra_body = {"provider": {"require_parameters": bool(provider["require_parameters"])}}

    valid_emotions = job.get("emotion_options") or settings.get("validation", {}).get("emotions") or _emotion_ids({"emotions": settings.get("emotions", [])})
    trait_axes = _trait_axes(settings)
    reasoning_axes = _reasoning_axes(settings)
    temperament_ids = _temperament_ids(settings)
    speaker_roles = _speaker_roles(settings)
    transition_types = _transition_types(settings)
    action_tendencies = _oracle_action_tendencies(settings)
    misinterpretations = _oracle_misinterpretations(settings)
    task = job.get("task")

    if task == "A":
        min_chars, max_chars = _task_text_limits(settings, "A")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["text_ko", "text_en", "register", "dominant_trait", "temperament_expressed"],
            "properties": {
                "text_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "text_en": {"type": "string", "minLength": 3},
                "register": {"type": "string", "enum": [job.get("register", "haera")]},
                "dominant_trait": {"type": "string", "enum": [job.get("dominant_trait")] if job.get("dominant_trait") else trait_axes},
                "temperament_expressed": {"type": "string", "enum": [job.get("temperament_id")] if job.get("temperament_id") else temperament_ids},
            },
        }
    elif task == "B":
        min_chars, max_chars = _task_text_limits(settings, "B")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["text_ko", "text_en", "register", "emotion_expressed", "intensity", "mimetics", "temperament_influence"],
            "properties": {
                "text_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "text_en": {"type": "string", "minLength": 3},
                "register": {"type": "string", "enum": [job.get("register", "haera")]},
                "emotion_expressed": {"type": "string", "enum": list(valid_emotions)},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "mimetics": {"type": "array", "items": {"type": "string"}, "minItems": 0},
                "temperament_influence": {"type": "string", "minLength": 3},
            },
        }
    elif task == "C":
        min_chars, max_chars = _task_text_limits(settings, "C")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["speech_ko", "speech_en", "register", "emotion_expressed", "speaker_role", "temperament_tone"],
            "properties": {
                "speech_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "speech_en": {"type": "string", "minLength": 3},
                "register": {"type": "string", "enum": [job.get("register", "haera")]},
                "emotion_expressed": {"type": "string", "enum": list(valid_emotions)},
                "speaker_role": {"type": "string", "enum": [job.get("speaker_role")] if job.get("speaker_role") else speaker_roles},
                "temperament_tone": {"type": "string", "minLength": 3},
            },
        }
    elif task == "D":
        min_chars, max_chars = _task_text_limits(settings, "D")
        event_enum = [job.get("situation_id")] if job.get("situation_id") else None
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["text_ko", "text_en", "event_type"],
            "properties": {
                "text_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "text_en": {"type": "string", "minLength": 3},
                "event_type": {"type": "string", **({"enum": event_enum} if event_enum else {})},
            },
        }
    elif task == "E":
        min_chars, max_chars = _task_text_limits(settings, "E")
        action_options = job.get("action_options", [])
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["action_id", "confidence", "hint_ko", "hint_en", "personality_reasoning", "temperament_factor"],
            "properties": {
                "action_id": {"type": "integer", "enum": list(range(len(action_options)))},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "hint_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "hint_en": {"type": "string", "minLength": 3},
                "personality_reasoning": {"type": "string", "enum": [job.get("personality_reasoning")] if job.get("personality_reasoning") else reasoning_axes},
                "temperament_factor": {"type": "string", "minLength": 3},
            },
        }
    elif task == "F":
        min_chars, max_chars = _task_text_limits(settings, "F")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["emotion", "intensity", "cause_ko", "cause_en", "previous_emotion", "transition_type", "temperament_amplifier"],
            "properties": {
                "emotion": {"type": "string", "enum": list(valid_emotions)},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "cause_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "cause_en": {"type": "string", "minLength": 3},
                "previous_emotion": {"type": "string", "enum": list(valid_emotions)},
                "transition_type": {"type": "string", "enum": transition_types},
                "temperament_amplifier": {"type": "string", "minLength": 3},
            },
        }
    elif task == "G":
        min_chars, max_chars = _task_text_limits(settings, "G")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "interpretation_ko",
                "interpretation_en",
                "action_tendency",
                "confidence",
                "register",
                "misinterpretation_type",
                "temperament_bias",
            ],
            "properties": {
                "interpretation_ko": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                "interpretation_en": {"type": "string", "minLength": 5},
                "action_tendency": {"type": "string", "enum": action_tendencies},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "register": {"type": "string", "enum": [job.get("register")] if job.get("register") else list(DEFAULT_REGISTERS)},
                "misinterpretation_type": {"type": "string", "enum": misinterpretations},
                "temperament_bias": {"type": "string", "minLength": 3},
            },
        }
    elif task == "H":
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "name",
                "description_en",
                "resource_modifiers",
                "special_zones",
                "special_resources",
                "agent_modifiers",
            ],
            "properties": {
                "name": {"type": "string", "pattern": "^[A-Z][a-zA-Z]+$"},
                "description_en": {"type": "string", "minLength": 10},
                "resource_modifiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["target", "multiplier"],
                        "properties": {
                            "target": {"type": "string", "minLength": 1},
                            "multiplier": {"type": "number", "minimum": 0, "maximum": 5},
                        },
                    },
                },
                "special_zones": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "spawn_count_min", "spawn_count_max"],
                        "properties": {
                            "kind": {"type": "string", "minLength": 1},
                            "spawn_count_min": {"type": "integer", "minimum": 0, "maximum": 10},
                            "spawn_count_max": {"type": "integer", "minimum": 0, "maximum": 10},
                        },
                    },
                },
                "special_resources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "tags"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "tags": {"type": "array", "items": {"type": "string", "minLength": 1}},
                        },
                    },
                },
                "agent_modifiers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["system", "trigger", "effect"],
                        "properties": {
                            "system": {"type": "string", "minLength": 1},
                            "trigger": {"type": "string", "minLength": 1},
                            "effect": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
        }
    else:
        return None, extra_body

    return (
        {
            "type": "json_schema",
            "json_schema": {
                "name": f"worldsim_task_{task.lower()}",
                "strict": True,
                "schema": schema,
            },
        },
        extra_body,
    )


def call_teacher_api(*, job: dict, system_prompt: str, user_prompt: str, settings: dict) -> AttrDict:
    from openai import OpenAI

    provider = settings["provider"]
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set {provider['api_key_env']} before running generation")

    model = _task_teacher_model(settings, job.get("task", "")) or os.getenv(provider["model_env"], provider["default_model"])
    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    response_format, extra_body = build_response_format(job, settings)
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": provider.get("max_tokens", 128),
        "temperature": provider.get("temperature", 0.8),
        "extra_headers": {
            "HTTP-Referer": provider.get("request_headers", {}).get("referer", ""),
            "X-Title": provider.get("request_headers", {}).get("title", "WorldSim Training"),
        },
    }
    if response_format is not None:
        request_kwargs["response_format"] = response_format
    if extra_body is not None:
        request_kwargs["extra_body"] = extra_body

    try:
        response = client.chat.completions.create(**request_kwargs)
    except Exception:
        fallback = provider.get("response_format_fallback")
        if response_format is None or not fallback:
            raise
        fallback_kwargs = dict(request_kwargs)
        if fallback == "json_object":
            fallback_kwargs["response_format"] = {"type": "json_object"}
        else:
            fallback_kwargs.pop("response_format", None)
        response = client.chat.completions.create(**fallback_kwargs)
    return AttrDict(
        output=response.choices[0].message.content.strip(),
        usage=extract_usage(getattr(response, "usage", None)),
        model=model,
    )


def _append_jsonl(path: Path, row: dict) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _skip_output_path(output_path: Path) -> Path:
    return output_path.parent / "skipped.jsonl"


def _generation_retry_settings(settings: dict) -> tuple[int, float]:
    provider = settings.get("provider", {})
    attempts = max(1, int(provider.get("retry_attempts", 1)))
    backoff_seconds = float(provider.get("retry_backoff_seconds", 0))
    return attempts, backoff_seconds


def _enum_lookup_key(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    lowered = re.sub(r"[\s\"'`]+", "", lowered)
    lowered = re.sub(r"[^a-z가-힣_]+", "", lowered)
    return lowered or None


def _normalize_emotion_value(value: object) -> object:
    key = _enum_lookup_key(value)
    if key is None:
        return value
    return EMOTION_VALUE_MAP.get(key, value.strip().lower() if isinstance(value, str) else value)


def _normalize_register_value(value: object) -> object:
    key = _enum_lookup_key(value)
    if key is None:
        return value
    return REGISTER_VALUE_MAP.get(key, value.strip().lower() if isinstance(value, str) else value)


def normalize_generated_output(raw_text: object) -> object:
    payload = raw_text
    if isinstance(raw_text, str):
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return raw_text
    if not isinstance(payload, dict):
        return raw_text

    normalized = dict(payload)
    for field in ("emotion_expressed", "emotion", "previous_emotion"):
        if field in normalized:
            normalized[field] = _normalize_emotion_value(normalized[field])
    if "register" in normalized:
        normalized["register"] = _normalize_register_value(normalized["register"])
    return normalized


def parse_and_validate(raw_text: object, job: dict, settings: dict) -> tuple[str | None, str | None]:
    from scripts.validate_data import repair_and_validate_json_output

    candidate = {key: value for key, value in job.items() if key != "system_prompt"}
    normalized_output = normalize_generated_output(raw_text)
    if isinstance(normalized_output, (dict, list)):
        candidate["output"] = json.dumps(normalized_output, ensure_ascii=False, separators=(",", ":"))
    else:
        candidate["output"] = str(normalized_output)
    repaired_output, violations, _ = repair_and_validate_json_output(candidate, settings.get("validation", {}))
    if violations:
        return None, ",".join(violations)
    return repaired_output, None


def generate_dataset(
    repo_root: Path,
    *,
    generator=None,
    limit: int | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
    task_filter: set[str] | None = None,
    verbose: bool = True,
):
    load_local_env(repo_root)
    settings = load_generation_config(repo_root / "config")
    jobs = build_jobs(repo_root, seed=seed, task_filter=task_filter)
    selected_jobs = select_jobs(jobs, limit)
    output_path = _resolve_cli_output_path(repo_root, settings, output_path)
    skipped_path = _skip_output_path(output_path)
    ensure_directory(output_path.parent)
    output_path.write_text("", encoding="utf-8")
    skipped_path.write_text("", encoding="utf-8")
    totals: dict[str, float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    started_at = perf_counter()
    progress_every = int(reporting_settings(settings).get("progress_every", 10))
    retry_attempts, retry_backoff_seconds = _generation_retry_settings(settings)
    completed_rows = 0
    skipped_rows = 0

    for index, job in enumerate(selected_jobs, start=1):
        request_started_at = perf_counter()
        prompt = job.get("prompt") or f"[TASK] {job['task']}\n[VARIANT] {job.get('variant', 0)}"
        fallback_model = job.get("teacher_model") or _task_teacher_model(settings, job["task"]) or settings.get("provider", {}).get("default_model")
        normalized = None
        validated_output = None
        validation_error = None
        last_error = None
        for attempt in range(1, retry_attempts + 1):
            try:
                raw_result = (
                    generator(job, job["system_prompt"])
                    if generator is not None
                    else call_teacher_api(job=job, system_prompt=job["system_prompt"], user_prompt=prompt, settings=settings)
                )
                normalized = normalize_generation_result(raw_result, settings, fallback_model=fallback_model)
                validated_output, validation_error = parse_and_validate(normalized.output, job, settings)
                if validation_error is None:
                    break
                last_error = f"generation_validation_failed:{validation_error}"
                if attempt == retry_attempts:
                    break
                if verbose:
                    print(
                        f"[retry {attempt}/{retry_attempts - 1}] task={job['task']} variant={job.get('variant', 0)} "
                        f"validation={validation_error}",
                        flush=True,
                    )
            except Exception as exc:
                last_error = str(exc)
                if attempt == retry_attempts:
                    break
                if verbose:
                    print(
                        f"[retry {attempt}/{retry_attempts - 1}] task={job['task']} variant={job.get('variant', 0)}",
                        flush=True,
                    )
                if retry_backoff_seconds > 0:
                    sleep(retry_backoff_seconds * attempt)
        if normalized is None or validated_output is None:
            skipped_row = {key: value for key, value in job.items() if key != "system_prompt"}
            skipped_row["prompt"] = prompt
            skipped_row["skip_reason"] = last_error or "generation_failed_without_result"
            skipped_row["attempts"] = retry_attempts
            if normalized is not None:
                skipped_row["output"] = normalized.output
                skipped_row["model"] = normalized.model
                skipped_row["prompt_tokens"] = normalized.usage["prompt_tokens"]
                skipped_row["completion_tokens"] = normalized.usage["completion_tokens"]
                skipped_row["total_tokens"] = normalized.usage["total_tokens"]
                skipped_row["estimated_cost_usd"] = normalized.estimated_cost_usd
            _append_jsonl(skipped_path, skipped_row)
            skipped_rows += 1
            if verbose:
                print(
                    f"[skip] task={job['task']} variant={job.get('variant', 0)} reason={skipped_row['skip_reason']}",
                    flush=True,
                )
            continue
        request_elapsed = perf_counter() - request_started_at
        row = {key: value for key, value in job.items() if key != "system_prompt"}
        row["output"] = validated_output
        row["model"] = normalized.model
        row["prompt_tokens"] = normalized.usage["prompt_tokens"]
        row["completion_tokens"] = normalized.usage["completion_tokens"]
        row["total_tokens"] = normalized.usage["total_tokens"]
        row["estimated_cost_usd"] = normalized.estimated_cost_usd
        _append_jsonl(output_path, row)
        completed_rows += 1

        totals["prompt_tokens"] += normalized.usage["prompt_tokens"]
        totals["completion_tokens"] += normalized.usage["completion_tokens"]
        totals["total_tokens"] += normalized.usage["total_tokens"]
        totals["estimated_cost_usd"] += normalized.estimated_cost_usd

        if verbose and progress_every > 0 and (index % progress_every == 0 or index == len(selected_jobs)):
            print_progress(
                index=index,
                total=len(selected_jobs),
                elapsed_seconds=perf_counter() - started_at,
                totals=totals,
                last_request_seconds=request_elapsed,
            )

    elapsed_seconds = perf_counter() - started_at
    result = AttrDict(
        output_path=output_path,
        skipped_path=skipped_path,
        count=completed_rows,
        record_count=completed_rows,
        jobs_count=len(selected_jobs),
        skipped_count=skipped_rows,
        prompt_tokens=int(totals["prompt_tokens"]),
        completion_tokens=int(totals["completion_tokens"]),
        total_tokens=int(totals["total_tokens"]),
        estimated_cost_usd=round(totals["estimated_cost_usd"], 8),
        elapsed_seconds=elapsed_seconds,
        rows_per_second=(completed_rows / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        tokens_per_second=(totals["total_tokens"] / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
    )
    if verbose:
        print_final_summary(result=result)
    return result


def generate_records(
    *,
    jobs: list[dict],
    prompt_assets: dict,
    settings: dict,
    output_file: Path,
    limit: int | None = None,
) -> dict:
    selected_jobs = select_jobs(jobs, limit)
    rows: list[dict] = []

    for job in selected_jobs:
        user_prompt = render_prompt(job, prompt_assets)
        normalized = call_teacher_api(
            job=job,
            system_prompt=prompt_assets["system"],
            user_prompt=user_prompt,
            settings=settings,
        )
        validated_output, validation_error = parse_and_validate(normalized.output, job, settings)
        if validation_error is not None:
            raise RuntimeError(f"generation_validation_failed:{validation_error}")
        row = {
            **job,
            "prompt": user_prompt,
            "output": validated_output,
            "model": normalized.model,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        rows.append(row)

    ensure_directory(output_file.parent)
    output_file.write_text("", encoding="utf-8")
    for row in rows:
        _append_jsonl(output_file, row)
    return {"output_file": str(output_file), "count": len(rows)}


def print_job_summary(jobs: list[dict]) -> None:
    counts: dict[str, int] = {}
    for job in jobs:
        counts[job["task"]] = counts.get(job["task"], 0) + 1
    print(f"Total jobs: {len(jobs)}")
    for task in sorted(counts):
        print(f"  Task {task}: {counts[task]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WorldSim training data with OpenRouter.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", default=None, help="Comma-separated task ids, e.g. E,F")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_path(Path.cwd(), args.repo_root)
    settings = load_generation_config(repo_root / "config")
    task_filter = parse_task_filter(args.tasks)
    jobs = build_jobs(repo_root, seed=args.seed, task_filter=task_filter)
    output_path = _resolve_cli_output_path(repo_root, settings, args.output)
    print_job_summary(select_jobs(jobs, args.limit))
    if args.dry_run:
        print(f"Dry run only. Raw output would be written to {output_path}")
        return

    result = generate_dataset(repo_root, limit=args.limit, seed=args.seed, output_path=output_path, task_filter=task_filter)
    print(f"Wrote {result.count} rows to {result.output_path}")
    if result.skipped_count:
        print(f"Skipped {result.skipped_count} rows to {result.skipped_path}")


if __name__ == "__main__":
    main()
