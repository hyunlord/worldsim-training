#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, sleep

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, load_text, load_yaml, resolve_path

DEFAULT_REGISTERS = {
    "haera": "해라체로 써라. 문장을 -다, -는다, -았다, -었다 로 끝내라.",
    "hao": "하오체로 써라. 문장을 -오, -소, -시오 로 끝내라.",
    "hae": "해체로 써라. 문장을 -해, -야, -지, -어 로 끝내라.",
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
    return {
        "situations": load_yaml(config_dir / "situations.yaml").get("situations", []),
        "personalities": load_yaml(config_dir / "personalities.yaml").get("personalities", []),
        "emotions": load_yaml(config_dir / "emotions.yaml").get("emotions", []),
    }


def load_prompt_assets(prompts_dir: Path) -> dict:
    teacher_dir = prompts_dir / "teacher"
    assets = {"system": load_text(teacher_dir / "system.txt"), "tasks": {}}
    for task in ("A", "B", "C", "D", "E", "F"):
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
    for task in ("A", "B", "C", "D", "E", "F"):
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
    names = settings.get("names") or settings.get("generation", {}).get("task_d_names", ["돌이"])
    registers = _register_instructions(settings)
    default_register = settings.get("defaults", {}).get("register", "haera")
    jobs: list[dict] = []

    for personality in personalities:
        for variant in range(_variant_count(settings, "A")):
            jobs.append(
                {
                    "task": "A",
                    "layer": "L4",
                    "expected_format": "text",
                    "variant": variant,
                    "personality_id": personality["id"],
                    "personality_name": personality.get("ko", personality["id"]),
                    "personality_desc": personality.get("desc", ""),
                    "personality_keywords": personality.get("keywords", []),
                    "keywords": ", ".join(personality.get("keywords", [])),
                    "register": "haera",
                    "register_instruction": registers["haera"],
                    "system_prompt": system_prompt,
                }
            )

    for situation in situations:
        for emotion in emotions:
            for personality in personalities:
                for variant in range(_variant_count(settings, "B")):
                    intensity = rng.choice(emotion.get("intensities", [0.6]))
                    jobs.append(
                        {
                            "task": "B",
                            "layer": "L4",
                            "expected_format": "text",
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
                            "situation_id": situation["id"],
                            "scenario_name": situation.get("ko", situation["id"]),
                            "scenario_desc": situation.get("desc", ""),
                            "situation": situation.get("ko", situation["id"]),
                            "system_prompt": system_prompt,
                        }
                    )

    for situation in situations:
        for personality in personalities:
            for variant in range(_variant_count(settings, "C")):
                emotion = rng.choice(emotions) if emotions else {"id": "neutral", "ko": "담담함"}
                register = personality.get("default_register", default_register)
                jobs.append(
                    {
                        "task": "C",
                        "layer": "L4",
                        "expected_format": "text",
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
                        "situation_id": situation["id"],
                        "scenario_name": situation.get("ko", situation["id"]),
                        "scenario_desc": situation.get("desc", ""),
                        "situation": situation.get("ko", situation["id"]),
                        "system_prompt": system_prompt,
                    }
                )

    for situation in situations:
        for variant in range(_variant_count(settings, "D")):
            jobs.append(
                {
                    "task": "D",
                    "layer": "L4",
                    "expected_format": "text",
                    "variant": variant,
                    "name": names[variant % len(names)],
                    "register": "haera",
                    "register_instruction": registers["haera"],
                    "situation_id": situation["id"],
                    "scenario_name": situation.get("ko", situation["id"]),
                    "scenario_desc": situation.get("desc", ""),
                    "situation": situation.get("ko", situation["id"]),
                    "system_prompt": system_prompt,
                }
            )

    for situation in situations:
        action_options = situation.get("action_options") or situation.get("typical_actions") or []
        if not action_options:
            continue
        for personality_index, personality in enumerate(personalities):
            for variant in range(_variant_count(settings, "E")):
                emotion = rng.choice(emotions) if emotions else {"id": "fear", "ko": "두려움", "intensities": [0.7]}
                intensity = rng.choice(emotion.get("intensities", [0.7]))
                options_line = " ".join(f"{idx}:{option}" for idx, option in enumerate(action_options))
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
                        "situation_id": situation["id"],
                        "scenario_name": situation.get("ko", situation["id"]),
                        "scenario_desc": situation.get("desc", ""),
                        "situation": situation.get("ko", situation["id"]),
                        "action_options": action_options,
                        "options_line": options_line,
                        "system_prompt": system_prompt,
                    }
                )

    for situation in situations:
        for personality_index, personality in enumerate(personalities):
            for current_emotion in emotions:
                for variant in range(_variant_count(settings, "F")):
                    current_intensity = rng.choice(current_emotion.get("intensities", [0.5]))
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
                            "situation_id": situation["id"],
                            "scenario_name": situation.get("ko", situation["id"]),
                            "scenario_desc": situation.get("desc", ""),
                            "situation": situation.get("ko", situation["id"]),
                            "system_prompt": system_prompt,
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
    return template.format(
        personality_name=job.get("personality_name", ""),
        personality_desc=job.get("personality_desc", ""),
        personality_keywords=", ".join(job.get("personality_keywords", [])),
        keywords=job.get("keywords", ", ".join(job.get("personality_keywords", []))),
        emotion_name=job.get("emotion_name", ""),
        emotion=job.get("emotion", job.get("emotion_name", "")),
        emotion_intensity=job.get("emotion_intensity", ""),
        intensity=job.get("intensity", job.get("emotion_intensity", "")),
        mimetic=job.get("mimetic", ""),
        scenario_name=job.get("scenario_name", ""),
        scenario_desc=job.get("scenario_desc", ""),
        situation=job.get("scenario_desc") or job.get("scenario_name", ""),
        register_instruction=job.get("register_instruction", ""),
        register=job.get("register", ""),
        action_options=job.get("options_line", job.get("action_options", "")),
        options_line=job.get("options_line", ""),
        current_emotion_name=job.get("current_emotion_name", ""),
        current_emotion_intensity=job.get("current_emotion_intensity", ""),
        name=job.get("name", ""),
        variant=job.get("variant", 0),
    )


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


def default_raw_output_path(repo_root: Path, stamp: str | None = None) -> Path:
    settings = load_generation_config(repo_root / "config")
    raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
    ensure_directory(raw_dir)
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
        usage = extract_usage(result.get("usage"))
        model = result.get("model") or fallback_model or settings.get("provider", {}).get("default_model")
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
    print(f"  rows={result.count}", flush=True)
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

    if job.get("task") == "E":
        min_chars, max_chars = _task_text_limits(settings, "E")
        action_options = job.get("action_options", [])
        return (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "worldsim_task_e_action",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["action_id", "confidence", "hint"],
                        "properties": {
                            "action_id": {"type": "integer", "enum": list(range(len(action_options)))},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "hint": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                        },
                    },
                },
            },
            extra_body,
        )

    if job.get("task") == "F":
        min_chars, max_chars = _task_text_limits(settings, "F")
        valid_emotions = (
            settings.get("validation", {})
            .get("layer3_json", {})
            .get("task_f", {})
            .get(
                "valid_emotions",
                ["joy", "sadness", "fear", "anger", "trust", "disgust", "surprise", "anticipation"],
            )
        )
        return (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "worldsim_task_f_emotion",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["emotion", "intensity", "cause"],
                        "properties": {
                            "emotion": {"type": "string", "enum": list(valid_emotions)},
                            "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                            "cause": {"type": "string", "minLength": min_chars, "maxLength": max_chars},
                        },
                    },
                },
            },
            extra_body,
        )

    return None, extra_body


def call_teacher_api(*, job: dict, system_prompt: str, user_prompt: str, settings: dict) -> AttrDict:
    from openai import OpenAI

    provider = settings["provider"]
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set {provider['api_key_env']} before running generation")

    model = os.getenv(provider["model_env"], provider["default_model"])
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

    response = client.chat.completions.create(**request_kwargs)
    return AttrDict(
        output=response.choices[0].message.content.strip(),
        usage=extract_usage(getattr(response, "usage", None)),
        model=model,
    )


def _append_jsonl(path: Path, row: dict) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generation_retry_settings(settings: dict) -> tuple[int, float]:
    provider = settings.get("provider", {})
    attempts = max(1, int(provider.get("retry_attempts", 1)))
    backoff_seconds = float(provider.get("retry_backoff_seconds", 0))
    return attempts, backoff_seconds


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
    output_path = output_path or default_raw_output_path(repo_root)
    ensure_directory(output_path.parent)
    output_path.write_text("", encoding="utf-8")
    totals: dict[str, float] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    started_at = perf_counter()
    progress_every = int(reporting_settings(settings).get("progress_every", 10))
    provider_default_model = settings.get("provider", {}).get("default_model")
    retry_attempts, retry_backoff_seconds = _generation_retry_settings(settings)
    completed_rows = 0

    for index, job in enumerate(selected_jobs, start=1):
        request_started_at = perf_counter()
        prompt = job.get("prompt") or f"[TASK] {job['task']}\n[VARIANT] {job.get('variant', 0)}"
        for attempt in range(1, retry_attempts + 1):
            try:
                raw_result = (
                    generator(job, job["system_prompt"])
                    if generator is not None
                    else call_teacher_api(job=job, system_prompt=job["system_prompt"], user_prompt=prompt, settings=settings)
                )
                break
            except Exception:
                if attempt == retry_attempts:
                    raise
                if verbose:
                    print(
                        f"[retry {attempt}/{retry_attempts - 1}] task={job['task']} variant={job.get('variant', 0)}",
                        flush=True,
                    )
                if retry_backoff_seconds > 0:
                    sleep(retry_backoff_seconds * attempt)
        normalized = normalize_generation_result(raw_result, settings, fallback_model=provider_default_model)
        request_elapsed = perf_counter() - request_started_at
        row = {key: value for key, value in job.items() if key != "system_prompt"}
        row["output"] = normalized.output
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
        count=completed_rows,
        record_count=completed_rows,
        jobs_count=len(selected_jobs),
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
        row = {
            **job,
            "prompt": user_prompt,
            "output": normalized.output,
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
    task_filter = parse_task_filter(args.tasks)
    jobs = build_jobs(repo_root, seed=args.seed, task_filter=task_filter)
    output_path = resolve_path(repo_root, args.output) if args.output else default_raw_output_path(repo_root)
    print_job_summary(select_jobs(jobs, args.limit))
    if args.dry_run:
        print(f"Dry run only. Raw output would be written to {output_path}")
        return

    result = generate_dataset(repo_root, limit=args.limit, seed=args.seed, output_path=output_path, task_filter=task_filter)
    print(f"Wrote {result.count} rows to {result.output_path}")


if __name__ == "__main__":
    main()
