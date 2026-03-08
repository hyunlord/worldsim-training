#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import AttrDict, ensure_directory, load_text, load_yaml, resolve_path, write_jsonl

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
    for task in ("A", "B", "C", "D"):
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
    for task in ("A", "B", "C", "D"):
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

    return jobs


def build_jobs(catalogs_or_repo_root: dict | Path, settings: dict | None = None, seed: int | None = None) -> list[dict]:
    if isinstance(catalogs_or_repo_root, Path):
        repo_root = catalogs_or_repo_root
        repo_settings = load_generation_config(repo_root / "config")
        catalogs = load_catalogs(repo_root / "config")
        prompt_assets = _repo_prompt_assets(repo_root, repo_settings)
        jobs = _build_jobs_from_catalogs(catalogs, repo_settings, system_prompt=prompt_assets["system"], seed=seed)
        if prompt_assets["tasks"]:
            for job in jobs:
                job["prompt"] = render_prompt(job, prompt_assets)
        return jobs
    if settings is None:
        raise ValueError("settings must be provided when building jobs from catalogs")
    return _build_jobs_from_catalogs(catalogs_or_repo_root, settings, seed=seed)


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
        name=job.get("name", ""),
        variant=job.get("variant", 0),
    )


def build_output_path(output_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(output_dir) / f"generated_{timestamp}.jsonl"


def default_raw_output_path(repo_root: Path, stamp: str | None = None) -> Path:
    settings = load_generation_config(repo_root / "config")
    raw_dir = resolve_path(repo_root, settings.get("paths", {}).get("raw_dir", "data/raw"))
    ensure_directory(raw_dir)
    if stamp is None:
        return build_output_path(raw_dir)
    return raw_dir / f"generated_{stamp}.jsonl"


def call_teacher_api(*, system_prompt: str, user_prompt: str, settings: dict) -> str:
    from openai import OpenAI

    provider = settings["provider"]
    api_key = os.getenv(provider["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set {provider['api_key_env']} before running generation")

    model = os.getenv(provider["model_env"], provider["default_model"])
    client = OpenAI(api_key=api_key, base_url=provider["base_url"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=provider.get("max_tokens", 128),
        temperature=provider.get("temperature", 0.8),
        extra_headers={
            "HTTP-Referer": provider.get("request_headers", {}).get("referer", ""),
            "X-Title": provider.get("request_headers", {}).get("title", "WorldSim Training"),
        },
    )
    return response.choices[0].message.content.strip()


def generate_dataset(
    repo_root: Path,
    *,
    generator=None,
    limit: int | None = None,
    seed: int | None = None,
    output_path: Path | None = None,
):
    load_local_env(repo_root)
    settings = load_generation_config(repo_root / "config")
    jobs = build_jobs(repo_root, seed=seed)
    selected_jobs = jobs[:limit] if limit is not None else jobs
    output_path = output_path or default_raw_output_path(repo_root)
    rows: list[dict] = []

    for job in selected_jobs:
        prompt = job.get("prompt") or f"[TASK] {job['task']}\n[VARIANT] {job.get('variant', 0)}"
        output_text = (
            generator(job, job["system_prompt"])
            if generator is not None
            else call_teacher_api(system_prompt=job["system_prompt"], user_prompt=prompt, settings=settings)
        )
        row = {key: value for key, value in job.items() if key != "system_prompt"}
        row["output"] = output_text
        rows.append(row)

    write_jsonl(output_path, rows)
    return AttrDict(output_path=output_path, count=len(rows), record_count=len(rows), jobs_count=len(selected_jobs))


def generate_records(
    *,
    jobs: list[dict],
    prompt_assets: dict,
    settings: dict,
    output_file: Path,
    limit: int | None = None,
) -> dict:
    selected_jobs = jobs[:limit] if limit else jobs
    rows: list[dict] = []

    for job in selected_jobs:
        user_prompt = render_prompt(job, prompt_assets)
        output_text = call_teacher_api(
            system_prompt=prompt_assets["system"],
            user_prompt=user_prompt,
            settings=settings,
        )
        row = {
            **job,
            "prompt": user_prompt,
            "output": output_text,
            "model": os.getenv(settings["provider"]["model_env"], settings["provider"]["default_model"]),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        rows.append(row)

    write_jsonl(output_file, rows)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_path(Path.cwd(), args.repo_root)
    jobs = build_jobs(repo_root, seed=args.seed)
    output_path = resolve_path(repo_root, args.output) if args.output else default_raw_output_path(repo_root)
    print_job_summary(jobs[: args.limit] if args.limit else jobs)
    if args.dry_run:
        print(f"Dry run only. Raw output would be written to {output_path}")
        return

    result = generate_dataset(repo_root, limit=args.limit, seed=args.seed, output_path=output_path)
    print(f"Wrote {result.count} rows to {result.output_path}")


if __name__ == "__main__":
    main()
