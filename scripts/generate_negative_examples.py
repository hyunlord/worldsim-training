#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import write_jsonl

REPETITION_WORDS = ["꼼꼼하게", "조심스럽게", "두려워하며", "살피며", "기다리며"]
SINO_KOREAN_PAIRS = [
    ("식량", "먹거리"),
    ("전투", "싸움"),
    ("준비", "마련"),
    ("동맹", "손잡기"),
    ("이동", "옮겨가기"),
    ("수집", "모으기"),
    ("공격", "덤비기"),
    ("방어", "막아서기"),
    ("도구", "연장"),
    ("건축", "집짓기"),
    ("생존", "살아남기"),
    ("위험", "아슬아슬한 것"),
]
REGISTER_MIXES = [
    "앞으로 나가시오. 곧 달려든다.",
    "내가 먼저 살피겠소. 그러니 몸을 낮춰라.",
    "모두 숨으시오. 곧 불을 지켜야 해.",
]
EXCESSIVE_LENGTH_SENTENCES = [
    "먼저 발자국을 훑어보고 바람결을 읽었다.",
    "해가 기울기 전에 먹거리 흔적도 다시 더듬었다.",
    "무리의 발걸음까지 하나하나 살펴보았다.",
    "숨을 죽인 채 나뭇잎 떨림도 오래 헤아렸다.",
    "돌 하나를 뒤집고도 다시 먼 쪽 숲그늘을 보았다.",
    "마지막에는 불씨 자리까지 돌아가며 다시 살폈다.",
]
MODERN_WORDS = ["OK", "알겠어", "바로", "진짜", "완전", "대박", "레벨", "스킬"]
SCHEMA_LEAK_KEYS = ["WORLD", "WORLD_DESC", "WORLD_VOCAB", "schema_explanation", "text_to_generate"]
SOCIAL_FIELDS = ["temperament_tone_ko", "temperament_tone_en", "tone_hint", "temperament_analysis_ko"]
ENGLISH_KO_LINES = [
    "I move at dawn.",
    "Wait and defend.",
    "Trust the hunter.",
    "The sign means retreat.",
]

CATEGORY_TARGETS = {
    "repetition_loop": 100,
    "forbidden_sino_korean": 100,
    "register_mixing": 80,
    "excessive_length": 50,
    "modern_vocabulary": 50,
    "schema_leakage": 50,
    "key_hallucination": 40,
    "english_in_ko_field": 30,
}


def _base_row(*, source_task: str, reason: str, output: object, corrected_output: object) -> dict:
    return {
        "task": "NEG",
        "label": "reject",
        "source_task": source_task,
        "output": output,
        "reason": reason,
        "corrected_output": corrected_output,
    }


def generate_repetition_example(rng: random.Random) -> dict:
    word = rng.choice(REPETITION_WORDS)
    repeated = " ".join([word] * rng.randint(10, 30))
    return _base_row(
        source_task="B",
        reason="repetition_loop",
        output=repeated,
        corrected_output="벌벌 떨며 둘레를 살폈다. 몸을 낮추고 숨을 죽였다.",
    )


def generate_sino_korean_example(rng: random.Random) -> dict:
    sino, pure = rng.choice(SINO_KOREAN_PAIRS)
    tail = rng.choice(["하였다.", "준비했다.", "시작했다."])
    return {
        **_base_row(
            source_task="B",
            reason="forbidden_sino_korean",
            output=f"{sino}을 {tail}",
            corrected_output=f"{pure}를 챙겼다.",
        ),
        "correction_hint": f"{sino} -> {pure}",
    }


def generate_register_mixing_example(rng: random.Random) -> dict:
    return _base_row(
        source_task="C",
        reason="register_mixing",
        output=rng.choice(REGISTER_MIXES),
        corrected_output="곧 앞으로 나선다. 모두 몸을 낮춘다.",
    )


def generate_excessive_length_example(rng: random.Random) -> dict:
    sentence_count = rng.randint(5, 7)
    text = " ".join(EXCESSIVE_LENGTH_SENTENCES[:sentence_count])
    return _base_row(
        source_task="A",
        reason="excessive_length",
        output=text,
        corrected_output="바람결을 읽고 발자국을 살폈다.",
    )


def generate_modern_vocabulary_example(rng: random.Random) -> dict:
    modern = rng.choice(MODERN_WORDS)
    return _base_row(
        source_task="F",
        reason="modern_vocabulary",
        output=f"{modern}, 바로 움직이자.",
        corrected_output="곧 몸을 일으켜 발을 옮긴다.",
    )


def generate_schema_leakage_example(rng: random.Random) -> dict:
    key = rng.choice(SCHEMA_LEAK_KEYS)
    leaked = {
        key: "copied prompt section",
        "text_ko": "숨을 죽이고 풀숲을 살폈다.",
        "text_en": "They hushed themselves and scanned the brush.",
    }
    return _base_row(
        source_task="B",
        reason="schema_leakage",
        output=leaked,
        corrected_output={
            "text_ko": "숨을 죽이고 풀숲을 살폈다.",
            "text_en": "They hushed themselves and scanned the brush.",
        },
    )


def generate_key_hallucination_example(rng: random.Random) -> dict:
    count = rng.randint(5, 12)
    payload = {f"{rng.choice(SOCIAL_FIELDS)}_{index}": "haera" for index in range(1, count + 1)}
    payload["speech_ko"] = "앞을 살핀다."
    return _base_row(
        source_task="C",
        reason="key_hallucination",
        output=payload,
        corrected_output={
            "speech_ko": "앞을 살핀다.",
            "speech_en": "They watch the front.",
            "register": "haera",
            "emotion_expressed": "fear",
            "speaker_role": "scout",
            "temperament_tone": "watchful and clipped",
        },
    )


def generate_english_in_ko_field_example(rng: random.Random) -> dict:
    return _base_row(
        source_task="G",
        reason="english_in_ko_field",
        output={
            "interpretation_ko": rng.choice(ENGLISH_KO_LINES),
            "interpretation_en": "This means the band should wait behind the ridge.",
        },
        corrected_output={
            "interpretation_ko": "이 말은 등성이 뒤에 숨어 때를 기다리라고 여긴다.",
            "interpretation_en": "This means the band should hide behind the ridge and wait.",
        },
    )


GENERATOR_SPECS = [
    ("repetition_loop", generate_repetition_example),
    ("forbidden_sino_korean", generate_sino_korean_example),
    ("register_mixing", generate_register_mixing_example),
    ("excessive_length", generate_excessive_length_example),
    ("modern_vocabulary", generate_modern_vocabulary_example),
    ("schema_leakage", generate_schema_leakage_example),
    ("key_hallucination", generate_key_hallucination_example),
    ("english_in_ko_field", generate_english_in_ko_field_example),
]


def build_negative_examples(*, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for category, generator in GENERATOR_SPECS:
        target = CATEGORY_TARGETS[category]
        for _ in range(target):
            row = generator(rng)
            row["category"] = category
            rows.append(row)
    rng.shuffle(rows)
    return rows[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic negative examples for contrastive training.")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--output", default="data/samples/negative_examples.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_negative_examples(count=args.count, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(f"Generated {len(rows)} negative examples -> {output_path}")


if __name__ == "__main__":
    main()
