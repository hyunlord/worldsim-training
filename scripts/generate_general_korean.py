#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import write_jsonl

FLOWERS = [
    ("봄에 피는 꽃 세 가지를 알려줘.", "진달래, 개나리, 벚꽃이 봄에 핀다."),
    ("비 온 뒤 나는 냄새를 한 문장으로 말해줘.", "비 그친 뒤 젖은 흙내가 잔잔히 올라온다."),
    ("강가 저녁빛을 한 문장으로 그려줘.", "강가에 붉은 빛이 번지며 물결이 잔잔히 반짝인다."),
    ("겨울 새벽 공기를 한 문장으로 적어줘.", "겨울 새벽 공기는 살을 에듯 차갑고 맑다."),
]
CREATIVE_SUBJECTS = [
    "가을 하늘",
    "산마루 바람",
    "새벽 안개",
    "멧새 울음",
    "솔숲 그림자",
    "달빛 강물",
]
PROVERBS = [
    ("돌다리도 두들겨 보고 건너라", "아무리 익숙한 길도 다시 살펴보고 가라는 뜻이다."),
    ("가는 말이 고와야 오는 말이 곱다", "내가 고운 말을 해야 남도 고운 말로 답한다는 뜻이다."),
    ("티끌 모아 태산", "작은 것도 차곡차곡 모이면 크게 불어난다는 뜻이다."),
]
INSTRUCTION_CASES = [
    ("다음 낱말을 세 글자로 줄여줘: 가을하늘", "가을빛"),
    ("다음 말을 반대로 바꿔줘: 깊은 밤", "밝은 낮"),
    ("다음 낱말을 두 낱말로 나눠줘: 물안개", "물 안개"),
]
JSON_CASES = [
    (
        "다음 낱말을 JSON으로 묶어라: 풀, 물, 불",
        {"풀": "푸른 풀", "물": "맑은 물", "불": "붉은 불"},
    ),
    (
        "다음 셋을 JSON으로 적어라: 아침, 낮, 밤",
        {"아침": "해 돋을 무렵", "낮": "해가 높이 뜬 때", "밤": "해가 지고 난 뒤"},
    ),
]

CATEGORY_TARGETS = {
    "qa": 100,
    "creative": 80,
    "json_formatting": 60,
    "proverb": 30,
    "instruction": 30,
}


def _row(category: str, prompt: str, output: object) -> dict:
    return {
        "task": "GEN",
        "label": "retain",
        "category": category,
        "prompt": prompt,
        "output": output,
    }


def _build_qa_rows(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    for _ in range(count):
        prompt, output = rng.choice(FLOWERS)
        rows.append(_row("qa", prompt, output))
    return rows


def _build_creative_rows(rng: random.Random, count: int) -> list[dict]:
    patterns = [
        "{subject}을 한 문장으로 묘사해.",
        "{subject}을 짧게 그려줘.",
        "{subject}을 순우리말 한 문장으로 적어줘.",
    ]
    endings = [
        "{subject}에 맑은 빛이 어리고 바람이 가볍게 스쳐 간다.",
        "{subject} 위로 잔잔한 기운이 번지며 숨결이 느릿하게 흐른다.",
        "{subject} 둘레에 고운 빛이 어려 마음이 차분히 가라앉는다.",
    ]
    rows: list[dict] = []
    for _ in range(count):
        subject = rng.choice(CREATIVE_SUBJECTS)
        prompt = rng.choice(patterns).format(subject=subject)
        output = rng.choice(endings).format(subject=subject)
        rows.append(_row("creative", prompt, output))
    return rows


def _build_json_rows(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    for _ in range(count):
        prompt, output = rng.choice(JSON_CASES)
        rows.append(_row("json_formatting", prompt, output))
    return rows


def _build_proverb_rows(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    for _ in range(count):
        proverb, meaning = rng.choice(PROVERBS)
        rows.append(_row("proverb", f'"{proverb}"의 뜻을 풀어줘.', meaning))
    return rows


def _build_instruction_rows(rng: random.Random, count: int) -> list[dict]:
    rows: list[dict] = []
    for _ in range(count):
        prompt, output = rng.choice(INSTRUCTION_CASES)
        rows.append(_row("instruction", prompt, output))
    return rows


def build_general_korean_examples(*, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = [
        *_build_qa_rows(rng, CATEGORY_TARGETS["qa"]),
        *_build_creative_rows(rng, CATEGORY_TARGETS["creative"]),
        *_build_json_rows(rng, CATEGORY_TARGETS["json_formatting"]),
        *_build_proverb_rows(rng, CATEGORY_TARGETS["proverb"]),
        *_build_instruction_rows(rng, CATEGORY_TARGETS["instruction"]),
    ]
    rng.shuffle(rows)
    return rows[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate template-based general Korean retention examples.")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--output", default="data/samples/general_korean.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_general_korean_examples(count=args.count, seed=args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(f"Generated {len(rows)} general Korean examples -> {output_path}")


if __name__ == "__main__":
    main()
