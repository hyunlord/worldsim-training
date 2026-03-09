from __future__ import annotations

import re

CANONICAL_EMOTIONS = (
    "joy",
    "sadness",
    "fear",
    "anger",
    "trust",
    "disgust",
    "surprise",
    "anticipation",
)
CANONICAL_REGISTERS = ("haera", "hao", "hae")

_EMOTION_MAP = {
    "joy": "joy",
    "happy": "joy",
    "happiness": "joy",
    "기쁨": "joy",
    "즐거움": "joy",
    "sadness": "sadness",
    "sad": "sadness",
    "sorrow": "sadness",
    "슬픔": "sadness",
    "fear": "fear",
    "afraid": "fear",
    "fright": "fear",
    "terror": "fear",
    "공포": "fear",
    "두려움": "fear",
    "겁": "fear",
    "anger": "anger",
    "angry": "anger",
    "rage": "anger",
    "분노": "anger",
    "trust": "trust",
    "belief": "trust",
    "faith": "trust",
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
    "expectancy": "anticipation",
    "기대": "anticipation",
}
_REGISTER_MAP = {
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
_EDGE_PUNCTUATION = "\"'“”‘’`.,!?;:()[]{}"


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def cleanup_punctuation(value: str) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?]){2,}", r"\1", cleaned)
    return cleaned.strip()


def normalize_text(value: str) -> str:
    return cleanup_punctuation(value)


def _normalize_enum_token(value: str) -> str:
    token = normalize_whitespace(value).strip(_EDGE_PUNCTUATION).lower()
    token = token.replace("-", "_")
    token = re.sub(r"\s+", "_", token)
    return token


def normalize_emotion(value: str) -> str | None:
    token = _normalize_enum_token(value)
    if not token:
        return None
    return _EMOTION_MAP.get(token)


def normalize_register(value: str) -> str | None:
    token = _normalize_enum_token(value)
    if not token:
        return None
    return _REGISTER_MAP.get(token)


def strip_outer_quotes(value: str) -> tuple[str, bool]:
    cleaned = normalize_text(value)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        return cleaned[1:-1].strip(), True
    return cleaned, False
