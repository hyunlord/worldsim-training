from __future__ import annotations

import re


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_SINGLE_QUOTE_KEY_PATTERN = re.compile(r"'([^'\\]+)'\s*:")
_SINGLE_QUOTE_VALUE_PATTERN = re.compile(r':\s*\'([^\'\\]*)\'')
_UNQUOTED_VALUE_PATTERN = re.compile(r'("[\w]+")\s*:\s*([a-zA-Z_][\w]*)\s*([,}\]])')
_RESERVED_BAREWORDS = {"true", "false", "null"}


def strip_markdown_fences(text: str) -> tuple[str, bool]:
    match = _FENCE_PATTERN.match(text.strip())
    if match is None:
        return text, False
    return match.group(1).strip(), True


def extract_first_json_object(text: str) -> tuple[str, bool]:
    start = -1
    open_char = ""
    close_char = ""
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char in ("{", "[") and start == -1:
            start = index
            open_char = char
            close_char = "}" if char == "{" else "]"
            depth = 1
            continue

        if start != -1:
            if char == open_char:
                depth += 1
            elif char == close_char:
                depth -= 1
                if depth == 0:
                    extracted = text[start : index + 1]
                    return (extracted, extracted != text)

    return text, False


def remove_trailing_commas(text: str) -> tuple[str, bool]:
    repaired = _TRAILING_COMMA_PATTERN.sub(r"\1", text)
    return repaired, repaired != text


def replace_single_quotes_if_safe(text: str) -> tuple[str, bool]:
    if '"' in text:
        return text, False

    repaired = _SINGLE_QUOTE_KEY_PATTERN.sub(r'"\1":', text)
    repaired = _SINGLE_QUOTE_VALUE_PATTERN.sub(r': "\1"', repaired)
    return repaired, repaired != text


def fix_unquoted_values(text: str) -> tuple[str, bool]:
    def replacer(match: re.Match[str]) -> str:
        key_part = match.group(1)
        value = match.group(2)
        suffix = match.group(3)
        if value.lower() in _RESERVED_BAREWORDS:
            return f"{key_part}: {value}{suffix}"
        return f'{key_part}: "{value}"{suffix}'

    repaired = _UNQUOTED_VALUE_PATTERN.sub(replacer, text)
    return repaired, repaired != text


def fix_missing_closing_braces(text: str) -> tuple[str, bool]:
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            open_braces += 1
        elif char == "}":
            open_braces -= 1
        elif char == "[":
            open_brackets += 1
        elif char == "]":
            open_brackets -= 1

    repaired = text
    applied = False
    if open_braces == 1:
        repaired += "}"
        applied = True
    if open_brackets == 1:
        repaired += "]"
        applied = True
    return repaired, applied


def repair_json(raw_text: str) -> tuple[str, list[str]]:
    repairs: list[str] = []
    text = str(raw_text)

    for repair_name, repair_fn in (
        ("fence_strip", strip_markdown_fences),
        ("first_json_extract", extract_first_json_object),
        ("trailing_comma", remove_trailing_commas),
        ("single_to_double_quotes", replace_single_quotes_if_safe),
        ("unquoted_value_fix", fix_unquoted_values),
        ("missing_closing_brace", fix_missing_closing_braces),
    ):
        text, applied = repair_fn(text)
        if applied:
            repairs.append(repair_name)

    return text, repairs
