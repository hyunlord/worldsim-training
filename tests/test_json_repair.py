from __future__ import annotations

from training.lib.json_repair import (
    extract_first_json_object,
    fix_missing_closing_braces,
    fix_unquoted_values,
    remove_trailing_commas,
    repair_json,
    strip_markdown_fences,
)


def test_strip_markdown_fences_handles_json_and_plain_fences() -> None:
    repaired, applied = strip_markdown_fences("```json\n{\"key\": \"value\"}\n```")
    assert applied is True
    assert repaired == '{"key": "value"}'

    repaired, applied = strip_markdown_fences("```\n{\"key\": \"value\"}\n```")
    assert applied is True
    assert repaired == '{"key": "value"}'


def test_extract_first_json_object_from_mixed_text() -> None:
    repaired, applied = extract_first_json_object('Here is the output: {"key":"value"} Let me know')
    assert applied is True
    assert repaired == '{"key":"value"}'


def test_remove_trailing_commas_handles_objects_and_arrays() -> None:
    repaired, applied = remove_trailing_commas('{"key":"value",}')
    assert applied is True
    assert repaired == '{"key":"value"}'

    repaired, applied = remove_trailing_commas('["a","b",]')
    assert applied is True
    assert repaired == '["a","b"]'


def test_fix_unquoted_values_wraps_identifier_but_not_reserved_words() -> None:
    repaired, applied = fix_unquoted_values('{"speaker_role": chief, "seen": true}')
    assert applied is True
    assert repaired == '{"speaker_role": "chief", "seen": true}'


def test_fix_missing_closing_braces_adds_single_missing_closer() -> None:
    repaired, applied = fix_missing_closing_braces('{"key":"value"')
    assert applied is True
    assert repaired == '{"key":"value"}'


def test_repair_json_applies_multiple_repairs_in_order() -> None:
    repaired, repairs = repair_json("```json\n{'speaker_role': chief,}\n```")
    assert repaired == '{"speaker_role": "chief"}'
    assert repairs == [
        "fence_strip",
        "trailing_comma",
        "single_to_double_quotes",
        "unquoted_value_fix",
    ]


def test_repair_json_leaves_valid_json_unchanged() -> None:
    repaired, repairs = repair_json('{"key":"value"}')
    assert repaired == '{"key":"value"}'
    assert repairs == []


def test_repair_json_handles_empty_string() -> None:
    repaired, repairs = repair_json("")
    assert repaired == ""
    assert repairs == []
