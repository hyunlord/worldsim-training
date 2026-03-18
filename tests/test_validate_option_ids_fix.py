from __future__ import annotations

from scripts.validate_data import _validate_option_ids


def test_validate_option_ids_accepts_plain_string_options_by_index() -> None:
    record = {"action_options": ["flee", "hide", "confront", "warn", "freeze"]}

    assert _validate_option_ids("E", {"action_id": 3}, record) == []
    assert _validate_option_ids("E", {"action_id": 7}, record) == ["invalid_action_id"]


def test_validate_option_ids_keeps_dict_option_behavior() -> None:
    record = {"action_options": [{"id": 0, "ko": "a"}, {"id": 1, "ko": "b"}]}

    assert _validate_option_ids("E", {"action_id": 1}, record) == []
    assert _validate_option_ids("E", {"action_id": 5}, record) == ["invalid_action_id"]
