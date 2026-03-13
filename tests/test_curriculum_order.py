from __future__ import annotations

from scripts.curriculum_order import curriculum_order


def test_curriculum_order_preserves_stage_order() -> None:
    rows = [{"task": task, "id": index} for index, task in enumerate(["H", "C", "A", "E", "NEG", "I"])]
    ordered = curriculum_order(rows, seed=42)
    tasks = [row["task"] for row in ordered]

    assert tasks.index("E") < tasks.index("A") < tasks.index("C") < tasks.index("H")
    assert tasks.index("I") < tasks.index("A")


def test_curriculum_order_interleaves_within_stage() -> None:
    rows = [
        {"task": "E", "id": 1},
        {"task": "F", "id": 2},
        {"task": "I", "id": 3},
        {"task": "J", "id": 4},
        {"task": "E", "id": 5},
        {"task": "F", "id": 6},
    ]
    ordered = curriculum_order(rows, seed=42)
    ordered_tasks = [row["task"] for row in ordered[:6]]

    assert ordered_tasks != ["E", "E", "F", "F", "I", "J"]
    assert set(ordered_tasks) == {"E", "F", "I", "J"}


def test_curriculum_order_preserves_all_rows_and_is_deterministic() -> None:
    rows = [{"task": task, "id": index} for index, task in enumerate(["E", "A", "C", "H", "NEG", "GEN", "M", "K"])]
    first = curriculum_order(rows, seed=7)
    second = curriculum_order(rows, seed=7)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert sorted(row["id"] for row in first) == list(range(len(rows)))


def test_curriculum_order_handles_empty_input() -> None:
    assert curriculum_order([], seed=42) == []
