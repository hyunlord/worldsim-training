from __future__ import annotations


def _row(task: str, label: str) -> dict:
    return {
        "task": task,
        "messages": [
            {"role": "user", "content": label},
            {"role": "assistant", "content": "{}"},
        ],
    }


def test_select_generation_rows_returns_up_to_per_task() -> None:
    from training.lib.qlora_smoke import _select_generation_rows

    train_rows = [_row("A", f"train-{idx}") for idx in range(10)]
    eval_rows: list[dict] = []

    picked = _select_generation_rows(train_rows, eval_rows, per_task=5)

    assert len([row for row in picked if row["task"] == "A"]) == 5


def test_select_generation_rows_prefers_eval_rows_over_train_rows() -> None:
    from training.lib.qlora_smoke import _select_generation_rows

    train_rows = [_row("A", f"train-{idx}") for idx in range(5)]
    eval_rows = [_row("A", f"eval-{idx}") for idx in range(3)]

    picked = _select_generation_rows(train_rows, eval_rows, per_task=5)

    labels = [row["messages"][0]["content"] for row in picked if row["task"] == "A"]
    assert labels[:3] == ["eval-0", "eval-1", "eval-2"]
    assert labels[3:] == ["train-0", "train-1"]


def test_select_generation_rows_handles_fewer_than_per_task_rows() -> None:
    from training.lib.qlora_smoke import _select_generation_rows

    train_rows = [_row("B", "train-0")]
    eval_rows = [_row("B", "eval-0")]

    picked = _select_generation_rows(train_rows, eval_rows, per_task=5)

    assert len([row for row in picked if row["task"] == "B"]) == 2


def test_select_generation_rows_per_task_one_matches_old_behavior() -> None:
    from training.lib.qlora_smoke import _select_generation_rows

    train_rows = [_row("A", "train-a"), _row("B", "train-b"), _row("A", "train-a-2")]
    eval_rows = [_row("A", "eval-a"), _row("B", "eval-b")]

    picked = _select_generation_rows(train_rows, eval_rows, per_task=1)

    assert len(picked) == 2
    assert [row["task"] for row in picked] == ["A", "B"]
    assert [row["messages"][0]["content"] for row in picked] == ["eval-a", "eval-b"]


def test_select_generation_rows_total_matches_sum_of_available_rows() -> None:
    from training.lib.qlora_smoke import _select_generation_rows

    train_rows = [
        _row("A", "train-a0"),
        _row("A", "train-a1"),
        _row("B", "train-b0"),
        _row("C", "train-c0"),
        _row("C", "train-c1"),
        _row("C", "train-c2"),
    ]
    eval_rows = [_row("A", "eval-a0"), _row("C", "eval-c0")]

    picked = _select_generation_rows(train_rows, eval_rows, per_task=2)

    assert len(picked) == 5
    assert len([row for row in picked if row["task"] == "A"]) == 2
    assert len([row for row in picked if row["task"] == "B"]) == 1
    assert len([row for row in picked if row["task"] == "C"]) == 2
