import json
from pathlib import Path

from scripts.generate_negative_examples import CATEGORY_TARGETS, build_negative_examples, main as negative_main


def test_build_negative_examples_covers_all_categories() -> None:
    rows = build_negative_examples(count=500, seed=42)

    assert len(rows) == 500
    categories = {row["category"] for row in rows}
    assert categories == set(CATEGORY_TARGETS)
    assert all(row["task"] == "NEG" for row in rows)
    assert all(row["label"] == "reject" for row in rows)
    assert all("reason" in row for row in rows)


def test_negative_generator_cli_writes_requested_count(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "negative_examples.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_negative_examples.py", "--count", "10", "--output", str(output_path), "--seed", "42"],
    )

    negative_main()

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 10
    assert all(row["task"] == "NEG" for row in rows)
