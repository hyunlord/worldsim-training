import json
from pathlib import Path

from scripts.generate_general_korean import CATEGORY_TARGETS, build_general_korean_examples, main as general_main


def test_build_general_korean_examples_covers_all_categories() -> None:
    rows = build_general_korean_examples(count=300, seed=42)

    assert len(rows) == 300
    categories = {row["category"] for row in rows}
    assert categories == set(CATEGORY_TARGETS)
    assert all(row["task"] == "GEN" for row in rows)
    assert all(row["label"] == "retain" for row in rows)
    assert all("prompt" in row and "output" in row for row in rows)


def test_general_korean_generator_cli_writes_requested_count(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "general_korean.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_general_korean.py", "--count", "10", "--output", str(output_path), "--seed", "42"],
    )

    general_main()

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 10
    assert all(row["task"] == "GEN" for row in rows)
