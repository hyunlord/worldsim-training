from pathlib import Path

from scripts.validate_data import load_validation_rules, validate_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row}\n".replace("'", '"'))


def test_validate_file_splits_pass_fail_and_writes_report(tmp_path: Path) -> None:
    raw_file = tmp_path / "data" / "raw" / "sample.jsonl"
    validated_dir = tmp_path / "data" / "validated"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "generation.yaml").write_text(
        """
validation:
  forbidden_words: ["식량"]
  meta_patterns: ["WorldSim"]
  task_limits:
    A: {min_chars: 5, max_chars: 40, sentences: 1}
    B: {min_chars: 5, max_chars: 60, sentences: 2}
    C: {min_chars: 5, max_chars: 30, sentences: 1}
    D: {min_chars: 5, max_chars: 30, sentences: 1}
""".strip(),
        encoding="utf-8",
    )
    write_jsonl(
        raw_file,
        [
            {"task": "A", "register": "haera", "output": "풀숲을 살피며 조심스레 걸음을 옮겼다."},
            {"task": "D", "register": "haera", "output": "식량을 찾았다."},
        ],
    )

    rules = load_validation_rules(config_dir)
    summary = validate_file(raw_file, validated_dir=validated_dir, rules=rules)

    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert (validated_dir / "passed.jsonl").exists()
    assert (validated_dir / "failed.jsonl").exists()
    assert (validated_dir / "report.json").exists()

