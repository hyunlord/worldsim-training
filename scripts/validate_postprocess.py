#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import read_jsonl, resolve_path
from scripts.lib.postprocess import CANONICAL_TASKS, load_postprocess_policy, validate_records


def validate_postprocess(repo_root: Path, *, input_file: Path, output_dir: Path) -> dict:
    records = [row for row in read_jsonl(input_file) if str(row.get("task", "")) in CANONICAL_TASKS]
    result = validate_records(records, load_postprocess_policy(repo_root / "config"), output_dir=output_dir)
    return {
        "input_file": str(input_file),
        "output_dir": str(output_dir),
        **result.report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layered postprocess validation for Tasks A/B/C.")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data/validated/postprocess")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    output = validate_postprocess(
        repo_root=repo_root,
        input_file=resolve_path(repo_root, args.input),
        output_dir=resolve_path(repo_root, args.output_dir),
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
