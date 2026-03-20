from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_extended_comparison.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_extended_comparison_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 16


def test_extended_comparison_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Setup & Load Config",
        "Server Helpers",
        "Programmatic Test Prompt Generator",
        "Run All Tests",
        "Auto-Grade",
        "Per Model Summary",
        "Per Task Breakdown",
        "Personality Consistency Analysis",
        "Save Results",
    ]:
        assert keyword in full_text


def test_extended_comparison_notebook_has_expected_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "ALL_PROMPTS",
        "auto_grade",
        "pair_id",
        "consistency",
        "extended_model_comparison.json",
        "start_server",
        "query_model",
        "TEMPERAMENTS",
        "deception_scenarios",
    ]:
        assert keyword in code_text


def test_extended_comparison_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_extended_comparison_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
