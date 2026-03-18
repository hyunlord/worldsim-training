from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_model_comparison.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_model_comparison_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 12


def test_model_comparison_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Setup & Model Inventory",
        "Convert Base Models to GGUF",
        "Server Helper & Test Prompts",
        "Run All Tests",
        "Side-by-Side Comparison",
        "Personality Consistency Check",
        "Aggregate Scores",
    ]:
        assert keyword in full_text


def test_model_comparison_notebook_has_expected_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "TEST_PROMPTS",
        "base_08b",
        "ft_08b",
        "base_2b",
        "ft_2b",
        "start_server",
        "json_valid",
        "model_comparison.json",
        "convert_base_to_gguf",
        "llama-server",
    ]:
        assert keyword in code_text


def test_model_comparison_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_model_comparison_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
