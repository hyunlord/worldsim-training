from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_v3_data_and_train.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_v3_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 25


def test_v3_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment & Prerequisites",
        "Batch Preview",
        "Batch 1: English Logic Tasks",
        "Batch 2: New Tasks",
        "Validation",
        "Assemble v3 Dataset",
        "Convert to Training Format",
        "Train 0.8B on v3 Data",
        "Train 2B on v3 Data",
        "GGUF Conversion",
        "Grand Summary",
    ]:
        assert keyword in full_text


def test_v3_notebook_has_expected_pipeline_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "batch_v3_01_english_logic" in code_text
    assert "batch_v3_02_new_tasks" in code_text
    assert "assemble_v3_dataset" in code_text
    assert "convert_mixed_final_to_training_format" in code_text
    assert "curriculum_order_v3" in code_text
    assert "Qwen/Qwen3.5-0.8B-Base" in code_text
    assert "Qwen/Qwen3.5-2B-Base" in code_text
    assert "run_baseline_or_raise" in code_text
    assert "merge_and_convert_gguf" in code_text
    assert "Q4_K_M" in code_text
    assert "worldsim-v3" in code_text


def test_v3_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_v3_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
