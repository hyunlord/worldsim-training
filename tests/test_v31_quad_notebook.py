from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_v31_quad_model.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_v31_quad_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 16


def test_v31_quad_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment & Prerequisite Check",
        "Generate Reinforcement Data",
        "Validate New Data",
        "Merge + Reassemble v3.1 Dataset",
        "Convert to Training Format",
        "Train All 4 Models",
        "GGUF Conversion",
        "Convert Base 4B/9B to GGUF",
        "Grand Summary",
    ]:
        assert keyword in full_text


def test_v31_quad_notebook_has_expected_pipeline_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "Qwen/Qwen3.5-4B-Base",
        "Qwen/Qwen3.5-9B-Base",
        "L3_EN",
        "batch_v31_04_task_f_reinforce",
        "batch_v31_05_personality_pairs",
        "assemble_v3_dataset",
        "convert_mixed_final_to_training_format",
        "curriculum_order_v3",
        "run_baseline_or_raise",
        "merge_and_convert_gguf",
        "worldsim-v31",
    ]:
        assert keyword in code_text


def test_v31_quad_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_v31_quad_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
