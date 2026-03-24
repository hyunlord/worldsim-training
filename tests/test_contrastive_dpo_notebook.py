from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_contrastive_dpo.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_contrastive_dpo_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 22


def test_contrastive_dpo_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment Setup",
        "Generate Contrastive Outputs",
        "Build DPO Pairs",
        "Reward Gap Analysis",
        "DPO Training",
        "GGUF Conversion",
        "SFT vs DPO Comparison",
        "Auto-Grade & Compare",
        "Personality Consistency",
        "Save Results",
    ]:
        assert keyword in full_text


def test_contrastive_dpo_notebook_has_expected_pipeline_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "google/gemini-2.5-flash",
        "CONTRASTIVE_PAIRS",
        "DPOTrainer",
        "combined_reward",
        "call_teacher",
        "build_task_e_prompt",
        "build_student_prompt",
        "chosen",
        "contrastive",
        "worldsim-dpo-contrastive-v31-qwen3.5-2b-q4_k_m.gguf",
    ]:
        assert keyword in code_text


def test_contrastive_dpo_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_contrastive_dpo_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
