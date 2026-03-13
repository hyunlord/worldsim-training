from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_v2_dataset_assembly.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_v2_dataset_assembly_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 12


def test_v2_dataset_assembly_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])
    for keyword in ["Repo Root", "Source Inventory", "Run Assembly", "Convert", "Curriculum", "Final Dataset Stats", "Comparison"]:
        assert keyword in full_text


def test_v2_dataset_assembly_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_v2_dataset_assembly_notebook_imports_existing_modules() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "from scripts.assemble_v2_dataset import" in code_text
    assert "from scripts.convert_mixed_final_to_training_format import" in code_text
    assert "from scripts.curriculum_order import" in code_text
    assert "from scripts.common import" in code_text
