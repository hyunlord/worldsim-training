from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_v2_baseline_train.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_v2_baseline_train_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 15


def test_v2_baseline_train_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Repo Root",
        "Environment Visibility",
        "True QLoRA Preflight",
        "v2 Dataset Assembly",
        "v2 Dataset Statistics",
        "Trainer Invocation",
        "Guardrail Impact Summary",
        "v1 vs v2 Comparison",
    ]:
        assert keyword in full_text


def test_v2_baseline_train_notebook_has_v2_dataset_references_and_hyperparameters() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "worldsim-v2-mix" in code_text
    assert "train_curriculum.jsonl" in code_text
    assert "dev_converted.jsonl" in code_text
    assert "curriculum_order" in code_text
    assert "max_steps" in code_text
    assert "1296" in code_text
    assert "eval_steps" in code_text
    assert "100" in code_text
    assert "save_steps" in code_text
    assert "save_total_limit" in code_text
    assert "3" in code_text


def test_v2_baseline_train_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_v2_baseline_train_notebook_imports_existing_modules() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from training.lib.qlora_smoke import" in code_text
    assert "from scripts.assemble_v2_dataset import" in code_text
    assert "from scripts.convert_mixed_final_to_training_format import" in code_text
    assert "from scripts.curriculum_order import" in code_text
    assert "from scripts.common import" in code_text
    assert "run_baseline_or_raise" in code_text
    assert "extract_metrics" in code_text
    assert "print_report" in code_text
