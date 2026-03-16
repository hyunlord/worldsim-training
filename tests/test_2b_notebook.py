from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_2b_train_and_convert.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_2b_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 18


def test_2b_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment & Dataset Verification",
        "Training Config",
        "Train",
        "Guardrail Evaluation",
        "LoRA Merge",
        "Convert HF",
        "Quantize",
        "Verify GGUF",
        "Copy to artifacts",
        "Next: Re-run Parallel Benchmark",
    ]:
        assert keyword in full_text


def test_2b_notebook_has_expected_training_and_conversion_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "Qwen/Qwen3.5-2B-Base" in code_text
    assert "worldsim-2b-v2-mix" in code_text
    assert "resolve_baseline_notebook_config" in code_text
    assert "run_baseline_or_raise" in code_text
    assert "merge_and_unload" in code_text
    assert "Q4_K_M" in code_text
    assert "worldsim-v2-qwen3.5-2b" in code_text
    assert "llama-completion" in code_text
    assert "strip_think" in code_text


def test_2b_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_2b_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
