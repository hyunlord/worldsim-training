from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_gguf_conversion.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_gguf_conversion_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 12


def test_gguf_conversion_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment & Adapter Discovery",
        "LoRA Merge",
        "Build llama.cpp",
        "Convert HF -> GGUF",
        "Quantize -> Q4_K_M",
        "Verify GGUF",
        "Copy to artifacts/gguf",
    ]:
        assert keyword in full_text


def test_gguf_conversion_notebook_has_expected_conversion_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "BASELINE_MODEL_NAME" in code_text
    assert "BASELINE_DATASET_ID" in code_text
    assert "merge_and_unload" in code_text
    assert "convert_hf_to_gguf.py" in code_text
    assert "Q4_K_M" in code_text
    assert "llama-quantize" in code_text
    assert "llama-cli" in code_text
    assert "llama-server" in code_text
    assert "artifacts" in code_text
    assert "gguf" in code_text


def test_gguf_conversion_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_gguf_conversion_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")


def test_gguf_conversion_notebook_imports_existing_modules() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from training.lib.qlora_smoke import BASELINE_DATASET_ID, BASELINE_MODEL_NAME" in code_text
    assert "from peft import PeftModel" in code_text
    assert "from transformers import AutoModelForCausalLM, AutoTokenizer" in code_text
    assert "import subprocess" in code_text
    assert "import shutil" in code_text
