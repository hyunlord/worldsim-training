from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_gemma4_benchmark.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_gemma4_benchmark_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 20


def test_gemma4_benchmark_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Environment & Model Download",
        "Server Helpers & Test Prompts",
        "Phase 1 — Base Model Test",
        "Phase 1 Results — Base Model Comparison",
        "Fine-Tune Gemma4 E2B + E4B",
        "GGUF Conversion",
        "Full 7-Model Comparison",
        "Full Comparison Table",
        "Personality Consistency",
        "WorldSim Recommendation",
        "Save Results",
    ]:
        assert keyword in full_text


def test_gemma4_benchmark_notebook_has_expected_pipeline_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "google/gemma-4-E2B",
        "google/gemma-4-E4B",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "unsloth/gemma-4-E2B-it-GGUF",
        "unsloth/gemma-4-E4B-it-GGUF",
        "hf_hub_download",
        "start_server",
        "ALL_PROMPTS",
        "auto_grade",
        "SmokeRunConfig",
        "run_baseline_or_raise",
        "worldsim-v31-gemma4-e2b",
        "worldsim-v31-gemma4-e4b",
        "gemma4_full_benchmark.json",
    ]:
        assert keyword in code_text


def test_gemma4_benchmark_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_gemma4_benchmark_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
