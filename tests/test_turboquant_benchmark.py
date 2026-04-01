from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_turboquant_benchmark.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_turboquant_benchmark_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 18


def test_turboquant_benchmark_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Build TurboQuant llama.cpp Fork",
        "llama-bench Speed Test",
        "Server Helpers & Test Prompts",
        "Quality + Speed Benchmark",
        "Auto-Grade",
        "Results Summary",
        "Personality Consistency",
        "WorldSim Recommendation",
        "Save Results",
    ]:
        assert keyword in full_text


def test_turboquant_benchmark_notebook_has_expected_pipeline_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    for keyword in [
        "turboquant",
        "turbo3",
        "turbo4",
        "q8_0",
        "TheTom/llama-cpp-turboquant",
        "start_server",
        "auto_grade",
        "SPEED_SUMMARY",
        "llama-bench",
        "ALL_PROMPTS",
        "turboquant_kv_benchmark.json",
    ]:
        assert keyword in code_text


def test_turboquant_benchmark_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_turboquant_benchmark_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
