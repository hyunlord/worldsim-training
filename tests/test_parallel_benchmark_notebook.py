from __future__ import annotations

import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "dgx_spark_parallel_benchmark.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def test_parallel_benchmark_notebook_is_valid_json_with_expected_nbformat() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) >= 16


def test_parallel_benchmark_notebook_contains_expected_sections() -> None:
    notebook = _load_notebook()
    full_text = " ".join(" ".join(cell.get("source", [])) for cell in notebook["cells"])

    for keyword in [
        "Setup & Path Discovery",
        "Server Management & Request Helpers",
        "Test Prompts",
        "Sequential vs Parallel Slots",
        "Mixed Task Types in Parallel",
        "Free Text vs Grammar-Constrained",
        "Results Summary",
        "Recommendation",
    ]:
        assert keyword in full_text


def test_parallel_benchmark_notebook_has_expected_benchmark_references() -> None:
    notebook = _load_notebook()
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "n_parallel" in code_text
    assert "aiohttp" in code_text
    assert "llama-server" in code_text
    assert "parallel_slot_benchmark.json" in code_text
    assert "asyncio.get_event_loop().run_until_complete" in code_text
    assert "run_concurrent_requests" in code_text
    assert "MODEL_08B" in code_text
    assert "MODEL_2B" in code_text


def test_parallel_benchmark_notebook_code_cells_have_empty_outputs() -> None:
    notebook = _load_notebook()

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []


def test_parallel_benchmark_notebook_code_cells_parse_as_python() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
