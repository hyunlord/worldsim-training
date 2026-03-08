from pathlib import Path


EXPECTED_DIRS = [
    "docs/design",
    "docs/plans",
    "docs/references",
    "config",
    "prompts/teacher",
    "prompts/validation",
    "scripts",
    "data/raw",
    "data/validated",
    "data/final",
    "data/samples",
    "training/configs",
    "training/recipes",
    "training/adapters",
    "eval/holdout",
    "eval/reports",
    "artifacts/merged",
    "artifacts/gguf",
    "artifacts/manifests",
    "tests",
]

EXPECTED_AGENTS = [
    "AGENTS.md",
    "docs/AGENTS.md",
    "config/AGENTS.md",
    "prompts/AGENTS.md",
    "scripts/AGENTS.md",
    "data/AGENTS.md",
    "training/AGENTS.md",
    "eval/AGENTS.md",
    "artifacts/AGENTS.md",
    "tests/AGENTS.md",
]


def test_expected_directories_exist() -> None:
    root = Path.cwd()
    for relative_path in EXPECTED_DIRS:
        assert (root / relative_path).is_dir(), relative_path


def test_expected_agents_files_exist() -> None:
    root = Path.cwd()
    for relative_path in EXPECTED_AGENTS:
        assert (root / relative_path).is_file(), relative_path
