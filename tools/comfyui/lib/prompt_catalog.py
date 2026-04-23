"""YAML prompt catalog loader for ComfyUI batch generation."""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path


class CatalogError(Exception):
    """Raised when the YAML catalog is invalid."""
    pass


@dataclass
class BuildingPrompt:
    """A single building's generation parameters."""
    name: str
    positive: str       # style_prefix + building positive merged
    negative: str       # global negative_prompt
    seed: int
    batch_size: int
    notes: str


class PromptCatalog:
    """Loads and validates a building prompt catalog from YAML."""

    def __init__(self, buildings: list[BuildingPrompt], version: int = 1):
        self._buildings = {b.name: b for b in buildings}
        self.version = version

    @classmethod
    def load(cls, yaml_path: Path) -> "PromptCatalog":
        """Load catalog from YAML file.

        Validation rules:
        - 'version' must be present and == 1
        - 'global' must have 'negative_prompt' and 'style_prefix' (both non-empty strings)
        - 'buildings' must be a dict with at least one entry
        - Each building must have: 'positive' (non-empty str), 'seed' (int), 'batch_size' (int > 0)
        - 'notes' is optional (defaults to "")
        - No duplicate building names (enforced by YAML dict keys)
        - style_prefix is prepended to each building's positive prompt with a newline separator

        Raises CatalogError on any validation failure.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise CatalogError(f"Catalog file not found: {yaml_path}")

        try:
            raw = yaml_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CatalogError(f"Cannot read catalog file: {exc}") from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise CatalogError(f"Invalid YAML syntax: {exc}") from exc

        if not isinstance(data, dict):
            raise CatalogError("Catalog root must be a YAML mapping")

        # --- version ---
        if "version" not in data:
            raise CatalogError("Missing required top-level key 'version'")
        if data["version"] != 1:
            raise CatalogError(
                f"Unsupported catalog version {data['version']} (expected 1)"
            )

        # --- global ---
        if "global" not in data:
            raise CatalogError("Missing required top-level key 'global'")
        glb = data["global"]
        if not isinstance(glb, dict):
            raise CatalogError("'global' must be a mapping")

        for field in ("negative_prompt", "style_prefix"):
            if field not in glb:
                raise CatalogError(f"'global' missing required field '{field}'")
            val = glb[field]
            if not isinstance(val, str) or not val.strip():
                raise CatalogError(
                    f"'global.{field}' must be a non-empty string"
                )

        negative_prompt = glb["negative_prompt"].strip()
        style_prefix = glb["style_prefix"].strip()

        # --- buildings ---
        if "buildings" not in data:
            raise CatalogError("Missing required top-level key 'buildings'")
        buildings_raw = data["buildings"]
        if not isinstance(buildings_raw, dict):
            raise CatalogError("'buildings' must be a mapping")
        if len(buildings_raw) == 0:
            raise CatalogError("'buildings' must contain at least one entry")

        buildings: list[BuildingPrompt] = []
        for name, bldg in buildings_raw.items():
            if not isinstance(bldg, dict):
                raise CatalogError(
                    f"Building '{name}' must be a mapping, got {type(bldg).__name__}"
                )

            # positive (required, non-empty string)
            if "positive" not in bldg:
                raise CatalogError(
                    f"Building '{name}' missing required field 'positive'"
                )
            positive = bldg["positive"]
            if not isinstance(positive, str) or not positive.strip():
                raise CatalogError(
                    f"Building '{name}' field 'positive' must be a non-empty string"
                )

            # seed (required, int)
            if "seed" not in bldg:
                raise CatalogError(
                    f"Building '{name}' missing required field 'seed'"
                )
            seed = bldg["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise CatalogError(
                    f"Building '{name}' field 'seed' must be an integer"
                )

            # batch_size (required, int > 0)
            if "batch_size" not in bldg:
                raise CatalogError(
                    f"Building '{name}' missing required field 'batch_size'"
                )
            batch_size = bldg["batch_size"]
            if not isinstance(batch_size, int) or isinstance(batch_size, bool):
                raise CatalogError(
                    f"Building '{name}' field 'batch_size' must be an integer"
                )
            if batch_size <= 0:
                raise CatalogError(
                    f"Building '{name}' field 'batch_size' must be > 0, got {batch_size}"
                )

            # notes (optional, defaults to "")
            notes = bldg.get("notes", "")
            if not isinstance(notes, str):
                raise CatalogError(
                    f"Building '{name}' field 'notes' must be a string"
                )

            # Merge style_prefix + positive
            merged_positive = f"{style_prefix}\n{positive.strip()}"

            buildings.append(
                BuildingPrompt(
                    name=str(name),
                    positive=merged_positive,
                    negative=negative_prompt,
                    seed=seed,
                    batch_size=batch_size,
                    notes=notes,
                )
            )

        return cls(buildings=buildings, version=data["version"])

    def buildings(self) -> list[BuildingPrompt]:
        """Return all buildings in insertion order."""
        return list(self._buildings.values())

    def get(self, name: str) -> BuildingPrompt | None:
        """Get a building by name, or None if not found."""
        return self._buildings.get(name)

    def names(self) -> list[str]:
        """Return all building names in insertion order."""
        return list(self._buildings.keys())

    def __len__(self) -> int:
        return len(self._buildings)

    def __repr__(self) -> str:
        return f"PromptCatalog(version={self.version}, buildings={len(self._buildings)})"
