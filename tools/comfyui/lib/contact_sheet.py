"""Contact sheet generator: thumbnails, HTML, and zip archive."""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image
from jinja2 import Environment, FileSystemLoader


@dataclass
class SheetEntry:
    """One sprite variant in the contact sheet."""
    building: str           # e.g. "campfire"
    index: int              # 1..8
    filename: str           # "campfire_001.png"
    relpath: str            # "campfire/campfire_001.png"
    thumb_relpath: str      # "_contact_sheet_thumbs/campfire_001.webp"
    seed: int
    prompt_positive: str
    notes: str


def _read_meta(building_dir: Path) -> dict:
    """Read _meta.json from a building directory.

    Returns an empty dict if the file is missing or malformed.
    """
    meta_path = building_dir / "_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_index(filename: str) -> int:
    """Extract the numeric index from a filename like 'campfire_003.png'.

    Returns 0 if the pattern cannot be parsed.
    """
    stem = Path(filename).stem          # "campfire_003"
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) == 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 0


def build_thumbnails(
    output_root: Path,
    thumb_dir: Path,
    size: tuple[int, int] = (256, 256),
) -> list[SheetEntry]:
    """Scan *output_root* for ``<building>/<building>_NNN.png`` files.

    Generate 256x256 WebP thumbnails into *thumb_dir* using Pillow
    (``NEAREST`` resample for crisp pixel art).  Read each building's
    ``_meta.json`` for seed / prompt / notes.

    Returns entries sorted by ``(building, index)``.  Skips regeneration
    when the thumbnail already exists and is newer than the source PNG.
    """
    output_root = Path(output_root)
    thumb_dir = Path(thumb_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)

    entries: list[SheetEntry] = []

    # Only look at immediate subdirectories (skip dirs starting with "_")
    subdirs = sorted(
        d for d in output_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    for building_dir in subdirs:
        building_name = building_dir.name
        meta = _read_meta(building_dir)
        seed = meta.get("seed", 0)
        prompt_positive = meta.get("positive_prompt", "")
        notes = meta.get("notes", "")

        pngs = sorted(building_dir.glob("*.png"))
        for png_path in pngs:
            index = _extract_index(png_path.name)
            thumb_stem = png_path.stem  # e.g. "campfire_001"
            thumb_name = f"{thumb_stem}.webp"
            thumb_path = thumb_dir / thumb_name

            # Skip regeneration if thumb is newer than source
            if thumb_path.exists():
                if thumb_path.stat().st_mtime >= png_path.stat().st_mtime:
                    entries.append(SheetEntry(
                        building=building_name,
                        index=index,
                        filename=png_path.name,
                        relpath=f"{building_name}/{png_path.name}",
                        thumb_relpath=f"{thumb_dir.name}/{thumb_name}",
                        seed=seed,
                        prompt_positive=prompt_positive,
                        notes=notes,
                    ))
                    continue

            # Generate thumbnail
            with Image.open(png_path) as img:
                thumb = img.resize(size, resample=Image.NEAREST)
                thumb.save(thumb_path, format="WEBP", quality=90)

            entries.append(SheetEntry(
                building=building_name,
                index=index,
                filename=png_path.name,
                relpath=f"{building_name}/{png_path.name}",
                thumb_relpath=f"{thumb_dir.name}/{thumb_name}",
                seed=seed,
                prompt_positive=prompt_positive,
                notes=notes,
            ))

    # Sort by (building, index)
    entries.sort(key=lambda e: (e.building, e.index))
    return entries


def render_html(
    entries: list[SheetEntry],
    template_path: Path,
    output_html: Path,
    title: str = "WorldSim Building Concepts",
    generated_at: datetime | None = None,
    workflow_hash: str = "",
) -> None:
    """Render the Jinja2 contact-sheet template with entries grouped by
    building.

    The template receives:

    - ``title``, ``generated_at``, ``workflow_hash``
    - ``total_buildings``, ``total_variants``, ``total_size_mb``
    - ``groups``: list of dicts with keys ``name``, ``seed``,
      ``batch_size``, ``notes``, ``prompt``, ``entries``
    """
    template_path = Path(template_path)
    output_html = Path(output_html)

    if generated_at is None:
        generated_at = datetime.now()

    # Group entries by building
    groups_dict: dict[str, list[SheetEntry]] = {}
    for entry in entries:
        groups_dict.setdefault(entry.building, []).append(entry)

    # Calculate total source PNG size
    total_bytes = 0
    output_root = output_html.parent  # contact sheet sits next to building dirs
    for entry in entries:
        src = output_root / entry.relpath
        if src.exists():
            total_bytes += src.stat().st_size

    total_size_mb = round(total_bytes / (1024 * 1024), 2)

    # Build template groups
    groups: list[dict] = []
    for name, group_entries in groups_dict.items():
        first = group_entries[0]
        groups.append({
            "name": name,
            "seed": first.seed,
            "batch_size": len(group_entries),
            "notes": first.notes,
            "prompt": first.prompt_positive,
            "entries": group_entries,
        })

    # Sort groups alphabetically
    groups.sort(key=lambda g: g["name"])

    # Render template
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=True,
    )
    template = env.get_template(template_path.name)

    html = template.render(
        title=title,
        generated_at=generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        workflow_hash=workflow_hash,
        total_buildings=len(groups),
        total_variants=len(entries),
        total_size_mb=total_size_mb,
        groups=groups,
    )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def create_zip(
    output_root: Path,
    zip_path: Path,
    include_thumbs: bool = False,
) -> int:
    """Zip the concept tree for backup/sharing.

    Includes all ``<building>/*.png``, ``<building>/_meta.json``, and
    ``_contact_sheet.html``.  Excludes ``_archive/`` and
    ``_contact_sheet_thumbs/`` (unless *include_thumbs* is True).

    Returns total bytes written.
    """
    output_root = Path(output_root)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_dirs = {"_archive"}
    if not include_thumbs:
        excluded_dirs.add("_contact_sheet_thumbs")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add building subdirectories
        for subdir in sorted(output_root.iterdir()):
            if not subdir.is_dir():
                continue
            if subdir.name in excluded_dirs:
                continue
            # Only include immediate subdirs that are building dirs
            # (those not starting with "_", plus thumbs if requested)
            if subdir.name.startswith("_") and subdir.name != "_contact_sheet_thumbs":
                continue

            for fpath in sorted(subdir.rglob("*")):
                if not fpath.is_file():
                    continue
                # For building dirs: include .png and _meta.json
                if not subdir.name.startswith("_"):
                    if fpath.suffix == ".png" or fpath.name == "_meta.json":
                        arcname = fpath.relative_to(output_root)
                        zf.write(fpath, arcname)
                else:
                    # This is _contact_sheet_thumbs (only when include_thumbs)
                    arcname = fpath.relative_to(output_root)
                    zf.write(fpath, arcname)

        # Add _contact_sheet.html if it exists
        html_path = output_root / "_contact_sheet.html"
        if html_path.exists():
            zf.write(html_path, html_path.name)

    return zip_path.stat().st_size
