#!/usr/bin/env python3
"""apply_tile_selection.py — Apply user's tile selection to game assets.

Reads _tile_review/selection.json (exactly 3 files per material).
Copies selected files from _tile_review/{category}/{material}/candidates/
to assets/sprites/{walls,floors}/{material}/{1,2,3}.png.

Usage:
    python tools/aseprite/apply_tile_selection.py \\
        --review-dir assets/sprites/_tile_review \\
        --walls-dst assets/sprites/walls \\
        --floors-dst assets/sprites/floors
"""

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image


def verify_tile(path):
    img = Image.open(path)
    if img.size != (16, 16):
        raise ValueError(f"{path}: size {img.size} (expected 16×16)")
    if img.mode != "RGB":
        raise ValueError(f"{path}: mode {img.mode} (expected RGB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", required=True, type=Path)
    parser.add_argument("--walls-dst", required=True, type=Path)
    parser.add_argument("--floors-dst", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection_path = args.review_dir / "selection.json"
    if not selection_path.exists():
        print(f"ERROR: {selection_path} not found")
        return 1

    selection = json.loads(selection_path.read_text())
    errors = []
    written = []

    for category, dst_root in [("walls", args.walls_dst),
                                ("floors", args.floors_dst)]:
        if category not in selection:
            print(f"WARN: '{category}' not in selection.json, skipping")
            continue
        for mat, files in selection[category].items():
            # SKIP_V1 sentinel — preserve existing v1 tiles unchanged
            if files == ["SKIP_V1"] or (len(files) >= 1 and files[0] == "SKIP_V1"):
                print(f"  {category}/{mat}: SKIP_V1 (preserving v1 tiles)")
                continue

            if len(files) != 3:
                errors.append(
                    f"{category}/{mat}: {len(files)} files selected (need exactly 3)"
                )
                continue
            cands_dir = args.review_dir / category / mat / "candidates"
            # Verify all source files exist and are valid tiles
            src_paths = []
            for fname in files:
                src = cands_dir / fname
                if not src.exists():
                    errors.append(f"{category}/{mat}: source not found: {src}")
                    continue
                try:
                    verify_tile(src)
                except ValueError as e:
                    errors.append(str(e))
                    continue
                src_paths.append(src)

            if len(src_paths) != 3:
                continue

            dst_dir = dst_root / mat
            if args.dry_run:
                print(f"[dry-run] {category}/{mat}: {[f.name for f in src_paths]} → {dst_dir}/1-3.png")
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            # Clear existing tiles
            for old in dst_dir.glob("*.png"):
                old.unlink()
            # Copy renamed to 1.png, 2.png, 3.png
            for idx, src in enumerate(src_paths, start=1):
                dst = dst_dir / f"{idx}.png"
                shutil.copy2(src, dst)
                written.append(f"{category}/{mat}/{idx}.png ← {src.name}")
                print(f"  {category}/{mat}: {src.name} → {idx}.png")

    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    if not args.dry_run:
        print(f"\nWrote {len(written)} tiles to game tree.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
