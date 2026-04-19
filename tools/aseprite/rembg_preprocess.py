#!/usr/bin/env python3
"""rembg_preprocess.py — Phase 1.5: AI background removal for building sprites.

Reads concepts_final/{building}/*.png, applies rembg to produce transparent
foreground-only PNGs, writes to _raw_cleanup_stage/00_rembg/{building}/.

Usage:
    python tools/aseprite/rembg_preprocess.py \
        --source-root assets/sprites/concepts_final \
        --staging-root _raw_cleanup_stage \
        --model u2net
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

try:
    from rembg import remove, new_session
except ImportError:
    print("ERROR: rembg not installed. Run: pip install rembg onnxruntime", file=sys.stderr)
    sys.exit(1)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def process_building(
    source_dir: Path,
    output_dir: Path,
    session,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Process all PNGs in source_dir → output_dir. Returns (success, fail) counts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(source_dir.glob("*.png"))
    if not pngs:
        logging.warning(f"No PNGs found in {source_dir}")
        return (0, 0)

    success, fail = 0, 0
    for png_path in pngs:
        out_path = output_dir / png_path.name
        if dry_run:
            logging.info(f"[DRY] {png_path} → {out_path}")
            continue
        try:
            with Image.open(png_path) as im:
                result = remove(im, session=session)
                if result.mode != "RGBA":
                    result = result.convert("RGBA")
                result.save(out_path, "PNG")
            success += 1
        except Exception as e:
            logging.error(f"FAIL {png_path}: {e}")
            fail += 1
    return (success, fail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path,
                        help="concepts_final/ directory (flat: {building}/*.png)")
    parser.add_argument("--staging-root", required=True, type=Path,
                        help="_raw_cleanup_stage/ parent directory")
    parser.add_argument("--model", default="u2net",
                        choices=["u2net", "isnet-general-use", "silueta", "u2netp"])
    parser.add_argument("--only", default=None,
                        help="Process only this building id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = args.staging_root / "00_rembg"
    output_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading rembg model: {args.model}")
    session = new_session(args.model)

    total_success, total_fail = 0, 0
    building_dirs = sorted(
        d for d in args.source_root.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    for building_dir in building_dirs:
        building_id = building_dir.name
        if args.only and building_id != args.only:
            continue
        if building_id == "shelter":
            logging.info("[SKIP] shelter — configured skip (game decomposes into tiles)")
            continue

        logging.info(f"[PROCESS] {building_id}")
        out_dir = output_root / building_id
        s, f = process_building(building_dir, out_dir, session, dry_run=args.dry_run)
        total_success += s
        total_fail += f
        logging.info(f"  → {s} success, {f} fail")

    logging.info("")
    logging.info(f"TOTAL: {total_success} success, {total_fail} fail")
    if args.dry_run:
        logging.info("(Dry run — no files written)")

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
