#!/usr/bin/env python3
"""
Phase 1 auto-processor for WorldSim building sprites.

Pipeline for each of the concepts_final PNGs:
  1. Load 1024×1024 source
  2. Downscale to target png_size using nearest-neighbor (pixel-art preservation)
  3. Remove background via 4-corner flood-fill (alpha=0 for sampled colors)
  4. Quantize to 16-color palette (Pillow.quantize method=MEDIANCUT)
  5. Save to _raw_cleanup_stage/03_palette_reduced/{building}/{index}.png

After all variants processed:
  6. Build target directory structure: variant folder ONLY
       assets/sprites/{category}/{building}/1.png     ← all 16 variants (1-indexed, contiguous)
       assets/sprites/{category}/{building}/2.png
       ...
       assets/sprites/{category}/{building}/16.png
     NOTE: No representative file {building}.png is produced.
     Game's variant loader reads from the folder directly.
  7. Write _raw_cleanup_stage/_report.json with per-file stats.

Usage:
    python tools/aseprite/auto_process.py \
        --config tools/aseprite/configs/sprite_classification.yaml \
        --source-root assets/sprites/concepts_final \
        --staging-root _raw_cleanup_stage \
        --output-root assets/sprites
"""

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
import numpy as np


@dataclass
class ProcessingStats:
    building: str
    variant: int
    source_path: str
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    bg_pixels_removed: int
    bg_pixels_total: int
    palette_size_used: int
    output_path: str
    skipped: bool
    skip_reason: Optional[str] = None


def downscale_nearest(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    return img.resize(target_size, Image.Resampling.NEAREST)


def remove_background_flood(
    img: Image.Image,
    tolerance: int = 32,
    corner_samples: int = 4,
) -> tuple[Image.Image, int, int]:
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    h, w, _ = arr.shape
    corners = [
        arr[0, 0],
        arr[0, w - 1],
        arr[h - 1, 0],
        arr[h - 1, w - 1],
    ][:corner_samples]

    mask = np.zeros((h, w), dtype=bool)
    for corner_rgb in corners:
        diff = arr.astype(np.int16) - corner_rgb.astype(np.int16)
        dist_sq = (diff ** 2).sum(axis=-1)
        mask |= dist_sq <= (tolerance ** 2)

    rgba = np.dstack([arr, np.where(mask, 0, 255).astype(np.uint8)])
    bg_pixels = int(mask.sum())
    total_pixels = h * w
    return Image.fromarray(rgba, mode="RGBA"), bg_pixels, total_pixels


def binarize_alpha(img: Image.Image, threshold: int = 128) -> Image.Image:
    """Pixel art requires hard {0, 255} alpha. rembg produces 0..255 soft edges;
    binarize before palette reduction or halos bleed through."""
    if img.mode != "RGBA":
        return img
    arr = np.array(img)
    arr[:, :, 3] = np.where(arr[:, :, 3] > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def quantize_palette(img: Image.Image, palette_size: int = 16) -> tuple[Image.Image, int]:
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    rgb = img.convert("RGB")
    quantized = rgb.quantize(colors=palette_size, method=Image.Quantize.MEDIANCUT)
    actual_palette_size = len(quantized.getpalette()) // 3

    rgba = quantized.convert("RGBA")
    alpha = np.array(img)[:, :, 3]
    rgba_arr = np.array(rgba)
    rgba_arr[:, :, 3] = alpha
    return Image.fromarray(rgba_arr, mode="RGBA"), actual_palette_size


def process_one(
    source_png: Path,
    target_size: tuple[int, int],
    staging_dir: Path,
    building: str,
    variant: int,
    palette_size: int,
    bg_tolerance: int,
    use_rembg: bool = False,
    alpha_threshold: int = 128,
) -> ProcessingStats:
    img = Image.open(source_png)
    source_size = img.size

    s1 = downscale_nearest(img, target_size)
    stage1_dir = staging_dir / "01_downscaled" / building
    stage1_dir.mkdir(parents=True, exist_ok=True)
    s1.save(stage1_dir / f"{variant}.png")

    if use_rembg:
        # Source already has rembg alpha (0..255 soft). Binarize for pixel art.
        s2 = binarize_alpha(s1 if s1.mode == "RGBA" else s1.convert("RGBA"),
                            threshold=alpha_threshold)
        alpha_arr = np.array(s2)[:, :, 3]
        bg_pixels = int((alpha_arr == 0).sum())
        total_pixels = alpha_arr.size
    else:
        s2, bg_pixels, total_pixels = remove_background_flood(s1, tolerance=bg_tolerance)

    stage2_dir = staging_dir / "02_bg_removed" / building
    stage2_dir.mkdir(parents=True, exist_ok=True)
    s2.save(stage2_dir / f"{variant}.png")

    s3, palette_used = quantize_palette(s2, palette_size=palette_size)
    stage3_dir = staging_dir / "03_palette_reduced" / building
    stage3_dir.mkdir(parents=True, exist_ok=True)
    stage3_path = stage3_dir / f"{variant}.png"
    s3.save(stage3_path)

    return ProcessingStats(
        building=building,
        variant=variant,
        source_path=str(source_png),
        source_size=source_size,
        output_size=target_size,
        bg_pixels_removed=bg_pixels,
        bg_pixels_total=total_pixels,
        palette_size_used=palette_used,
        output_path=str(stage3_path),
        skipped=False,
    )


def classify_to_game_tree(
    staging_root: Path,
    output_root: Path,
    classifier: dict,
    force: bool = False,
) -> list[str]:
    """
    Copy staged variants into the game asset tree:
      assets/sprites/{category}/{building}/{variant}.png for every variant.

    NO representative file is produced. The game's variant loader reads the folder
    directly. Contiguity (1.png..N.png with no gaps) is required — the loader
    caches null for missing indices and any entity seeding into a gap renders blank.
    """
    written = []

    for building, cfg in classifier["buildings"].items():
        if cfg.get("skip_reason"):
            logging.info(f"[SKIP] {building}: {cfg['skip_reason']}")
            continue

        target_dir = Path(cfg["target_dir"])
        variant_dir = target_dir / building

        source_variants_dir = staging_root / "03_palette_reduced" / building
        if not source_variants_dir.exists():
            logging.warning(f"No staging output for {building}, skipping classify")
            continue

        variant_dir.mkdir(parents=True, exist_ok=True)

        variants_found = []
        for variant_png in sorted(source_variants_dir.glob("*.png"), key=lambda p: int(p.stem)):
            variant_idx = int(variant_png.stem)
            dest = variant_dir / f"{variant_idx}.png"
            if dest.exists() and not force:
                logging.info(f"[EXISTS] {dest} — use --force to overwrite")
            else:
                shutil.copy2(variant_png, dest)
                written.append(str(dest))
            variants_found.append(variant_idx)

        if variants_found:
            expected = set(range(1, max(variants_found) + 1))
            actual = set(variants_found)
            missing = expected - actual
            if missing:
                logging.error(
                    f"[GAP] {building}: missing variant indices {sorted(missing)}. "
                    f"Loader will cache these as null and skip rendering. "
                    f"Re-run ComfyUI batch for this building or manually fill gaps."
                )

    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--staging-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--only", type=str, default="", help="Comma-separated building names")
    ap.add_argument("--skip-classify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--use-rembg-source", action="store_true",
                    help="Read from {staging_root}/00_rembg/ instead of --source-root. "
                         "Skips 4-corner flood-fill and applies alpha binarization.")
    ap.add_argument("--alpha-threshold", type=int, default=128,
                    help="Alpha binarization threshold when --use-rembg-source (default 128)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # When using rembg source, redirect input from staging/00_rembg
    if args.use_rembg_source:
        effective_source = args.staging_root / "00_rembg"
        logging.info(f"Using rembg pre-processed source: {effective_source}")
    else:
        effective_source = args.source_root

    classifier = yaml.safe_load(args.config.read_text())
    proc_cfg = classifier["processing"]
    palette_size = proc_cfg["palette_size"]
    bg_tolerance = proc_cfg["background_removal"]["tolerance"]

    only_filter = set(s.strip() for s in args.only.split(",") if s.strip())
    all_stats: list[ProcessingStats] = []

    for building, cfg in classifier["buildings"].items():
        if only_filter and building not in only_filter:
            continue

        if cfg.get("skip_reason"):
            logging.info(f"[SKIP] {building}: {cfg['skip_reason']}")
            all_stats.append(ProcessingStats(
                building=building, variant=0,
                source_path="", source_size=(0, 0), output_size=(0, 0),
                bg_pixels_removed=0, bg_pixels_total=0, palette_size_used=0,
                output_path="",
                skipped=True, skip_reason=cfg["skip_reason"],
            ))
            continue

        target_size = tuple(cfg["png_size"])
        source_dir = effective_source / building
        if not source_dir.exists():
            logging.error(f"Source missing: {source_dir}")
            continue

        variant_pngs = sorted(source_dir.glob(f"{building}_*.png"))
        logging.info(f"[PROCESS] {building}: {len(variant_pngs)} variants → {target_size}")

        if args.dry_run:
            logging.info(f"  Would process: {[p.name for p in variant_pngs]}")
            continue

        for variant_png in variant_pngs:
            variant_idx = int(variant_png.stem.split("_")[-1])
            stats = process_one(
                source_png=variant_png,
                target_size=target_size,
                staging_dir=args.staging_root,
                building=building,
                variant=variant_idx,
                palette_size=palette_size,
                bg_tolerance=bg_tolerance,
                use_rembg=args.use_rembg_source,
                alpha_threshold=args.alpha_threshold,
            )
            all_stats.append(stats)
            logging.info(
                f"  variant {variant_idx}: "
                f"{stats.bg_pixels_removed}/{stats.bg_pixels_total} bg px, "
                f"palette {stats.palette_size_used}"
            )

    args.staging_root.mkdir(parents=True, exist_ok=True)
    report_path = args.staging_root / "_report.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "source_root": str(args.source_root),
        "staging_root": str(args.staging_root),
        "output_root": str(args.output_root),
        "palette_size": palette_size,
        "bg_tolerance": bg_tolerance,
        "stats": [asdict(s) for s in all_stats],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logging.info(f"Report → {report_path}")

    if not args.skip_classify and not args.dry_run:
        logging.info("Classifying to game asset tree (variant folders only, no representative file)...")
        written = classify_to_game_tree(
            staging_root=args.staging_root,
            output_root=args.output_root,
            classifier=classifier,
            force=args.force,
        )
        logging.info(f"Wrote {len(written)} files to game tree")

    return 0


if __name__ == "__main__":
    sys.exit(main())
