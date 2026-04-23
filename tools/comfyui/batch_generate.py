#!/usr/bin/env python3
"""Batch sprite generation driver for ComfyUI.

Feeds a YAML prompt catalog through a ComfyUI workflow, generating
multiple variants per building concept.  Manages the local ComfyUI
server lifecycle, handles retries, and produces a contact-sheet HTML
plus zip archive for triage.

Usage:
    python tools/comfyui/batch_generate.py \
      --config tools/comfyui/configs/buildings.yaml \
      --workflow tools/comfyui/workflows/building_pixelate.json \
      --output-root assets/sprites/concepts
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure lib/ and the script's own directory are importable.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lib.comfy_client import ComfyClient, ComfyClientError
from lib.prompt_catalog import PromptCatalog, CatalogError
from lib.server_manager import ServerManager, ServerStartupError
from lib.contact_sheet import build_thumbnails, render_html, create_zip
from validate_and_fix import (
    ui_to_api,
    validate_workflow,
    validate_after_error,
    ValidationError,
)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch generate pixel-art building sprites via ComfyUI",
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="YAML prompt catalog (e.g. configs/buildings.yaml)",
    )
    p.add_argument(
        "--workflow", required=True, type=Path,
        help="ComfyUI workflow JSON (e.g. workflows/building_pixelate.json)",
    )
    p.add_argument(
        "--output-root", required=True, type=Path,
        help="Output directory for generated sprites",
    )
    p.add_argument(
        "--comfy-host", default="127.0.0.1",
        help="ComfyUI host (default: 127.0.0.1)",
    )
    p.add_argument(
        "--comfy-port", type=int, default=8188,
        help="ComfyUI port (default: 8188)",
    )
    p.add_argument(
        "--only", default=None,
        help="Comma-separated building names to process (default: all)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Regenerate even if variants already exist",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without generating",
    )
    p.add_argument(
        "--no-contact-sheet", action="store_true",
        help="Skip HTML contact sheet generation",
    )
    p.add_argument(
        "--no-zip", action="store_true",
        help="Skip zip archive creation",
    )
    p.add_argument(
        "--no-auto-start", action="store_true",
        help="Fail fast if server is down (don't auto-start)",
    )
    return p.parse_args()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def compute_workflow_hash(workflow_path: Path) -> str:
    """SHA-256 of the canonical JSON, truncated to 12 hex chars."""
    with open(workflow_path) as f:
        raw = json.load(f)
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _find_nodes(api_wf: dict, class_type: str) -> list[tuple[str, dict]]:
    """Return ``[(node_id, node_dict)]`` for every node of *class_type*."""
    return [
        (nid, node)
        for nid, node in api_wf.items()
        if node.get("class_type") == class_type
    ]


def patch_workflow(api_workflow: dict, building, workflow_hash: str) -> dict:
    """Deep-copy the API workflow and patch it for *building*."""
    wf = copy.deepcopy(api_workflow)

    # --- KSampler: seed -------------------------------------------------
    ksamplers = _find_nodes(wf, "KSampler")
    if not ksamplers:
        raise ValueError("No KSampler node found in workflow")
    _ks_id, ks_node = ksamplers[0]
    ks_node["inputs"]["seed"] = building.seed

    # --- CLIPTextEncode: positive / negative text -----------------------
    #   Identify which CLIPTextEncode is wired to KSampler.positive vs
    #   .negative by following the link references.
    pos_ref = ks_node["inputs"].get("positive")  # e.g. ["3", 0]
    neg_ref = ks_node["inputs"].get("negative")  # e.g. ["4", 0]

    if pos_ref and isinstance(pos_ref, list):
        pos_nid = str(pos_ref[0])
        if pos_nid in wf:
            wf[pos_nid]["inputs"]["text"] = building.positive

    if neg_ref and isinstance(neg_ref, list):
        neg_nid = str(neg_ref[0])
        if neg_nid in wf:
            wf[neg_nid]["inputs"]["text"] = building.negative

    # --- EmptyLatentImage: batch_size -----------------------------------
    for _, eli in _find_nodes(wf, "EmptyLatentImage"):
        eli["inputs"]["batch_size"] = building.batch_size

    # --- SaveImage: filename_prefix -------------------------------------
    for _, sav in _find_nodes(wf, "SaveImage"):
        sav["inputs"]["filename_prefix"] = building.name

    return wf


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

MAX_ATTEMPTS = 5


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.comfy_host}:{args.comfy_port}"

    # ── Load workflow ───────────────────────────────────────────────
    if not args.workflow.exists():
        print(f"ERROR: Workflow not found: {args.workflow}", file=sys.stderr)
        sys.exit(2)
    workflow_hash = compute_workflow_hash(args.workflow)
    with open(args.workflow) as f:
        ui_workflow = json.load(f)

    # ── Load prompt catalog ─────────────────────────────────────────
    try:
        catalog = PromptCatalog.load(args.config)
    except CatalogError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # ── Filter buildings ────────────────────────────────────────────
    buildings = catalog.buildings()
    if args.only:
        only_names = [n.strip() for n in args.only.split(",")]
        available = catalog.names()
        unknown = [n for n in only_names if n not in available]
        if unknown:
            print(
                f"ERROR: Unknown buildings: {unknown}. "
                f"Available: {available}",
                file=sys.stderr,
            )
            sys.exit(2)
        buildings = [b for b in buildings if b.name in only_names]

    # ── Setup logging ───────────────────────────────────────────────
    print(f"[setup] Workflow hash: {workflow_hash} ({args.workflow.name})")
    print(f"[setup] {len(buildings)} buildings queued, output → {args.output_root}")

    # ── Dry-run mode ────────────────────────────────────────────────
    if args.dry_run:
        print("\n[dry-run] Plan:")
        for i, b in enumerate(buildings, 1):
            td = args.output_root / b.name
            existing = len(list(td.glob("*.png"))) if td.exists() else 0
            if existing >= b.batch_size and not args.force:
                status = f"skip ({existing}/{b.batch_size} present)"
            else:
                status = "generate"
            print(
                f"  [{i}/{len(buildings)}] {b.name:<20s} "
                f"seed={b.seed} batch={b.batch_size} → {status}"
            )
        print("\n[dry-run] No generation performed. Remove --dry-run to execute.")
        sys.exit(0)

    # ── Server ──────────────────────────────────────────────────────
    mgr = ServerManager(base_url=base_url)
    print(f"[setup] Checking ComfyUI server at {base_url}...")
    if args.no_auto_start:
        if not mgr.is_alive():
            print(
                f"ERROR: ComfyUI server not responding at {base_url}.\n"
                f"Start manually:  bash tools/comfyui/serve.sh\n"
                f"Or remove --no-auto-start to auto-launch.",
                file=sys.stderr,
            )
            sys.exit(2)
        print("[setup] Server is up")
    else:
        try:
            mgr.ensure_running()
        except ServerStartupError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(2)

    # ── Prepare client + convert workflow once ──────────────────────
    client = ComfyClient(base_url=base_url)
    comfy_version = mgr.server_version()
    api_workflow = ui_to_api(ui_workflow)

    # ── Main generation loop ────────────────────────────────────────
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, str, float, int]] = []  # (name, status, elapsed, attempts)
    total_start = time.time()

    for idx, building in enumerate(buildings, 1):
        target_dir = args.output_root / building.name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already complete
        existing_pngs = list(target_dir.glob("*.png"))
        if len(existing_pngs) >= building.batch_size and not args.force:
            print(
                f"[{idx}/{len(buildings)}] {building.name:<20s} "
                f"skipped ({len(existing_pngs)}/{building.batch_size} "
                f"already present, use --force to regenerate)"
            )
            results.append((building.name, "skipped", 0, 0))
            continue

        # Remove old PNGs when --force is used
        if args.force:
            for old in existing_pngs:
                old.unlink()

        print(
            f"[{idx}/{len(buildings)}] {building.name:<20s} "
            f"seed={building.seed} batch={building.batch_size} ... ",
            end="",
            flush=True,
        )

        # Patch workflow for this building
        patched = patch_workflow(api_workflow, building, workflow_hash)

        # Initial validation
        try:
            patched = validate_workflow(patched, base_url)
        except ValidationError as e:
            print(f"VALIDATION ERROR: {e}")
            results.append((building.name, "failed", 0, 0))
            continue

        # Retry loop
        b_start = time.time()
        success = False
        last_error = ""
        downloaded: list[str] = []
        attempts = 0

        for attempt in range(1, MAX_ATTEMPTS + 1):
            attempts = attempt
            try:
                prompt_id = client.queue_prompt(patched)
                entry = client.wait_for_completion(prompt_id, timeout_s=300)

                # Download outputs
                images = client.get_output_images(entry)
                downloaded = []
                for i, img_info in enumerate(images, 1):
                    dest = target_dir / f"{building.name}_{i:03d}.png"
                    client.download_output(
                        filename=img_info["filename"],
                        subfolder=img_info["subfolder"],
                        type_=img_info["type"],
                        dest_path=dest,
                    )
                    downloaded.append(dest.name)

                success = True
                break

            except (ComfyClientError, ValidationError, Exception) as e:
                last_error = str(e)
                if attempt < MAX_ATTEMPTS:
                    print(
                        f"\n{'':>30s}retry {attempt}: "
                        f"{str(e)[:80]}"
                    )
                    try:
                        patched = validate_after_error(
                            patched, str(e), base_url
                        )
                    except ValidationError:
                        pass  # continue with same workflow
                    time.sleep(5)

        elapsed = time.time() - b_start

        if success:
            # Write _meta.json
            meta = {
                "building": building.name,
                "seed": building.seed,
                "batch_size": building.batch_size,
                "positive_prompt": building.positive,
                "negative_prompt": building.negative,
                "notes": building.notes,
                "workflow_hash": workflow_hash,
                "workflow_file": args.workflow.name,
                "generated_at": datetime.now(timezone.utc)
                .astimezone()
                .isoformat(),
                "comfy_host": base_url,
                "comfy_version": comfy_version,
                "elapsed_seconds": round(elapsed, 1),
                "attempts": attempts,
                "outputs": downloaded,
            }
            meta_path = target_dir / "_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            suffix = f"({attempts} attempt{'s' if attempts > 1 else ''})"
            print(f"completed in {elapsed:.1f}s {suffix} \u2713")
            results.append((building.name, "succeeded", elapsed, attempts))
        else:
            print(
                f"FAILED after {attempts} attempts ({elapsed:.1f}s): "
                f"{last_error[:100]}"
            )
            results.append((building.name, "failed", elapsed, attempts))

    # ── Post-processing ─────────────────────────────────────────────
    succeeded = sum(1 for _, s, _, _ in results if s == "succeeded")
    failed = sum(1 for _, s, _, _ in results if s == "failed")
    skipped = sum(1 for _, s, _, _ in results if s == "skipped")

    # Contact sheet
    if not args.no_contact_sheet and succeeded + skipped > 0:
        print("\n[post] Generating contact sheet...")
        thumb_dir = args.output_root / "_contact_sheet_thumbs"
        try:
            entries = build_thumbnails(args.output_root, thumb_dir)
            template_path = _SCRIPT_DIR / "templates" / "contact_sheet.html.j2"
            html_path = args.output_root / "_contact_sheet.html"
            render_html(
                entries=entries,
                template_path=template_path,
                output_html=html_path,
                title="WorldSim Building Concepts",
                generated_at=datetime.now(),
                workflow_hash=workflow_hash,
            )
            print(f"[post] {len(entries)} thumbnails created (256\u00d7256 webp)")
            print(f"[post] HTML written \u2192 {html_path}")
        except Exception as e:
            print(f"[post] Contact sheet generation failed: {e}")

    # Zip archive
    if not args.no_zip and succeeded + skipped > 0:
        print("[post] Creating zip archive...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = args.output_root / "_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        zip_path = archive_dir / f"concepts_{timestamp}.zip"
        try:
            total_bytes = create_zip(
                args.output_root, zip_path, include_thumbs=False,
            )
            size_mb = total_bytes / (1024 * 1024)
            print(f"[post] Zip written \u2192 {zip_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"[post] Zip creation failed: {e}")

    # ── Summary ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\n=== Summary ===")
    print(f"  {succeeded} succeeded, {failed} failed, {skipped} skipped")
    print(f"  Total elapsed: {mins}m {secs}s")
    print(f"  Output root:   {args.output_root}")
    html_path = args.output_root / "_contact_sheet.html"
    if html_path.exists():
        print(f"  Contact sheet: {html_path}")
    archive_dir = args.output_root / "_archive"
    if archive_dir.exists():
        zips = sorted(archive_dir.glob("*.zip"))
        if zips:
            print(f"  Zip archive:   {zips[-1]}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
