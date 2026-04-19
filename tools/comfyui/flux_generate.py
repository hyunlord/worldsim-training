#!/usr/bin/env python3
"""Lightweight FLUX runner for Experiment B.

Bypasses batch_generate.py's UI→API conversion (which is SDXL-shaped) and
submits FLUX workflows directly in API format. Reuses ComfyClient.

Usage:
    python tools/comfyui/flux_generate.py \\
        --config tools/comfyui/configs/experiment_b.yaml \\
        --workflow tools/comfyui/workflows/building_flux_schnell.json \\
        --output-root assets/sprites/experiment_b/flux_raw
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from lib.comfy_client import ComfyClient, ComfyClientError
from lib.prompt_catalog import PromptCatalog


def patch_flux_workflow(wf: dict, building, negative: str) -> dict:
    """Patch seed, positive text, negative text, batch_size, filename_prefix."""
    out = copy.deepcopy(wf)
    # Drop meta keys (anything not a node id)
    out = {k: v for k, v in out.items() if not k.startswith("_")}

    # Node 4: positive CLIPTextEncode
    out["4"]["inputs"]["text"] = building.positive
    # Node 5: negative CLIPTextEncode (cfg=1 -> inert, but keep populated for parity)
    out["5"]["inputs"]["text"] = negative
    # Node 6: EmptySD3LatentImage
    out["6"]["inputs"]["batch_size"] = building.batch_size
    # Node 7: KSampler
    out["7"]["inputs"]["seed"] = building.seed
    # Node 9: SaveImage
    out["9"]["inputs"]["filename_prefix"] = building.name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--only", default="", help="Comma-separated building names")
    ap.add_argument("--comfy-host", default="127.0.0.1")
    ap.add_argument("--comfy-port", type=int, default=8188)
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-prompt completion timeout in seconds")
    args = ap.parse_args()

    client = ComfyClient(f"http://{args.comfy_host}:{args.comfy_port}")
    catalog = PromptCatalog.load(args.config)
    wf_template = json.loads(args.workflow.read_text())
    only_filter = set(s.strip() for s in args.only.split(",") if s.strip())

    args.output_root.mkdir(parents=True, exist_ok=True)

    timings = []
    for b in catalog.buildings():
        if only_filter and b.name not in only_filter:
            continue

        print(f"[{b.name}] seed={b.seed} batch={b.batch_size} ...", flush=True)
        t0 = time.time()
        patched = patch_flux_workflow(wf_template, b, b.negative)
        try:
            prompt_id = client.queue_prompt(patched)
            entry = client.wait_for_completion(prompt_id, timeout_s=args.timeout)
        except ComfyClientError as e:
            print(f"  FAIL: {e}", flush=True)
            timings.append({"building": b.name, "elapsed_s": None, "error": str(e)})
            continue

        images = client.get_output_images(entry)
        bldg_dir = args.output_root / b.name
        bldg_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images, start=1):
            dest = bldg_dir / f"{b.name}_{idx:03d}.png"
            client.download_output(img["filename"], img["subfolder"], img["type"], dest)
        elapsed = time.time() - t0
        print(f"  {len(images)} imgs in {elapsed:.1f}s", flush=True)
        timings.append({
            "building": b.name,
            "seed": b.seed,
            "batch_size": b.batch_size,
            "elapsed_s": round(elapsed, 2),
            "images": [i["filename"] for i in images],
        })

    (args.output_root / "_timing.json").write_text(
        json.dumps({"timings": timings, "total_s": sum(
            t["elapsed_s"] or 0 for t in timings
        )}, indent=2, ensure_ascii=False)
    )
    print(f"\nTiming log → {args.output_root / '_timing.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
