#!/usr/bin/env python3
"""Smoke test: generate a simple pixel art sprite via ComfyUI API."""
import json, time, urllib.request, sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from config import COMFYUI_URL


def queue_test_prompt():
    prompt_text = (
        "pixelart, pixel art, simple stone age campfire, "
        "top-down isometric view, stones and wood, centered, "
        "no background, game sprite"
    )
    neg_text = "blurry, realistic, 3d render, photo, text, watermark, frame"

    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["10", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt_text, "clip": ["10", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg_text, "clip": ["10", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "test_worldsim",
                "images": ["8", 0],
            },
        },
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "lora_name": "pixel-art-xl.safetensors",
                "strength_model": 0.8,
                "strength_clip": 0.8,
                "model": ["4", 0],
                "clip": ["4", 1],
            },
        },
    }

    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    return resp["prompt_id"]


def wait_for_completion(prompt_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        with urllib.request.urlopen(
            f"{COMFYUI_URL}/history/{prompt_id}"
        ) as r:
            data = json.loads(r.read())
        if prompt_id in data:
            return data[prompt_id]
        time.sleep(1)
    raise TimeoutError(f"Generation timeout after {timeout}s")


def main():
    # Check server
    try:
        with urllib.request.urlopen(
            f"{COMFYUI_URL}/system_stats", timeout=5
        ) as r:
            stats = json.loads(r.read())
        vram_gb = stats["devices"][0]["vram_total"] / 1e9
        print(f"ComfyUI server up — VRAM: {vram_gb:.0f}GB total")
    except Exception:
        print(f"ComfyUI server not responding at {COMFYUI_URL}")
        print("Start with: bash tools/comfyui/serve.sh")
        sys.exit(1)

    print("Queueing test generation...")
    pid = queue_test_prompt()
    print(f"  prompt_id: {pid}")

    print("Waiting for completion...")
    t0 = time.time()
    result = wait_for_completion(pid)
    elapsed = time.time() - t0

    outputs = result.get("outputs", {})
    for node_out in outputs.values():
        if "images" in node_out:
            for img in node_out["images"]:
                print(
                    f"Generated: ComfyUI/output/{img['filename']} "
                    f"({elapsed:.1f}s)"
                )

    print(f"\nSmoke test passed. Generation took {elapsed:.1f}s.")
    print(f"Check image at: ~/ComfyUI/output/test_worldsim_*.png")


if __name__ == "__main__":
    main()
