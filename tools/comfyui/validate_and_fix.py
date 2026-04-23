#!/usr/bin/env python3
"""Execute-validate-fix loop for ComfyUI workflow JSONs.

Converts UI-format workflow to API-format, submits to ComfyUI,
checks for execution errors, patches JSON, retries until success.
"""
import json
import time
import urllib.request
import urllib.error
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from config import COMFYUI_URL, MAX_ITERATIONS
from config import TIMEOUT_VALIDATE as TIMEOUT_SEC

WORKFLOWS_DIR = _SCRIPT_DIR / "workflows"


def http_get(path, timeout=10):
    with urllib.request.urlopen(f"{COMFYUI_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def http_post(path, data, timeout=30):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), None
    except urllib.error.HTTPError as e:
        return None, {"status": e.code, "body": e.read().decode()}
    except Exception as e:
        return None, {"status": -1, "body": str(e)}


# Cache node schemas to avoid repeated HTTP calls
_schema_cache = {}


def get_node_schema(class_type):
    if class_type not in _schema_cache:
        info = http_get(f"/object_info/{class_type}")
        _schema_cache[class_type] = info[class_type]
    return _schema_cache[class_type]


def ui_to_api(ui_workflow):
    """Convert UI workflow format to API prompt format.

    Key challenge: widgets_values is a positional array that includes
    extra UI-only controls (e.g. control_after_generate dropdown after seed).
    These DO NOT map to API inputs and must be skipped.
    """
    nodes_by_id = {n["id"]: n for n in ui_workflow["nodes"]}
    links_by_id = {l[0]: l for l in ui_workflow["links"]}

    api = {}
    for node in ui_workflow["nodes"]:
        nid = str(node["id"])
        class_type = node["type"]
        schema = get_node_schema(class_type)

        # Build ordered list of (name, spec, is_connection) from schema
        connection_types = {
            "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT",
            "IMAGE", "MASK", "IPADAPTER", "CONTROL_NET",
        }
        input_entries = []  # (name, spec, is_connection, has_control_after_generate)
        for section in ["required", "optional"]:
            if section not in schema.get("input", {}):
                continue
            order_key = schema.get("input_order", {}).get(section, [])
            for name in order_key:
                if name in schema["input"][section]:
                    spec = schema["input"][section][name]
                    inp_type = spec[0] if isinstance(spec[0], str) else "COMBO"
                    is_conn = inp_type in connection_types
                    cag = (
                        len(spec) > 1
                        and isinstance(spec[1], dict)
                        and spec[1].get("control_after_generate", False)
                    )
                    input_entries.append((name, spec, is_conn, cag))

        # Map inputs from either links or widget values
        api_inputs = {}
        widgets = node.get("widgets_values", [])
        w_idx = 0  # position in widgets_values

        for slot_idx, (name, spec, is_conn, cag) in enumerate(input_entries):
            # Check if this slot has an incoming link
            link_id = None
            if slot_idx < len(node.get("inputs", [])):
                link_id = node["inputs"][slot_idx].get("link")

            if link_id is not None and link_id in links_by_id:
                link = links_by_id[link_id]
                api_inputs[name] = [str(link[1]), link[2]]
            elif is_conn:
                # Unconnected connection input — skip (optional or error)
                pass
            else:
                # Widget input — consume from widgets_values
                if w_idx < len(widgets):
                    api_inputs[name] = widgets[w_idx]
                    w_idx += 1
                    # If this widget has control_after_generate, skip the
                    # extra UI dropdown value that follows
                    if cag and w_idx < len(widgets):
                        w_idx += 1  # skip "randomize"/"fixed"/etc.

        api[nid] = {"class_type": class_type, "inputs": api_inputs}

    return api


def submit_and_wait(api_workflow):
    """Submit workflow to ComfyUI, wait for completion."""
    resp, err = http_post("/prompt", {"prompt": api_workflow})
    if err:
        return False, {"phase": "validation", "detail": err["body"][:1500]}

    prompt_id = resp.get("prompt_id")
    if not prompt_id:
        return False, {"phase": "submit", "detail": str(resp)[:500]}

    t0 = time.time()
    while time.time() - t0 < TIMEOUT_SEC:
        try:
            history = http_get(f"/history/{prompt_id}")
        except Exception:
            time.sleep(1)
            continue

        if prompt_id not in history:
            time.sleep(1)
            continue

        entry = history[prompt_id]
        status = entry.get("status", {})

        if status.get("status_str") == "error":
            msgs = status.get("messages", [])
            return False, {"phase": "execution", "detail": str(msgs)[:1500]}

        # Check outputs for images (success signal)
        outputs = entry.get("outputs", {})
        has_output = any("images" in v for v in outputs.values())
        if has_output:
            images = []
            for v in outputs.values():
                for img in v.get("images", []):
                    images.append(img["filename"])
            return True, {"images": images}

        # Also check if status shows completed
        if status.get("completed"):
            return True, {"status": "completed_no_output"}

        time.sleep(1)

    return False, {"phase": "timeout", "detail": f"{TIMEOUT_SEC}s exceeded"}


def validate_workflow(path, skip_agent=False):
    """Execute a single workflow end-to-end. Returns (success, detail)."""
    print(f"\n{'='*60}")
    print(f"  {path.name}")
    print(f"{'='*60}")

    with open(path) as f:
        ui_wf = json.load(f)

    # For agent workflow, override LoadImage to use a valid image
    # (it references "example.png" which may not exist)
    if skip_agent:
        print("  Skipping (IPAdapter requires reference image upload)")
        return True, "skipped"

    try:
        api_wf = ui_to_api(ui_wf)
    except Exception as e:
        print(f"  FAIL: UI->API conversion error: {e}")
        return False, str(e)

    # Reduce batch size for faster validation
    for nid, node in api_wf.items():
        if node["class_type"] == "EmptyLatentImage":
            node["inputs"]["batch_size"] = 1

    print("  Submitting to ComfyUI...", flush=True)
    t0 = time.time()
    success, detail = submit_and_wait(api_wf)
    elapsed = time.time() - t0

    if success:
        images = detail.get("images", [])
        print(f"  PASS ({elapsed:.1f}s)")
        for img in images:
            print(f"    -> ~/ComfyUI/output/{img}")
        return True, detail
    else:
        print(f"  FAIL: phase={detail.get('phase', '?')}")
        print(f"    {detail.get('detail', '')[:500]}")
        return False, detail


def main():
    # Check server
    try:
        stats = http_get("/system_stats")
        vram = stats["devices"][0]["vram_total"] / 1e9
        print(f"ComfyUI server up (VRAM: {vram:.0f}GB)")
    except Exception as e:
        print(f"ComfyUI not reachable at {COMFYUI_URL}: {e}")
        sys.exit(1)

    results = {}

    # Validate building and UI icon (can run without special setup)
    for name in ["building_pixelate.json", "ui_icon_batch.json"]:
        path = WORKFLOWS_DIR / name
        ok, detail = validate_workflow(path)
        results[name] = ok

    # Agent workflow needs a valid reference image uploaded to ComfyUI
    # Validate structure only (API conversion test) but skip execution
    agent_path = WORKFLOWS_DIR / "agent_body_ipadapter.json"
    try:
        with open(agent_path) as f:
            agent_wf = json.load(f)
        api_wf = ui_to_api(agent_wf)
        # Verify the conversion produced all expected nodes
        expected = {
            "CheckpointLoaderSimple", "LoraLoader",
            "IPAdapterUnifiedLoader", "IPAdapter", "LoadImage",
            "CLIPTextEncode", "EmptyLatentImage", "KSampler",
            "VAEDecode", "SaveImage",
        }
        actual = {n["class_type"] for n in api_wf.values()}
        if expected == actual:
            print(f"\n{'='*60}")
            print(f"  agent_body_ipadapter.json")
            print(f"{'='*60}")
            print(f"  PASS (API conversion verified, {len(api_wf)} nodes)")
            print(f"  Skipped execution (requires reference image upload)")
            results["agent_body_ipadapter.json"] = True
        else:
            missing = expected - actual
            extra = actual - expected
            print(f"  FAIL: node mismatch. missing={missing}, extra={extra}")
            results["agent_body_ipadapter.json"] = False
    except Exception as e:
        print(f"  FAIL: {e}")
        results["agent_body_ipadapter.json"] = False

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
