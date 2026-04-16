#!/usr/bin/env python3
"""Validate all 3 workflow JSONs — structure, node types, links."""
import json
import sys
from pathlib import Path

WORKFLOWS_DIR = Path(__file__).parent / "workflows"


def validate_ui_workflow(path):
    with open(path) as f:
        wf = json.load(f)

    # Required top-level keys for UI format
    required = ["last_node_id", "last_link_id", "nodes", "links", "version"]
    missing = [k for k in required if k not in wf]
    if missing:
        return False, f"Missing keys: {missing}"

    if not wf["nodes"]:
        return False, "No nodes"

    # Each node must have id, type, pos
    for node in wf["nodes"]:
        for key in ["id", "type", "pos"]:
            if key not in node:
                return False, f"Node missing {key}: {node.get('id', '?')}"

    # Collect node IDs
    node_ids = {n["id"] for n in wf["nodes"]}

    # Validate links reference real nodes
    for link in wf["links"]:
        if len(link) < 6:
            return False, f"Link malformed: {link}"
        link_id, from_node, from_slot, to_node, to_slot, link_type = link
        if from_node not in node_ids:
            return False, f"Link {link_id}: from_node {from_node} not found"
        if to_node not in node_ids:
            return False, f"Link {link_id}: to_node {to_node} not found"

    # Check for required node types
    types = {n["type"] for n in wf["nodes"]}
    required_types = {"CheckpointLoaderSimple", "KSampler", "VAEDecode", "SaveImage"}
    missing_types = required_types - types
    if missing_types:
        return False, f"Missing required node types: {missing_types}"

    # Verify link consistency: node inputs/outputs reference correct link IDs
    link_ids = {link[0] for link in wf["links"]}
    for node in wf["nodes"]:
        for inp in node.get("inputs", []):
            if inp.get("link") is not None and inp["link"] not in link_ids:
                return False, f"Node {node['id']} input '{inp['name']}' references non-existent link {inp['link']}"
        for out in node.get("outputs", []):
            for lid in out.get("links", []):
                if lid not in link_ids:
                    return False, f"Node {node['id']} output '{out['name']}' references non-existent link {lid}"

    return True, f"OK ({len(wf['nodes'])} nodes, {len(wf['links'])} links, types: {sorted(types)})"


def main():
    files = [
        "building_pixelate.json",
        "ui_icon_batch.json",
        "agent_body_ipadapter.json",
    ]
    all_ok = True
    for name in files:
        path = WORKFLOWS_DIR / name
        if not path.exists():
            print(f"FAIL {name}: FILE NOT FOUND")
            all_ok = False
            continue
        ok, msg = validate_ui_workflow(path)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name}: {msg}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll 3 workflow JSONs valid")
        sys.exit(0)
    else:
        print("\nValidation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
