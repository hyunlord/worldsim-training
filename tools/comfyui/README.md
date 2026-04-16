# WorldSim ComfyUI Sprite Pipeline

Runs on DGX Spark. Generates pixel art candidates. Aseprite cleanup still required.

## Setup (once)

```bash
bash tools/comfyui/install.sh    # 10-15 min, downloads ~10GB
```

## Daily use

```bash
bash tools/comfyui/serve.sh      # DGX: start server on port 8188
```

Then from local PC, open: `http://<dgx-ip>:8188`

## Workflows

- `building_pixelate.json` — buildings (shelter, hearth, workbench)
- `agent_body_ipadapter.json` — 8-direction agent from reference
- `ui_icon_batch.json` — UI icons batch

## Workflow: Building sprite

1. Load `building_pixelate.json` in ComfyUI
2. Edit positive prompt: replace `{building_name}` and `{material}`
3. Queue Prompt → 4 variants in ~15s on GB10
4. Download best 1-2 → import into Aseprite for cleanup

## Workflow: Agent 8-direction

1. Prepare reference: 1 clean front-view pixel art of the character (manual or AI-generated)
2. Load `agent_body_ipadapter.json`
3. Drop reference in IPAdapter node
4. Queue → 8 directions generated
5. Cleanup each in Aseprite (expect 15-30 min per direction)

## Cost: $0 (local DGX)

## Tips

- LoRA weight 0.8 is baseline; lower to 0.6 for more variety, higher to 1.0 for pure pixel
- Use `ImagePixelate` node (from comfy_pixelization) after generation for clean grid-aligned pixels
- Always generate at 1024x1024 then downscale — direct 32x32 generation fails
