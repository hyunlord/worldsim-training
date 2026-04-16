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

## Workflows (drag into ComfyUI UI)

The 3 JSON files in `tools/comfyui/workflows/` are full node graphs.

### How to load
1. Start ComfyUI: `bash tools/comfyui/serve.sh`
2. Open http://<dgx-ip>:8188 in browser
3. Drag a `.json` file from File Explorer directly onto the ComfyUI canvas
4. OR: Click the workflow sidebar -> Load -> select file

### 1. building_pixelate.json
- Input: edit the positive prompt node to describe the building
- Output: 4 images at ~/ComfyUI/output/worldsim_building_*.png (pixelated after generation)
- Usage: replace "campfire" in prompt with "shelter", "hearth", "workbench", etc.
- Generation time: ~33s per batch of 4 on GB10

### 2. ui_icon_batch.json
- Same pipeline as building but with icon-specific prompt template
- Replace "wood axe" with "berry basket", "stone knife", etc.

### 3. agent_body_ipadapter.json
- Requires: a reference character image loaded into the "Load Image" node
- Generate one direction at a time by editing prompt: "from east", "from west", "from north", etc.
- 8 runs per character for full rotation
- IPAdapter weight 0.7 for consistency with prompt guidance

## Aseprite cleanup (after AI generation)
1. Open generated PNG in Aseprite
2. Sprite > Sprite Size > 32x32 (use "nearest" algorithm)
3. Sprite > Color Mode > Indexed (reduces palette)
4. Manual touch-up: outlines, anti-alias pixels
5. Save to `assets/sprites/structures/{name}.png`

## Cost: $0 (local DGX)

## Tips

- LoRA weight 0.8 is baseline; lower to 0.6 for more variety, higher to 1.0 for pure pixel
- Pixelization node (from comfy_pixelization) uses neural-network-based pixel art conversion
- Always generate at 1024x1024 then pixelate — direct 32x32 generation fails
- Change the seed in KSampler for different variations
