# Archived ComfyUI Workflows

Preserved for reference. Not used by any active pipeline.

## Files

### ui_icon_batch.json
- **Status**: Never executed in production
- **Original plan**: 51 UI icons via SDXL + Pixel Art LoRA, 32×32 output
- **Why archived**: UI icon work not started. Tier 1 MVP prioritized buildings
  + walls/floors + furniture. UI icons may be generated later or replaced by
  a different approach (vector, manual pixel art, or GPT Image for text-heavy icons).

### agent_body_ipadapter.json
- **Status**: Never executed in production
- **Original plan**: 8-direction agent body via IPAdapter reference + pose conditioning
- **Why archived**: Tier 2 (agent modular sprites) deprecated 2026-04-21. Discovered
  existing `palette_swap.gdshader` system (16×24 cell, G-channel tiers) provides
  256 combinations without additional sprite layers. Decision: extend palette
  shader (5-tier Shader v2) instead of pre-rendered modular approach.

### building_flux_schnell.json
- **Status**: Used only during Experiment B benchmark (2026-04-17 to 2026-04-19)
- **Why archived**: FLUX.1 schnell rejected. See `../configs/_archive/experiment_b.yaml`
  for detailed benchmark findings.

## Active workflow

Location: parent directory (`..`)
- `building_pixelate.json` — Production workflow for all building/wall/floor/
  furniture generation. SDXL + Pixel Art XL LoRA + comfy_pixelization.
  Used in Round 1 (buildings), Round 2 (walls/floors), Round 3 (furniture).
