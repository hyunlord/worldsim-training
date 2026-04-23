# Archived ComfyUI Configs

Preserved for historical reference. Not referenced by any active script.

## Files

### buildings_v2.yaml
- **Date archived**: 2026-04-23
- **Reason**: Superseded by `../buildings.yaml` (v3 FINAL, 2026-04-18)
- **History**: v2 was the first successful stone-age building prompt catalog.
  v3 kept 8/10 buildings unchanged (v2 hit rate ≥60%) and rewrote drying_rack
  + workbench with targeted fixes for their specific failure modes.

### experiment_b.yaml
- **Date archived**: 2026-04-23
- **Reason**: FLUX.1 schnell benchmark rejected (2026-04-19)
- **Findings**:
  - FLUX raw palette 36,000 colors vs SDXL+rembg 17,000 — worse for 16-color target
  - FLUX 33.2s/img vs SDXL 29.3s/img — 12B params offset step reduction
  - schnell-native pixel-art LoRA ecosystem immature (0 available)
- **Re-evaluation trigger**: 2026-10-19, OR schnell LoRA pool grows, OR dev license loosens

### walls_v1.yaml
- **Date archived**: 2026-04-23
- **Reason**: Superseded by `../walls.yaml` (v2, Round 2 final)
- **History**: Initial wall material prompts. v2 rewrote prompts to target
  seamless tileable texture instead of perspective object view.

### floors_v1.yaml
- **Date archived**: 2026-04-23
- **Reason**: Superseded by `../floors.yaml` (v2, Round 2 final)
- **History**: Initial floor material prompts. Same v2 rewrite rationale as walls.

## Active configs

Location: parent directory (`..`)
- `buildings.yaml` — 10 stone-age buildings (Round 1, v3 FINAL) + Round 3 additions
- `walls.yaml` — 7 wall materials (Round 2)
- `floors.yaml` — 3 floor materials (Round 2)
