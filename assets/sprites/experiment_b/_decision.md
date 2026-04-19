# Experiment B — FLUX.1 schnell Migration Decision

**Date**: _____ (fill in)
**Decider**: _____ (fill in)
**Scope tested**: Option B — raw FLUX schnell (no LoRA) vs SDXL+rembg (current production)

---

## Data inputs (to inform decision)

### Visual comparison
Open `_comparison.html` at http://100.70.109.50:8000/experiment_b/_comparison.html.
3 columns: `baseline_sdxl_rembg`, `flux_raw`, `flux_rembg`.

### Quantitative signals (variant 1 of each building)
| Signal | SDXL+rembg | FLUX raw | Interpretation |
|---|:---:|:---:|---|
| Unique colors / image | ~12,000 | ~43,000 (**~4×**) | FLUX has no pixel-art palette constraint without a style LoRA |
| Transparent pixels % | 40–86% | 0% | FLUX raw has no alpha masking — needs rembg too |
| Wall-clock seconds/image | 29.3s | 33.2s (**+13%**) | FLUX is NOT faster than SDXL at batch_size 4 (model size dominates) |

### LoRA ecosystem (HF search 2026-04-19)
- 32 FLUX pixel-art LoRAs found
- **All top candidates (likes≥29) are trained on FLUX.1 dev** (non-commercial license — using them on schnell base is legally OK for PNG distribution but quality transfer uncertain)
- Only one schnell-native pixel-art LoRA exists: `mindlywork/Pixel_Art_FLUX` (0 likes, 14 downloads — essentially unvalidated)

---

## Decision (pick one)

### ☐ A. Migrate to FLUX for Round 2+
Trigger conditions (visual):
- FLUX raw or FLUX+rembg clearly beats SDXL+rembg on 3+ of the 5 buildings
- FLUX solves the workbench structural failure (TABLE concept vs solid rock)
- FLUX produces isolated objects natively (no landscape backdrop)

Follow-ups:
- Extend to Option A: evaluate 2-3 dev-trained LoRAs on schnell (likely quality drop risk)
- Migrate walls.yaml / floors.yaml to FLUX workflow
- Assess Round 3 agent modular layers on FLUX

### ☐ B. Keep SDXL+rembg (current production), revisit in 6 months
Trigger conditions:
- FLUX raw is photorealistic/over-detailed (unique color count confirms)
- FLUX doesn't solve structural failures any better than SDXL
- Schnell LoRA ecosystem still too thin to commit

Documentation:
- Note re-evaluation target date: 2026-10-19
- Trigger for re-evaluation: (1) new schnell pixel-art LoRA with >100 likes, OR (2) FLUX.1 dev license change to permissive

### ☐ C. Hybrid — SDXL for Round 1 objects, FLUX for specific Round 3 needs
Trigger conditions:
- FLUX+rembg shows clear win ONLY on agent-style coherent multi-frame content
- Object pixel art stays on SDXL because LoRA ecosystem + palette is mature

---

## Author's verdict
_____________________________________________
_____________________________________________

## Next action
_____________________________________________
