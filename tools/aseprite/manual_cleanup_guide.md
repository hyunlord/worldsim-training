# Aseprite Manual Cleanup Guide — Phase 2

## Prerequisite
Phase 1 (`tools/aseprite/auto_process.py`) has completed. Your working files are at:
- `_raw_cleanup_stage/03_palette_reduced/{building}/1.png ~ 16.png` (the auto-processed outputs)
- `assets/sprites/{category}/{building}/1.png ~ 16.png` (copies already deployed — edit in place here)

**Key fact from 3rd cross-session handoff**: The game uses a **variant loader** that reads all variants via `{building}/{(entity_id % count) + 1}.png`. There is NO representative file. **All variants must be shippable quality** — you can't "pick the best and ignore the rest".

**Contiguity requirement**: Files MUST be named `1.png, 2.png, ..., N.png` with no gaps. If you delete a bad variant (e.g. 3.png), the loader caches null for that index and any entity whose seed maps to 3 will have no sprite. Either keep all N or remove and re-number (e.g. delete 3.png, rename 4..16 → 3..15, adjust file count).

## Per-file workflow (5-10 min each)

### Step 1 — Open and inspect
1. Aseprite → File → Open → `assets/sprites/{category}/{building}/1.png` (or any variant)
   - Open all 16 variants in tabs so you can switch quickly
2. Zoom in to 800% or higher for pixel-level work
3. Toggle transparency checkerboard: View → Show Pixel Grid + Show Checkerboard

### Step 2 — Refine alpha (most time-consuming)
The auto flood-fill is 80% accurate but misses:
- Shadows under the object that got left behind
- Semi-transparent AI-generated halos around edges
- Internal background pockets (e.g., sky visible through a gap in a shelter roof)

Tools:
- **Magic wand (W)** with contiguous=ON to select leftover background patches
- **Eraser (E)** with pixel-perfect mode for edge cleanup
- Alpha threshold: pixels should be either 0 or 255 (hard edges only, no anti-aliasing)

Verification: press `Ctrl+A` to select all, check for stray semi-transparent pixels.

### Step 3 — Outline touch-up
Game-dev session specified **1px dark outline** around the object. The Pixel Art XL LoRA gives a soft outline, but it's inconsistent.

**Scope note**: This 1px outline applies ONLY to building/furniture object sprites (what we're processing now). Wall/floor *tile* sprites (Round 2, future) must NOT have an outline — game code draws outlines for tiles.

Technique:
1. Select all opaque pixels: Select → All Opaque Pixels (or Magic Wand on alpha)
2. Select → Modify → Border, width=1
3. Fill selection with darkest palette color (usually a deep brown like `#2b1810`)

Skip this step for objects where the outline is already crisp (campfire fire pit, for instance).

### Step 4 — Pivot verification
The sprite's ground-center must align with the image center. For top-down viewed objects this usually means the bottom-most row of the object should be around y=2/3 of the image height.

Visual check:
- Draw a temporary cross at the image center (View → Grid with 16×16 spacing helps)
- The object's base/ground contact should sit at the horizontal centerline and near the bottom third vertically
- Move the object with Cel → Move or Ctrl+arrow if needed

### Step 5 — Save
- Save over the current file (don't Save As — the path is already correct)
- Move to next variant

No representative-file copy step. The game reads the variant folder directly.

## Variant processing strategy

Because all variants ship, the question is **"which variants need the most work?"** not "which is the best?".

Triage per building:
1. Scan all 16 variants in a grid view first (Aseprite File → Open All, arrange tabs)
2. Rank them by auto-output quality:
   - **Green**: minimal touch-up needed (5 min) — probably Aseprite Steps 2 + 4 only
   - **Yellow**: moderate work (8-10 min) — all steps, but auto flood-fill worked OK
   - **Red**: significant rework (15+ min) — broken alpha, missing outline, off-center pivot
3. Process in order: Green → Yellow → Red
4. If a Red variant is genuinely unsalvageable (AI garbage), **delete it and renumber the higher-indexed variants down** to keep contiguity. Example: delete totem/5.png → rename 6..16 → 5..15. Final count becomes 15 variants. Update `_manifest.json` after.

Target: **all retained variants are "ship it" quality**. Floor is 8 variants/building minimum; ceiling is 16.

## Building-by-building notes

**Important**: v3 full regen (batch_size 16, seeds 142-151) generates a DIFFERENT sample space than v2's batch_size 8 run. v2's success rates don't directly transfer. Treat these as rough initial estimates; reassess after first scan.

| Building | v2 baseline | Expected v3 (guess) | Time (16 variants) |
|---|---|---|---|
| campfire | 7/8 usable | ~12-14/16 | 80 min |
| hearth | 7/8 usable | ~12-14/16 | 90 min |
| cairn | 7/8 usable | ~12-14/16 | 90 min |
| gathering_marker | 8/8 usable | ~14-16/16 | 80 min (easiest) |
| totem | 8/8 usable | ~14-16/16 | 100 min |
| storage_pit | 6/8 usable | ~10-12/16 | 110 min |
| shelter | SKIP | SKIP | 0 min |
| stockpile | 3/8 usable | ~5-8/16 | 120 min |
| drying_rack | 2/8 usable (v3 rewrite) | ~10-12/16 if v3 works | 100 min |
| workbench | 0/8 usable (v3 rewrite) | ~8-10/16 if v3 works | 100 min |

**Total estimate**: 12-14 hours for all 9 buildings × 16 variants.

**If Phase 2 skipped** (auto-processing only): 0 min. Game gets raw auto-output and user evaluates in-game before deciding per-building what needs manual polish.

## Batch quality pass (after all buildings processed)

Final check before declaring the pipeline done — no ImageMagick required, pure Python:

```bash
source ~/ComfyUI/venv/bin/activate
python - <<'PY'
from pathlib import Path
from PIL import Image

expect = {
    "buildings/campfire": (32, 32),
    "buildings/stockpile": (64, 64),
    "buildings/totem": (32, 32),
    "buildings/cairn": (32, 32),
    "buildings/gathering_marker": (32, 32),
    "furniture/hearth": (32, 32),
    "furniture/storage_pit": (32, 32),
    "furniture/workbench": (64, 32),
    "furniture/drying_rack": (64, 32),
}

bad = []
for rel, size in expect.items():
    d = Path("assets/sprites") / rel
    if not d.exists():
        print(f"MISSING DIR {d}")
        continue
    pngs = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
    idxs = [int(p.stem) for p in pngs]
    if idxs and idxs != list(range(1, len(idxs) + 1)):
        print(f"GAP {d}: idxs={idxs}")
    for p in pngs:
        w, h = Image.open(p).size
        if (w, h) != size:
            bad.append(f"BAD {p}: got {w}x{h}, want {size[0]}x{size[1]}")
            print(bad[-1])
        else:
            print(f"OK  {p} ({w}x{h})")
print(f"\n{len(bad)} bad files" if bad else "\nAll OK")
PY
```

All must report OK and no GAP before committing.
