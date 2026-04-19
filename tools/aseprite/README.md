# Aseprite Pipeline Tools

Semi-automated sprite cleanup: ComfyUI 1024×1024 concept art → game-ready assets.

Two stages:
1. **Phase 1 (automated):** `auto_process.py` — downscale, bg removal, palette reduce, classify
2. **Phase 2 (manual):** `manual_cleanup_guide.md` — Aseprite pixel-level touch-up (optional)

Quick start:

```bash
# Activate Python env (ComfyUI venv has all deps)
source ~/ComfyUI/venv/bin/activate

# Phase 1: auto-process all 9 buildings (shelter auto-skipped)
python tools/aseprite/auto_process.py \
  --config tools/aseprite/configs/sprite_classification.yaml \
  --source-root assets/sprites/concepts_final \
  --staging-root _raw_cleanup_stage \
  --output-root assets/sprites \
  --force

# Phase 2: open Aseprite, follow manual_cleanup_guide.md
```

## Output layout

Variant-folder only (no representative file — game's variant loader reads the
folder directly):

```
assets/sprites/
├── buildings/{campfire,stockpile,totem,cairn,gathering_marker}/{1..16}.png
└── furniture/{hearth,storage_pit,workbench,drying_rack}/{1..16}.png
```

Variant PNGs are gitignored (regeneratable from `concepts_final/` + this tool).

## Flags

- `--dry-run` — print plan, no writes
- `--only <name1,name2>` — process a subset
- `--force` — overwrite existing variant files
- `--skip-classify` — run stages 1-3 only (staging), skip copy to game tree
