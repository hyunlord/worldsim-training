# Scripts Directory Rules

- Keep each script focused on a single pipeline stage.
- Read paths and thresholds from `config/generation.yaml` or CLI arguments instead of hardcoding them.
- Make file I/O explicit and predictable.
- Fail with clear error messages when required inputs are missing.
- Keep reusable helpers in local modules under `scripts/`.
