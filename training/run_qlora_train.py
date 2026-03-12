#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.lib.qlora_smoke import (  # noqa: F401
    BASELINE_MODEL_NAME,
    DEFAULT_TARGET_MODULES,
    RuntimeConfig,
    SmokeRunBlockedError,
    SmokeRunConfig,
    SmokeRunResult,
    main_baseline,
    parse_baseline_args,
    run_baseline,
    run_baseline_or_raise,
)


if __name__ == "__main__":
    raise SystemExit(main_baseline())
