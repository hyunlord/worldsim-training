#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.lib.qlora_smoke import (  # noqa: F401
    DEFAULT_MODEL_NAME,
    DEFAULT_TARGET_MODULES,
    DEFAULT_TASKS,
    RuntimeConfig,
    SmokeRunBlockedError,
    SmokeRunConfig,
    SmokeRunResult,
    build_trainer_kwargs,
    build_training_arguments_kwargs,
    coerce_smoke_config,
    count_parseable_json_samples,
    detect_runtime,
    get_environment_summary,
    load_message_rows,
    load_sample_generations,
    main,
    parse_args,
    pick_rows,
    preview_metrics,
    render_conversation,
    run_smoke,
    run_smoke_or_raise,
)


if __name__ == "__main__":
    raise SystemExit(main())
