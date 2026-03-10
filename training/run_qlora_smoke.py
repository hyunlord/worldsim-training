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
    NOTEBOOK_RUN_MODES,
    RuntimeConfig,
    SmokeRunBlockedError,
    SmokeRunConfig,
    SmokeRunResult,
    analyze_sample_generation,
    build_operational_judgment,
    build_trainer_kwargs,
    build_training_arguments_kwargs,
    coerce_smoke_config,
    count_parseable_json_samples,
    detect_runtime,
    get_environment_summary,
    get_true_qlora_preflight,
    load_message_rows,
    load_json_artifact,
    load_sample_generations,
    main,
    parse_args,
    pick_rows,
    preview_metrics,
    recommended_next_smoke_config,
    render_conversation,
    resolve_notebook_run_mode,
    run_smoke,
    run_smoke_or_raise,
    strip_json_fence,
    summarize_sample_generations,
)


if __name__ == "__main__":
    raise SystemExit(main())
