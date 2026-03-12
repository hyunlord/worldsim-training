from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_message_rows_requires_messages_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_jsonl(
        dataset_path,
        [
            {
                "task": "A",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"},
                    {"role": "assistant", "content": "{\"ok\":true}"},
                ],
            }
        ],
    )

    from training.run_qlora_smoke import load_message_rows

    rows = load_message_rows(dataset_path)
    assert len(rows) == 1
    assert rows[0]["task"] == "A"


def test_load_message_rows_rejects_invalid_messages_schema(tmp_path: Path) -> None:
    dataset_path = tmp_path / "train.jsonl"
    write_jsonl(dataset_path, [{"task": "A", "messages": [{"role": "user"}]}])

    from training.run_qlora_smoke import load_message_rows

    try:
        load_message_rows(dataset_path)
    except ValueError as exc:
        assert "invalid messages payload" in str(exc)
    else:
        raise AssertionError("Expected invalid messages payload to raise")


def test_render_conversation_prefers_chat_template() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            return f"templated:{len(messages)}:{add_generation_prompt}"

    from training.run_qlora_smoke import render_conversation

    rendered = render_conversation(
        FakeTokenizer(),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        add_generation_prompt=True,
    )

    assert rendered == "templated:2:True"


def test_render_conversation_has_plaintext_fallback() -> None:
    class PlainTokenizer:
        chat_template = None

    from training.run_qlora_smoke import render_conversation

    rendered = render_conversation(
        PlainTokenizer(),
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
        ],
        add_generation_prompt=False,
    )

    assert "<|system|>" in rendered
    assert "<|user|>" in rendered
    assert "sys" in rendered
    assert "usr" in rendered


def test_pick_rows_balances_tasks_when_truncating() -> None:
    rows = [
        {"task": "B", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}]},
        {"task": "B", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}]},
        {"task": "B", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}]},
        {"task": "G", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}]},
        {"task": "H", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "{}"}]},
    ]

    from training.run_qlora_smoke import pick_rows

    selected = pick_rows(rows, 3, seed=7)

    assert len(selected) == 3
    assert {row["task"] for row in selected} == {"B", "G", "H"}


def test_training_arguments_kwargs_matches_available_signature() -> None:
    from training.run_qlora_smoke import RuntimeConfig, build_training_arguments_kwargs

    available = {"output_dir", "max_steps", "logging_steps", "eval_strategy", "save_strategy", "report_to", "seed", "remove_unused_columns", "dataloader_pin_memory", "use_cpu"}
    kwargs = build_training_arguments_kwargs(
        RuntimeConfig(device="mps", use_qlora=False, fallback_reason="mps fallback", torch_dtype="float32"),
        available_parameters=available,
        output_dir="out",
        max_steps=3,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        seed=42,
        logging_steps=1,
        eval_steps=0,
        save_steps=0,
        save_total_limit=1,
    )

    assert kwargs["logging_steps"] == 1
    assert kwargs["eval_strategy"] == "no"
    assert kwargs["save_strategy"] == "no"
    assert "use_mps_device" not in kwargs
    assert "no_cuda" not in kwargs


def test_build_trainer_kwargs_prefers_processing_class_when_available() -> None:
    from training.run_qlora_smoke import build_trainer_kwargs

    kwargs = build_trainer_kwargs(
        available_parameters={"model", "args", "train_dataset", "eval_dataset", "data_collator", "processing_class"},
        model="model",
        args="args",
        train_dataset="train",
        eval_dataset="eval",
        data_collator="collator",
        tokenizer="tokenizer",
    )

    assert kwargs["processing_class"] == "tokenizer"
    assert "tokenizer" not in kwargs


def test_programmatic_run_smoke_accepts_mapping() -> None:
    from training.lib.qlora_smoke import SmokeRunConfig, coerce_smoke_config

    config = coerce_smoke_config(
        {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "train_file": "data/training/worldsim-v31-mix-v1/train_converted.jsonl",
            "dev_file": "data/training/worldsim-v31-mix-v1/dev_converted.jsonl",
            "output_dir": "outputs/test-smoke",
            "max_steps": 1,
        }
    )

    assert isinstance(config, SmokeRunConfig)
    assert config.output_dir == Path("outputs/test-smoke")
    assert config.max_steps == 1


def test_programmatic_run_baseline_uses_baseline_defaults() -> None:
    from training.lib.qlora_smoke import BASELINE_MODEL_NAME, SmokeRunConfig, coerce_smoke_config

    config = coerce_smoke_config({"run_mode": "baseline"}, default_run_mode="baseline")

    assert isinstance(config, SmokeRunConfig)
    assert config.run_mode == "baseline"
    assert config.model_name == BASELINE_MODEL_NAME
    assert config.max_steps == 200
    assert config.max_train_samples == 0
    assert config.max_eval_samples == 0
    assert config.gradient_accumulation_steps == 8
    assert config.learning_rate == 1e-4
    assert config.logging_steps == 5
    assert config.eval_steps == 25
    assert config.save_steps == 25
    assert config.save_total_limit == 2


def test_programmatic_run_smoke_normalizes_dataclass_paths() -> None:
    from training.lib.qlora_smoke import SmokeRunConfig, coerce_smoke_config

    config = coerce_smoke_config(
        SmokeRunConfig(
            output_dir="outputs/dataclass-smoke",  # type: ignore[arg-type]
            target_modules=["q_proj", "v_proj"],  # type: ignore[arg-type]
        )
    )

    assert config.output_dir == Path("outputs/dataclass-smoke")
    assert config.target_modules == ("q_proj", "v_proj")


def test_resolve_output_dir_uses_run_mode_roots(tmp_path: Path, monkeypatch) -> None:
    from training.lib import qlora_smoke

    monkeypatch.setitem(qlora_smoke.RUN_MODE_DEFAULTS, "smoke", {**qlora_smoke.RUN_MODE_DEFAULTS["smoke"], "output_root": tmp_path / "smoke"})
    monkeypatch.setitem(qlora_smoke.RUN_MODE_DEFAULTS, "baseline", {**qlora_smoke.RUN_MODE_DEFAULTS["baseline"], "output_root": tmp_path / "baseline"})

    smoke_output = qlora_smoke._resolve_output_dir(None, "smoke")
    baseline_output = qlora_smoke._resolve_output_dir(None, "baseline")

    assert smoke_output.parent == tmp_path / "smoke"
    assert baseline_output.parent == tmp_path / "baseline"


def test_build_training_arguments_kwargs_supports_baseline_step_strategies() -> None:
    from training.run_qlora_smoke import RuntimeConfig, build_training_arguments_kwargs

    available = {
        "output_dir",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "logging_steps",
        "eval_strategy",
        "eval_steps",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "report_to",
        "seed",
        "remove_unused_columns",
        "dataloader_pin_memory",
        "use_cpu",
    }
    kwargs = build_training_arguments_kwargs(
        RuntimeConfig(device="cpu", use_qlora=False, fallback_reason=None, torch_dtype="float32"),
        available_parameters=available,
        output_dir="out",
        max_steps=200,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        seed=42,
        logging_steps=5,
        eval_steps=25,
        save_steps=25,
        save_total_limit=2,
    )

    assert kwargs["eval_strategy"] == "steps"
    assert kwargs["eval_steps"] == 25
    assert kwargs["save_strategy"] == "steps"
    assert kwargs["save_steps"] == 25
    assert kwargs["save_total_limit"] == 2
    assert kwargs["logging_steps"] == 5
    assert kwargs["use_cpu"] is True


def test_resolve_notebook_run_mode_returns_expected_defaults() -> None:
    from training.lib.qlora_smoke import resolve_notebook_run_mode

    resolved = resolve_notebook_run_mode("longer_smoke", run_id="manual-run")

    assert resolved == {
        "run_mode": "longer_smoke",
        "run_id": "manual-run",
        "max_steps": 25,
        "max_train_samples": 256,
        "max_eval_samples": 64,
    }


def test_resolve_baseline_notebook_config_uses_real_baseline_defaults() -> None:
    from training.lib.qlora_smoke import BASELINE_MODEL_NAME, resolve_baseline_notebook_config

    config = resolve_baseline_notebook_config("run-123")

    assert config["run_mode"] == "baseline"
    assert config["run_id"] == "run-123"
    assert config["model_name"] == BASELINE_MODEL_NAME
    assert config["dry_run"] is False
    assert config["require_qlora"] is True
    assert config["max_train_samples"] == 0
    assert config["max_eval_samples"] == 0
    assert config["max_steps"] == 200
    assert config["gradient_accumulation_steps"] == 8
    assert config["learning_rate"] == 1e-4
    assert config["logging_steps"] == 5
    assert config["eval_steps"] == 25
    assert config["save_steps"] == 25
    assert config["save_total_limit"] == 2
    assert config["output_dir"] == Path("outputs/baseline/worldsim-v31-mix-v1/run-123")


def test_resolve_baseline_notebook_config_accepts_explicit_output_override() -> None:
    from training.lib.qlora_smoke import resolve_baseline_notebook_config

    config = resolve_baseline_notebook_config("run-123", output_dir_override="outputs/custom/run-123")

    assert config["output_dir"] == Path("outputs/custom/run-123")


def test_parse_baseline_args_uses_baseline_defaults() -> None:
    from training.lib.qlora_smoke import BASELINE_MODEL_NAME, parse_baseline_args

    args = parse_baseline_args([])

    assert args.run_mode == "baseline"
    assert args.model_name == BASELINE_MODEL_NAME
    assert args.max_steps == 200
    assert args.max_train_samples == 0
    assert args.max_eval_samples == 0
    assert args.gradient_accumulation_steps == 8
    assert args.logging_steps == 5
    assert args.eval_steps == 25
    assert args.save_steps == 25
    assert args.save_total_limit == 2


def test_run_qlora_train_uses_shared_baseline_entrypoint() -> None:
    entrypoint_path = Path("training/run_qlora_train.py")
    source = entrypoint_path.read_text(encoding="utf-8")

    assert "from training.lib.qlora_smoke import" in source
    assert "parse_baseline_args" in source
    assert "run_baseline" in source or "main_baseline" in source


def test_build_operational_judgment_distinguishes_fenced_only_case() -> None:
    from training.lib.qlora_smoke import build_operational_judgment

    judgment = build_operational_judgment(
        {"used_true_qlora": True},
        {
            "raw_parseable_json": 0,
            "fence_stripped_parseable_json": 7,
            "recoverable_fenced_json": 7,
            "malformed_json": 0,
            "enum_drift_total": 0,
        },
        output_dir="outputs/smoke_cuda_notebook/worldsim-v31-mix-v1/example",
    )

    assert judgment["true_qlora_passed"] is True
    assert judgment["operational_issue"] == "markdown_fencing_only"
    assert judgment["raw_json_parse_failed"] is True
    assert "markdown fences" in judgment["recommended_next_action"]


def test_register_baseline_run_skips_blocked_runs(tmp_path: Path) -> None:
    from training.lib.qlora_smoke import register_baseline_run

    registry_path = tmp_path / "model_registry.json"
    registry_path.write_text(json.dumps({"runs": [{"run_id": "existing", "adapter_dir": "keep"}]}), encoding="utf-8")

    entry = register_baseline_run(
        registry_path,
        config={"run_id": "blocked-run", "dataset": "worldsim-v31-mix-v1", "model_name": "Qwen/Base"},
        result={
            "status": "blocked",
            "output_dir": "outputs/baseline/worldsim-v31-mix-v1/blocked-run",
            "adapter_dir": None,
            "used_true_qlora": False,
            "train_loss": None,
            "eval_loss": None,
        },
        analysis_report={"overall_status": "structure_failure", "semantic_low_quality_count": 0},
        metrics={"retry_rate": 0.0},
    )

    assert entry is None
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["runs"] == [{"run_id": "existing", "adapter_dir": "keep"}]


def test_register_baseline_run_records_successful_metadata(tmp_path: Path) -> None:
    from training.lib.qlora_smoke import register_baseline_run

    registry_path = tmp_path / "model_registry.json"
    entry = register_baseline_run(
        registry_path,
        config={"run_id": "run-001", "dataset": "worldsim-v31-mix-v1", "model_name": "Qwen/Base"},
        result={
            "status": "ok",
            "output_dir": "outputs/baseline/worldsim-v31-mix-v1/run-001",
            "adapter_dir": "outputs/baseline/worldsim-v31-mix-v1/run-001/adapter",
            "used_true_qlora": True,
            "train_loss": 1.25,
            "eval_loss": 0.95,
        },
        analysis_report={"overall_status": "semantic_quality_issue", "semantic_low_quality_count": 2},
        metrics={"retry_rate": 0.125},
        created_at="2026-03-12T00:00:00Z",
    )

    assert entry is not None
    assert entry["run_id"] == "run-001"
    assert entry["created_at"] == "2026-03-12T00:00:00Z"
    assert entry["used_true_qlora"] is True
    assert entry["analyzer_overall_status"] == "semantic_quality_issue"
    assert entry["metrics"]["semantic_low_quality"] == 2
    assert entry["metrics"]["retry_rate"] == 0.125

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["runs"] == [entry]


def test_select_best_adapter_run_prefers_semantic_quality_then_eval_loss() -> None:
    from training.lib.qlora_smoke import select_best_adapter_run

    best_run = select_best_adapter_run(
        {
            "runs": [
                {
                    "run_id": "run-b",
                    "status": "ok",
                    "adapter_dir": "/tmp/run-b/adapter",
                    "metrics": {"semantic_low_quality": 1, "eval_loss": 0.8},
                },
                {
                    "run_id": "run-a",
                    "status": "ok",
                    "adapter_dir": "/tmp/run-a/adapter",
                    "metrics": {"semantic_low_quality": 0, "eval_loss": 1.2},
                },
                {
                    "run_id": "run-c",
                    "status": "blocked",
                    "adapter_dir": "/tmp/run-c/adapter",
                    "metrics": {"semantic_low_quality": 0, "eval_loss": 0.1},
                },
            ]
        }
    )

    assert best_run is not None
    assert best_run["run_id"] == "run-a"


def test_update_best_adapter_pointer_does_not_overwrite_when_no_candidate(tmp_path: Path) -> None:
    from training.lib.qlora_smoke import update_best_adapter_pointer

    pointer_path = tmp_path / "best_adapter.txt"
    pointer_path.write_text("/existing/adapter", encoding="utf-8")

    selected = update_best_adapter_pointer(pointer_path, None)

    assert selected is None
    assert pointer_path.read_text(encoding="utf-8") == "/existing/adapter"


def test_build_baseline_candidate_judgment_marks_semantic_issue_but_candidate(tmp_path: Path) -> None:
    from training.lib.qlora_smoke import build_baseline_candidate_judgment

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    judgment = build_baseline_candidate_judgment(
        {
            "status": "ok",
            "used_true_qlora": True,
            "runtime": {"device": "cuda"},
            "finite_losses": True,
            "output_dir": str(tmp_path / "run-001"),
            "adapter_dir": str(adapter_dir),
            "train_loss": 1.2,
            "eval_loss": 0.9,
        },
        {
            "overall_status": "semantic_quality_issue",
            "malformed_json_count": 0,
            "fenced_json_count": 0,
            "truncation_count": 0,
            "enum_drift_count": 0,
            "semantic_valid_count": 6,
            "semantic_low_quality_count": 1,
            "semantic_drift_count": 0,
            "language_drift_count": 0,
        },
    )

    assert judgment["used_true_qlora"] is True
    assert judgment["training_completed_successfully"] is True
    assert judgment["adapter_exists"] is True
    assert judgment["losses_finite"] is True
    assert judgment["structure_stable"] is True
    assert judgment["semantic_quality_is_primary_remaining_issue"] is True
    assert judgment["is_baseline_candidate"] is True
    assert judgment["verdict"] == "PASS_STRUCTURAL_BUT_SEMANTIC_WEAK"


def test_build_baseline_candidate_judgment_rejects_missing_adapter() -> None:
    from training.lib.qlora_smoke import build_baseline_candidate_judgment

    judgment = build_baseline_candidate_judgment(
        {
            "status": "ok",
            "used_true_qlora": True,
            "runtime": {"device": "cuda"},
            "finite_losses": True,
            "output_dir": "outputs/baseline/worldsim-v31-mix-v1/run-002",
            "adapter_dir": None,
            "train_loss": 1.2,
            "eval_loss": 0.9,
        },
        {
            "overall_status": "structurally_usable",
            "malformed_json_count": 0,
            "fenced_json_count": 0,
            "truncation_count": 0,
            "enum_drift_count": 0,
            "semantic_low_quality_count": 0,
            "semantic_drift_count": 0,
            "language_drift_count": 0,
        },
    )

    assert judgment["adapter_exists"] is False
    assert judgment["is_baseline_candidate"] is False
    assert judgment["verdict"] == "FAIL_ARTIFACT_INVALID"


def test_true_qlora_preflight_surfaces_blocker(monkeypatch) -> None:
    from training.lib import qlora_smoke

    monkeypatch.setattr(qlora_smoke, "get_environment_summary", lambda: {"torch": {"cuda_available": False}})

    def fail_runtime(*, prefer_qlora: bool, require_qlora: bool):
        assert prefer_qlora is True
        assert require_qlora is True
        raise RuntimeError("QLoRA unavailable")

    monkeypatch.setattr(qlora_smoke, "detect_runtime", fail_runtime)

    report = qlora_smoke.get_true_qlora_preflight()

    assert report["ok"] is False
    assert report["runtime"] is None
    assert "QLoRA unavailable" in report["blocker_reason"]


def test_get_environment_summary_includes_trl_key() -> None:
    from training.lib.qlora_smoke import get_environment_summary

    summary = get_environment_summary()

    assert "trl" in summary


def test_sample_summary_counts_fenced_json_and_enum_drift() -> None:
    from training.lib.qlora_smoke import analyze_sample_generation, strip_json_fence, summarize_sample_generations

    fenced = "```json\n{\"action_id\": 0}\n```"
    assert strip_json_fence(fenced) == "{\"action_id\": 0}"
    assert strip_json_fence("```json\n{\"action_id\": 0") == "{\"action_id\": 0"

    analyzed = analyze_sample_generation(
        {
            "task": "E",
            "generated_assistant": fenced,
            "json_parse_error": "JSONDecodeError",
        }
    )

    assert analyzed["classification"] == "fenced_recoverable"
    assert analyzed["failure_category"] == "fenced_json"
    assert analyzed["raw_parseable_json"] is False
    assert analyzed["fence_stripped_parseable_json"] is True
    assert analyzed["malformed_json"] is False

    samples = [
        {
            "task": "E",
            "generated_assistant": "```json\n{\"action_id\": 0, \"confidence\": 0.9, \"hint_ko\": \"곧 달아났다\", \"hint_en\": \"fled\", \"personality_reasoning\": \"high_HA\"}\n```",
            "json_parse_error": "JSONDecodeError",
        },
        {
            "task": "F",
            "generated_assistant": "{\"emotion\": \"panic\", \"intensity\": 0.8, \"cause_ko\": \"겁에 질렸다\", \"cause_en\": \"afraid\", \"previous_emotion\": \"trust\", \"transition_type\": \"sudden\", \"temperament_amplifier\": \"high_HA\"}",
            "json_parse_error": None,
        },
        {
            "task": "C",
            "generated_assistant": "{\"speech_ko\": \"나서거라\", \"speech_en\": \"step forward\", \"register\": \"hao\", \"emotion_expressed\": \"anger\", \"speaker_role\": \"chief\", \"temperament_tone\": \"direct\"}",
            "json_parse_error": None,
        },
    ]

    summary = summarize_sample_generations(samples)

    assert summary["total"] == 3
    assert summary["raw_parseable_json"] == 2
    assert summary["fenced_json"] == 1
    assert summary["fence_stripped_parseable_json"] == 3
    assert summary["recoverable_fenced_json"] == 1
    assert summary["malformed_json"] == 0
    assert summary["enum_drift_total"] == 1
    assert summary["enum_drift_fields"]["emotion"] == 1
    assert summary["failure_categories"]["fenced_json"] == 1
    assert len(summary["recoverable_examples"]) == 1


def test_sample_summary_detects_truncation_for_incomplete_fenced_json() -> None:
    from training.lib.qlora_smoke import analyze_sample_generation

    analyzed = analyze_sample_generation(
        {
            "task": "H",
            "generated_assistant": "```json\n{\"name\": \"World\", \"description_en\": \"desc\"",
            "json_parse_error": "JSONDecodeError",
        }
    )

    assert analyzed["fenced_json"] is True
    assert analyzed["fence_stripped_parseable_json"] is False
    assert analyzed["malformed_json"] is True
    assert analyzed["failure_category"] == "truncation"


def test_json_object_complete_stops_after_first_balanced_object() -> None:
    from training.lib.qlora_smoke import _json_object_complete

    assert _json_object_complete("{\"task\":\"A\"}") is True
    assert _json_object_complete("{\"task\":\"A\"}{\"task\":\"B\"}") is True
    assert _json_object_complete("{\"task\":\"A\"") is False
    assert _json_object_complete("{\"task\":\"A\", \"text\":\"brace } inside string\"}") is True


def test_trim_trivial_json_tail_removes_only_trailing_comma() -> None:
    from training.lib.qlora_smoke import _trim_trivial_json_tail

    trimmed, reason = _trim_trivial_json_tail('{"task":"A"},')
    assert trimmed == '{"task":"A"}'
    assert reason == "trim_trailing_comma"

    unchanged, reason = _trim_trivial_json_tail('{"task":"A"} trailing words')
    assert unchanged == '{"task":"A"} trailing words'
    assert reason is None


def test_trim_follow_on_json_object_keeps_only_first_object() -> None:
    from training.lib.qlora_smoke import _trim_follow_on_json_object

    trimmed, reason = _trim_follow_on_json_object('{"task":"A"},{"task":"B"}')
    assert trimmed == '{"task":"A"}'
    assert reason == "trim_follow_on_json_object"

    spaced_trimmed, reason = _trim_follow_on_json_object('{"task":"A"}   {"task":"B"}')
    assert spaced_trimmed == '{"task":"A"}'
    assert reason == "trim_follow_on_json_object"

    unchanged, reason = _trim_follow_on_json_object('{"task":"A"} trailing words')
    assert unchanged == '{"task":"A"} trailing words'
    assert reason is None


def test_build_sample_prompt_messages_appends_generic_generation_rules() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "G",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[과제]"},
                {"role": "assistant", "content": "{\"action_tendency\":\"wait\"}"},
            ],
        }
    )

    user_content = prompt_messages[-1]["content"]
    assert user_content.startswith("[과제]")
    assert "[SYSTEM ROLE]" in user_content
    assert "[OUTPUT RULES]" in user_content
    assert "Output must be a single JSON object." in user_content
    assert "The first character must be { and the last character must be }." in user_content
    assert "Do not copy instructions, schema descriptions, examples, enum lists, or placeholder text into the JSON values." in user_content
    assert "Use double quotes for every JSON key and every string value." in user_content


def test_build_sample_prompt_messages_strips_leaky_format_sections() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "B",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[과제]\n"
                        "반응을 써라.\n\n"
                        "[어투]\n"
                        "해라체로 써라. 문장을 -다, -는다 로 끝내라.\n\n"
                        "[출력 형식]\n"
                        '{"text_ko":"순우리말 2문장","text_en":"English 2 sentences"}\n\n'
                        "[유효값 다시 보기]\n"
                        "emotion_expressed must be exactly one of: joy, sadness\n\n"
                        "[규칙]\n"
                        "JSON만 출력하라\n"
                    ),
                },
                {"role": "assistant", "content": '{"text_ko":"..."}'},
            ],
        }
    )

    user_content = prompt_messages[-1]["content"]
    assert "[출력 형식]" not in user_content
    assert "[유효값 다시 보기]" not in user_content
    assert "[어투]" not in user_content
    assert "English 2 sentences" not in user_content
    assert "[규칙]" not in user_content
    assert "JSON만 출력하라" not in user_content
    assert "[ENUM CONSTRAINTS]" in user_content
    assert "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation" in user_content


def test_build_sample_prompt_messages_strips_rules_with_copy_prone_placeholders() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "E",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[과제]\n"
                        "행동을 골라라.\n\n"
                        "[규칙]\n"
                        "- hint_ko는 왜 이 행동을 골랐는지 순우리말 10~30글자로 써라\n"
                        "- hint_en은 English 1 sentence using the same phrase as the original 10~30 characters\n"
                        "- personality_reasoning은 위 선택지 중 하나로만 적어라\n"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
    )

    user_content = prompt_messages[-1]["content"]
    assert "English 1 sentence using the same phrase as the original 10~30 characters" not in user_content
    assert "순우리말 10~30글자" not in user_content
    assert "[규칙]" not in user_content


def test_build_sample_prompt_messages_strips_selection_lists_but_keeps_action_options() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "F",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[과제]\n"
                        "감정을 답하라.\n\n"
                        "[감정 선택지]\n"
                        "joy, sadness\n\n"
                        "[이전 감정 고르기]\n"
                        "기쁨=joy\n\n"
                        "[전이 방식 선택지]\n"
                        "gradual, sudden\n"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
    )
    user_content = prompt_messages[-1]["content"]
    assert "[감정 선택지]" not in user_content
    assert "[이전 감정 고르기]" not in user_content
    assert "[전이 방식 선택지]" not in user_content

    e_prompt = _build_sample_prompt_messages(
        {
            "task": "E",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[선택지]\n0:도망 1:숨기 2:맞서기"},
                {"role": "assistant", "content": "{}"},
            ],
        }
    )[-1]["content"]
    assert "[선택지]" in e_prompt
    assert "0:도망 1:숨기 2:맞서기" in e_prompt


def test_build_sample_prompt_messages_adds_g_specific_generation_rules() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "G",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[TASK] G"},
                {"role": "assistant", "content": "{\"action_tendency\":\"wait\"}"},
            ],
        }
    )

    user_content = prompt_messages[-1]["content"]
    assert "[SYSTEM ROLE]" in user_content
    assert "[OUTPUT RULES]" in user_content
    assert "[TASK CONTEXT]" in user_content
    assert "[ALLOWED JSON KEYS]" in user_content
    assert "[ENUM CONSTRAINTS]" in user_content
    assert "[TASK RULES]" in user_content
    assert "[EXAMPLE OUTPUT]" in user_content
    assert "interpretation_ko, interpretation_en, action_tendency, confidence, register, misinterpretation_type, temperament_bias" in user_content
    assert "action_tendency must be exactly one of: mobilize, defend, wait, retreat, celebrate, mourn" in user_content
    assert "misinterpretation_type must be exactly one of: overconfident_literal, cautious_reversal, optimistic_expansion, passive_deferral, symbolic_abstraction" in user_content
    assert "interpretation_ko must be exactly one Korean sentence that interprets the oracle meaning only." in user_content
    assert '{"interpretation_ko":"이 말은 지금은 공격보다 방어를 준비하라고 판단한다."' in user_content


def test_semantic_guard_task_g_blocks_personality_meta_reasoning() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_messages = _build_sample_prompt_messages(
        {
            "task": "G",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[과제] 신탁 풀이\n"
                        "[기질 이름]\n우울질\n"
                        "[인물 성격]\n겁많음, 꼼꼼함, 조용함\n"
                        "[ORACLE]\n북쪽 산 너머에 풍요가 있다"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
    )

    user_content = prompt_messages[-1]["content"]
    assert "Do not describe personality, situation summary, reasoning steps, or self-introduction." in user_content
    assert "Do not copy enum lists or placeholder text into action_tendency or misinterpretation_type." in user_content
    assert "[기질 이름]" not in user_content
    assert "[인물 성격]" not in user_content
    assert "[ORACLE]" in user_content


def test_build_sample_prompt_messages_strips_personality_sections_for_g_only() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_g = _build_sample_prompt_messages(
        {
            "task": "G",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[기질 이름]\n우울질\n\n"
                        "[기질 키워드]\n신중함, 불안함\n\n"
                        "[인물 성격]\n겁많음, 조용함\n\n"
                        "[ORACLE]\n북쪽 산 너머에 풍요가 있다"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
    )[-1]["content"]

    prompt_b = _build_sample_prompt_messages(
        {
            "task": "B",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": (
                        "[기질 이름]\n우울질\n\n"
                        "[기질 키워드]\n신중함, 불안함\n\n"
                        "[인물 성격]\n겁많음, 조용함\n\n"
                        "[상황]\n날랜 짐승이 나타났다"
                    ),
                },
                {"role": "assistant", "content": "{}"},
            ],
        }
    )[-1]["content"]

    assert "[기질 이름]" not in prompt_g
    assert "[기질 키워드]" not in prompt_g
    assert "[인물 성격]" not in prompt_g
    assert "[ORACLE]" in prompt_g
    assert "[기질 이름]" in prompt_b
    assert "[기질 키워드]" in prompt_b
    assert "[인물 성격]" in prompt_b


def test_build_sample_prompt_messages_adds_a_and_c_specific_generation_rules() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompt_a = _build_sample_prompt_messages(
        {
            "task": "A",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[TASK] A"},
                {"role": "assistant", "content": "{\"text_ko\":\"...\"}"},
            ],
        }
    )
    assert "[ALLOWED JSON KEYS]" in prompt_a[-1]["content"]
    assert "text_ko, text_en, register, dominant_trait, temperament_expressed" in prompt_a[-1]["content"]
    assert "Human:" not in prompt_a[-1]["content"]
    assert "Assistant:" not in prompt_a[-1]["content"]
    assert "register must be exactly one of: haera, hao, hae" in prompt_a[-1]["content"]
    assert "dominant_trait must be exactly one of: novelty_seeking, harm_avoidance, reward_dependence, persistence" in prompt_a[-1]["content"]
    assert "text_ko and text_en must be concrete persona descriptions, not labels or placeholders." in prompt_a[-1]["content"]
    assert '{"text_ko":"그는 앞장서되 위험을 먼저 헤아린다."' in prompt_a[-1]["content"]

    prompt_c = _build_sample_prompt_messages(
        {
            "task": "C",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "[TASK] C"},
                {"role": "assistant", "content": "{\"speech_ko\":\"...\"}"},
            ],
        }
    )
    assert "speech_ko, speech_en, register, emotion_expressed, speaker_role, temperament_tone" in prompt_c[-1]["content"]
    assert "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation" in prompt_c[-1]["content"]
    assert "speaker_role must be exactly one of: elder, hunter, shaman, warrior, healer, gatherer, craftsman, chief, scout, observer" in prompt_c[-1]["content"]
    assert "speech_ko and speech_en must be direct utterances only, not explanations or copied instructions." in prompt_c[-1]["content"]
    assert '{"speech_ko":"지금은 서두르지 말고 불빛 가까이 모여라."' in prompt_c[-1]["content"]


def test_build_sample_prompt_messages_adds_b_e_f_g_h_specific_generation_rules() -> None:
    from training.lib.qlora_smoke import _build_sample_prompt_messages

    prompts = {}
    for task in ("B", "E", "F", "G", "H"):
        prompts[task] = _build_sample_prompt_messages(
            {
                "task": task,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"[TASK] {task}"},
                    {"role": "assistant", "content": "{}"},
                ],
            }
        )[-1]["content"]

    for task in ("B", "E", "F", "G", "H"):
        assert "[OUTPUT RULES]" in prompts[task]
        assert "[ALLOWED JSON KEYS]" in prompts[task]
        assert "[TASK RULES]" in prompts[task]
        assert "[EXAMPLE OUTPUT]" in prompts[task]

    assert "text_ko and text_en must be real emotional reactions, not length instructions or schema text." in prompts["B"]
    assert "emotion_expressed must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation" in prompts["B"]
    assert "action_id, confidence, hint_ko, hint_en, personality_reasoning, temperament_factor" in prompts["E"]
    assert "personality_reasoning must be exactly one of: high_NS, high_HA, high_RD, high_P" in prompts["E"]
    assert "Only action_id and confidence are numeric; every other field is a string." in prompts["E"]
    assert "emotion must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation" in prompts["F"]
    assert "previous_emotion must be exactly one of: joy, sadness, fear, anger, trust, disgust, surprise, anticipation" in prompts["F"]
    assert "interpretation_ko and interpretation_en, action_tendency, confidence, register, misinterpretation_type, temperament_bias" not in prompts["G"]
    assert "interpretation_ko, interpretation_en, action_tendency, confidence, register, misinterpretation_type, temperament_bias" in prompts["G"]
    assert "Do not describe personality, situation summary, reasoning steps, or self-introduction." in prompts["G"]
    assert "name, description_en, resource_modifiers, special_zones, special_resources, agent_modifiers" in prompts["H"]


def test_sample_generation_max_new_tokens_g_uses_larger_budget() -> None:
    from training.lib.qlora_smoke import _sample_generation_max_new_tokens

    assert _sample_generation_max_new_tokens("G") == 384
    assert _sample_generation_max_new_tokens("H") == 384
    assert _sample_generation_max_new_tokens("F") == 384


def test_sample_generation_assistant_prefix_is_task_specific() -> None:
    from training.lib.qlora_smoke import _sample_generation_assistant_prefix

    assert _sample_generation_assistant_prefix("A") == '{"text_ko": "'
    assert _sample_generation_assistant_prefix("B") == '{"text_ko": "'
    assert _sample_generation_assistant_prefix("C") == '{"speech_ko": "'
    assert _sample_generation_assistant_prefix("E") == '{"action_id": '
    assert _sample_generation_assistant_prefix("F") == '{"emotion": "'
    assert _sample_generation_assistant_prefix("G") == '{"interpretation_ko": "이 말은 '


def test_normalize_known_enum_values_only_fixes_case_style() -> None:
    from training.lib.qlora_smoke import _normalize_known_enum_values

    normalized, details = _normalize_known_enum_values(
        "G",
        {
            "action_tendency": "Defend",
            "misinterpretation_type": "Overconfident Literal",
            "register": "HAO",
            "temperament_bias": "Melancholic Snake Case Phrase",
        },
    )

    assert normalized["action_tendency"] == "defend"
    assert normalized["misinterpretation_type"] == "overconfident_literal"
    assert normalized["register"] == "hao"
    assert normalized["temperament_bias"] == "Melancholic Snake Case Phrase"
    assert details == [
        {"field": "register", "from": "HAO", "to": "hao"},
        {"field": "action_tendency", "from": "Defend", "to": "defend"},
        {
            "field": "misinterpretation_type",
            "from": "Overconfident Literal",
            "to": "overconfident_literal",
        },
    ]


def test_enum_drift_issues_handles_non_string_values() -> None:
    from training.lib.qlora_smoke import _enum_drift_issues

    issues = _enum_drift_issues(
        "C",
        {
            "emotion_expressed": ["joy", "sadness"],
            "register": "haera",
        },
    )

    assert issues == [("emotion_expressed", '["joy", "sadness"]')]


def test_validate_g_semantics_classifies_language_and_semantic_drift() -> None:
    from training.lib.qlora_smoke import validate_g_semantics

    language_drift = validate_g_semantics(
        {
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "This oracle means we should advance with certainty.",
                    "misinterpretation_type": "overconfident_literal",
                },
                ensure_ascii=False,
            )
        }
    )
    assert language_drift["semantic_status"] == "LANGUAGE_DRIFT"

    semantic_drift = validate_g_semantics(
        {
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 이 말을 해석하고 곧 나서야 한다고 생각하오",
                    "misinterpretation_type": "overconfident_literal",
                },
                ensure_ascii=False,
            )
        }
    )
    assert semantic_drift["semantic_status"] == "SEMANTIC_DRIFT"


def test_validate_g_semantics_marks_valid_and_low_quality() -> None:
    from training.lib.qlora_smoke import validate_g_semantics

    valid = validate_g_semantics(
        {
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 이 말을 해석하며 과신으로 곧 나서야 한다고 생각하오",
                    "misinterpretation_type": "overconfident_literal",
                },
                ensure_ascii=False,
            )
        }
    )
    assert valid["semantic_status"] == "VALID"

    low_quality = validate_g_semantics(
        {
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 곧 나서야 하오",
                    "misinterpretation_type": "cautious_reversal",
                },
                ensure_ascii=False,
            )
        }
    )
    assert low_quality["semantic_status"] == "LOW_QUALITY"


def test_summarize_sample_generations_counts_g_semantic_statuses() -> None:
    from training.lib.qlora_smoke import summarize_sample_generations

    samples = [
        {
            "task": "G",
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 이 말을 해석하며 과신으로 곧 나서야 한다고 생각하오",
                    "misinterpretation_type": "overconfident_literal",
                    "action_tendency": "wait",
                    "register": "hao",
                },
                ensure_ascii=False,
            ),
            "json_parse_error": None,
        },
        {
            "task": "G",
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "This oracle means we should defend the camp now.",
                    "misinterpretation_type": "overconfident_literal",
                    "action_tendency": "wait",
                    "register": "hao",
                },
                ensure_ascii=False,
            ),
            "json_parse_error": None,
        },
        {
            "task": "G",
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 이 말을 해석하고 곧 나서야 한다고 생각하오",
                    "misinterpretation_type": "overconfident_literal",
                    "action_tendency": "wait",
                    "register": "hao",
                },
                ensure_ascii=False,
            ),
            "json_parse_error": None,
        },
        {
            "task": "G",
            "generated_assistant": json.dumps(
                {
                    "interpretation_ko": "나는 곧 나서야 하오",
                    "misinterpretation_type": "cautious_reversal",
                    "action_tendency": "wait",
                    "register": "hao",
                },
                ensure_ascii=False,
            ),
            "json_parse_error": None,
        },
    ]

    summary = summarize_sample_generations(samples)

    assert summary["semantic_valid"] == 1
    assert summary["language_drift"] == 1
    assert summary["semantic_drift"] == 1
    assert summary["semantic_low_quality"] == 1


def test_notebook_uses_shared_training_module() -> None:
    notebook_path = Path("notebooks/dgx_spark_qlora_smoke.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert "from training.lib.qlora_smoke import" in source
    assert "get_true_qlora_preflight" in source
    assert "load_json_artifact" in source
    assert "summarize_sample_generations" in source
    assert "resolve_notebook_run_mode" in source
    assert "build_operational_judgment" in source
    assert "recoverable_fenced_json" in source
    assert "failure_categories" in source
    assert "semantic_valid" in source
    assert "semantic_low_quality" in source
    assert "semantic_drift" in source
    assert "language_drift" in source
    assert "recommended_next_action" in source
    assert "RUN_MODE" in source
    assert "longer_smoke" in source


def test_baseline_notebook_uses_shared_training_module() -> None:
    notebook_path = Path("notebooks/dgx_spark_qlora_train.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert "get_environment_summary" in source
    assert "get_true_qlora_preflight" in source
    assert "resolve_baseline_notebook_config" in source
    assert "run_baseline_or_raise" in source
    assert "load_json_artifact" in source
    assert "load_sample_generations" in source
    assert "register_baseline_run" in source
    assert "select_best_adapter_run" in source
    assert "update_best_adapter_pointer" in source
    assert "build_baseline_candidate_judgment" in source
    assert "generate_report" in source
    assert "recommend_next_action" in source
    assert "OUTPUT_DIR_OVERRIDE" in source
    assert "CONFIG['output_dir'] =" not in source
    assert "trl" in source
    assert "RUN_MODE" not in source
    assert "'dry_run': True" not in source
    assert "'require_qlora': True" in source or '"require_qlora": True' in source
    assert "PASS_BASELINE_CANDIDATE" in source
    assert "FAIL_BLOCKED_RUNTIME" in source
    assert "GUARDRAIL IMPACT SUMMARY" in source
    assert "structured_metrics" in source


def test_generate_structured_retries_on_malformed_json() -> None:
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import generate_structured

    responses = iter(
        [
            '{"text_ko":"겁 많지만 빈틈없다",',
            '{"text_ko":"겁 많지만 빈틈없다","text_en":"Fearful but meticulous.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}',
        ]
    )

    result = generate_structured(lambda _prompt: next(responses), "prompt", TaskAOutput)

    assert result.attempt_count == 2
    assert result.last_error_kind is None
    assert result.model.text_ko == "겁 많지만 빈틈없다"
    assert result.attempts[0].json_error == "JSONDecodeError"


def test_generate_structured_retries_on_enum_drift() -> None:
    from training.lib.output_schema import TaskFOutput
    from training.lib.structured_generation import generate_structured

    responses = iter(
        [
            '{"emotion":"panic","intensity":0.8,"cause_ko":"겁에 질렸다","cause_en":"Fear struck.","previous_emotion":"trust","transition_type":"sudden","temperament_amplifier":"high_HA"}',
            '{"emotion":"fear","intensity":0.8,"cause_ko":"겁에 질렸다","cause_en":"Fear struck.","previous_emotion":"trust","transition_type":"sudden","temperament_amplifier":"high_HA"}',
        ]
    )

    result = generate_structured(lambda _prompt: next(responses), "prompt", TaskFOutput)

    assert result.attempt_count == 2
    assert result.model.emotion == "fear"
    assert result.attempts[0].validation_error is not None
    assert "emotion" in result.attempts[0].validation_error


def test_generate_structured_retries_on_missing_field() -> None:
    from training.lib.output_schema import TaskEOutput
    from training.lib.structured_generation import generate_structured

    responses = iter(
        [
            '{"action_id":0,"confidence":0.7,"hint_ko":"곧장 물러섰다","hint_en":"They stepped back.","personality_reasoning":"high_HA"}',
            '{"action_id":0,"confidence":0.7,"hint_ko":"곧장 물러섰다","hint_en":"They stepped back.","personality_reasoning":"high_HA","temperament_factor":"harm_avoidance_dominant"}',
        ]
    )

    result = generate_structured(lambda _prompt: next(responses), "prompt", TaskEOutput)

    assert result.attempt_count == 2
    assert result.model.temperament_factor == "harm_avoidance_dominant"
    assert result.attempts[0].validation_error is not None
    assert "temperament_factor" in result.attempts[0].validation_error


def test_generate_structured_raises_after_exhausting_retries() -> None:
    from training.lib.output_schema import TaskCOutput
    from training.lib.structured_generation import StructuredGenerationError, generate_structured

    responses = iter(
        [
            '{"speech_ko":"앞으로 나서라","speech_en":"Step forward.","register":"haera","emotion_expressed":"anger"}',
            '{"speech_ko":"앞으로 나서라","speech_en":"Step forward.","register":"haera","emotion_expressed":"anger"}',
            '{"speech_ko":"앞으로 나서라","speech_en":"Step forward.","register":"haera","emotion_expressed":"anger"}',
        ]
    )

    try:
        generate_structured(lambda _prompt: next(responses), "prompt", TaskCOutput, max_retry=2)
    except StructuredGenerationError as exc:
        assert exc.attempt_count == 3
        assert exc.last_error_kind == "validation"
        assert exc.attempts[-1].validation_error is not None
        assert "speaker_role" in exc.attempts[-1].validation_error
    else:
        raise AssertionError("Expected retries to exhaust and raise StructuredGenerationError")


def test_generate_structured_retry_prompt_includes_validation_feedback_and_previous_json() -> None:
    from training.lib.output_schema import TaskEOutput
    from training.lib.structured_generation import generate_structured

    seen_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        if 'missing_required_field: "temperament_factor"' in prompt and '"personality_reasoning":"high_HA"' in prompt:
            return '{"action_id":0,"confidence":0.7,"hint_ko":"곧장 물러섰다","hint_en":"They stepped back.","personality_reasoning":"high_HA","temperament_factor":"harm_avoidance_dominant"}'
        return '{"action_id":0,"confidence":0.7,"hint_ko":"곧장 물러섰다","hint_en":"They stepped back.","personality_reasoning":"high_HA"}'

    result = generate_structured(fake_llm, "prompt", TaskEOutput)

    assert result.attempt_count == 2
    assert len(seen_prompts) == 2
    assert "The previous output failed validation." in seen_prompts[1]
    assert 'missing_required_field: "temperament_factor"' in seen_prompts[1]
    assert '"personality_reasoning":"high_HA"' in seen_prompts[1]
    assert "Return ONLY corrected JSON." in seen_prompts[1]
    assert result.attempts[0].attempt_index == 0
    assert result.attempts[0].raw_output.endswith('"personality_reasoning":"high_HA"}')


def test_repair_json_strips_fences_and_trailing_text() -> None:
    from training.lib.structured_generation import repair_json

    repaired = repair_json('```json\n{"text_ko":"조심스럽다"}\n```\nExplanation')

    assert repaired["text"] == '{"text_ko":"조심스럽다"}'
    assert [action["kind"] for action in repaired["repair_actions"]] == ["first_json_extract"]


def test_repair_json_closes_missing_final_brace() -> None:
    from training.lib.structured_generation import repair_json

    repaired = repair_json('{"text_ko":"조심스럽다"')

    assert repaired["text"] == '{"text_ko":"조심스럽다"}'
    assert repaired["repair_actions"][-1]["kind"] == "missing_closing_brace"


def test_generate_structured_filters_extra_keys_without_retry() -> None:
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import generate_structured

    result = generate_structured(
        lambda _prompt: '{"text_ko":"겁 많지만 빈틈없다","text_en":"Fearful but meticulous.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic","schema_explanation":"do not copy"}',
        "prompt",
        TaskAOutput,
    )

    assert result.attempt_count == 1
    assert "schema_explanation" not in result.payload
    assert result.attempts[0].repair_actions is not None
    assert any(action["kind"] == "filter_extra_keys" for action in result.attempts[0].repair_actions)


def test_generate_structured_corrects_casing_only_enum_without_retry() -> None:
    from training.lib.output_schema import TaskFOutput
    from training.lib.structured_generation import generate_structured

    result = generate_structured(
        lambda _prompt: '{"emotion":"Fear","intensity":0.8,"cause_ko":"겁에 질렸다","cause_en":"Fear struck.","previous_emotion":"Trust","transition_type":"Sudden","temperament_amplifier":"high_HA"}',
        "prompt",
        TaskFOutput,
    )

    assert result.attempt_count == 1
    assert result.payload["emotion"] == "fear"
    assert result.payload["previous_emotion"] == "trust"
    assert result.payload["transition_type"] == "sudden"
    assert result.attempts[0].repair_actions is not None
    assert any(action["kind"] == "correct_enum_value" for action in result.attempts[0].repair_actions)


def test_generate_structured_retry_prompt_includes_json_feedback_and_previous_output() -> None:
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import generate_structured

    seen_prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        seen_prompts.append(prompt)
        if "json_parse_error" in prompt and '{"text_ko":"겁 많지만 빈틈없다",' in prompt:
            return '{"text_ko":"겁 많지만 빈틈없다","text_en":"Fearful but meticulous.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}'
        return '{"text_ko":"겁 많지만 빈틈없다",'

    result = generate_structured(fake_llm, "prompt", TaskAOutput)

    assert result.attempt_count == 2
    assert "The previous output failed validation." in seen_prompts[1]
    assert "Problem type: json_parse_error" in seen_prompts[1]
    assert '{"text_ko":"겁 많지만 빈틈없다",' in seen_prompts[1]


def test_build_structured_constraint_exposes_schema_contract() -> None:
    from training.lib.output_schema import TaskGOutput
    from training.lib.structured_generation import build_structured_constraint

    constraint = build_structured_constraint(TaskGOutput)

    assert constraint.mode == "json_schema"
    assert constraint.schema_name == "TaskGOutput"
    assert "interpretation_ko" in constraint.allowed_keys
    assert "action_tendency" in constraint.enum_fields


def test_resolve_structured_decoding_reports_transformers_fallback() -> None:
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import build_structured_constraint, resolve_structured_decoding

    metadata = resolve_structured_decoding(build_structured_constraint(TaskAOutput), backend="transformers")

    assert metadata["requested_mode"] == "json_schema"
    assert metadata["enabled"] is False
    assert metadata["supported"] is False
    assert metadata["used_mode"] == "none"
    assert "transformers backend" in metadata["reason"]


def test_generate_structured_exposes_repair_and_decoding_metadata() -> None:
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import build_structured_constraint, generate_structured

    result = generate_structured(
        lambda _prompt: '```json\n{"text_ko":"겁 많지만 빈틈없다","text_en":"Fearful but meticulous.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic","schema_explanation":"do not copy"}\n```',
        "prompt",
        TaskAOutput,
        structured_constraint=build_structured_constraint(TaskAOutput),
    )

    assert result.repair_actions
    assert result.structured_decoding["enabled"] is False
    assert any(action["kind"] == "fence_strip" for action in result.repair_actions)
    assert any(action["kind"] == "filter_extra_keys" for action in result.repair_actions)


def test_output_schema_helpers_expose_task_schema_and_enum_fields() -> None:
    from training.lib.output_schema import TASK_ENUM_FIELDS, TaskGOutput, get_schema_for_task

    assert get_schema_for_task("G") is TaskGOutput
    assert "action_tendency" in TASK_ENUM_FIELDS["G"]
    assert "mobilize" in TASK_ENUM_FIELDS["G"]["action_tendency"]
