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
    )

    assert kwargs["eval_strategy"] == "no"
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
    assert "JSON object 하나만 출력하라." in user_content
    assert "첫 글자는 반드시 { 여야 한다." in user_content
    assert "형식 예시나 placeholder 문구를 복사하지 마라." in user_content
    assert "모든 key 이름과 문자열 값은 JSON 큰따옴표를 써라." in user_content


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
    assert "emotion_expressed must be exactly one of: joy, sadness" not in user_content
    assert "[규칙]" not in user_content
    assert "JSON만 출력하라" not in user_content


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
    assert "interpretation_ko는 한국어 한두 문장만 짧게 써라." in user_content
    assert "action_tendency는 정확히 one of" in user_content
    assert "misinterpretation_type는 정확히 one of" in user_content


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
    assert "key 순서는 text_ko, text_en, register, dominant_trait, temperament_expressed 이다." in prompt_a[-1]["content"]
    assert "Human:" not in prompt_a[-1]["content"]
    assert "Assistant:" not in prompt_a[-1]["content"]
    assert "text_ko와 text_en에는 실제 묘사 문장을 쓰고 형용사 이름만 단독으로 쓰지 마라." in prompt_a[-1]["content"]
    assert "register는 숫자가 아니라 haera, hao, hae 중 문자열 하나다." in prompt_a[-1]["content"]
    assert "자기소개나 대화 라벨을 쓰지 마라." in prompt_a[-1]["content"]
    assert "dominant_trait는 정확히 one of" in prompt_a[-1]["content"]

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
    assert "key 순서는 speech_ko, speech_en, register, emotion_expressed, speaker_role, temperament_tone 이다." in prompt_c[-1]["content"]
    assert "emotion_expressed는 정확히 one of" in prompt_c[-1]["content"]
    assert "emotion_expressed에는 enum 목록 전체를 쓰지 말고 하나만 써라." in prompt_c[-1]["content"]
    assert "emotion_expressed는 JSON 배열이 아니라 문자열 하나다." in prompt_c[-1]["content"]
    assert "speech_ko와 speech_en에는 자기소개를 쓰지 말고 바로 대사를 써라." in prompt_c[-1]["content"]
    assert "실제 대사만 쓰고 지시문을 따라 적지 마라." in prompt_c[-1]["content"]
    assert "speaker_role은 정확히 one of" in prompt_c[-1]["content"]


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

    assert "placeholder 문구를 그대로 쓰지 마라." in prompts["B"]
    assert "text_ko와 text_en에는 길이 설명문이나 schema 문구를 쓰지 마라." in prompts["B"]
    assert "key 순서는 action_id, confidence, hint_ko, hint_en, personality_reasoning, temperament_factor 이다." in prompts["E"]
    assert "emotion은 정확히 one of" in prompts["F"]
    assert "confidence만 숫자이고 나머지 enum field는 문자열이다." in prompts["G"]
    assert "모든 key 이름은 반드시 JSON 큰따옴표를 써라." in prompts["F"]
    assert "자기소개를 쓰지 마라." in prompts["G"]
    assert "interpretation_ko와 interpretation_en에는 placeholder 문구를 쓰지 마라." in prompts["G"]
    assert "허용 key는 name, description_en, resource_modifiers, special_zones, special_resources, agent_modifiers 뿐이다." in prompts["H"]
    assert "emotion_expressed에는 enum 목록 전체를 쓰지 말고 하나만 써라." in prompts["B"]
    assert "hint_en에는 실제 영어 이유를 써라." in prompts["E"]
    assert "hint_ko와 hint_en에는 길이 설명문이나 예시 문구를 쓰지 마라." in prompts["E"]
    assert "action_id와 confidence만 숫자이고 나머지는 문자열이다." in prompts["E"]
    assert "cause_ko와 cause_en에는 선택지나 규칙 문구를 쓰지 마라." in prompts["F"]
    assert "emotion과 previous_emotion에는 숫자를 쓰지 마라." in prompts["F"]
    assert "action_tendency와 misinterpretation_type에는 목록 전체를 쓰지 말고 하나만 써라." in prompts["G"]


def test_sample_generation_max_new_tokens_g_uses_larger_budget() -> None:
    from training.lib.qlora_smoke import _sample_generation_max_new_tokens

    assert _sample_generation_max_new_tokens("G") == 512
    assert _sample_generation_max_new_tokens("H") == 512
    assert _sample_generation_max_new_tokens("F") == 288


def test_sample_generation_assistant_prefix_is_task_specific() -> None:
    from training.lib.qlora_smoke import _sample_generation_assistant_prefix

    assert _sample_generation_assistant_prefix("A") == '{"text_ko": "'
    assert _sample_generation_assistant_prefix("B") == '{"text_ko": "'
    assert _sample_generation_assistant_prefix("C") == '{"speech_ko": "'
    assert _sample_generation_assistant_prefix("E") == '{"action_id": '
    assert _sample_generation_assistant_prefix("F") == '{"emotion": "'
    assert _sample_generation_assistant_prefix("G") == '{"interpretation_ko": "'


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
