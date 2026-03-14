from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_outlines_available_flag_is_bool() -> None:
    from training.lib.qlora_smoke import OUTLINES_AVAILABLE

    assert isinstance(OUTLINES_AVAILABLE, bool)


def test_create_outlines_model_returns_none_when_wrapper_unavailable(monkeypatch) -> None:
    from training.lib import qlora_smoke

    class BrokenOutlines:
        @staticmethod
        def from_transformers(model, tokenizer):
            raise RuntimeError("boom")

    monkeypatch.setattr(qlora_smoke, "OUTLINES_AVAILABLE", True)
    monkeypatch.setattr(qlora_smoke, "_outlines_module", BrokenOutlines)

    assert qlora_smoke._create_outlines_model(object(), object()) is None
    assert qlora_smoke._create_outlines_model(None, None) is None


def test_generate_sample_outlines_serializes_dict_result(monkeypatch) -> None:
    from training.lib import qlora_smoke
    from training.lib.output_schema import TaskAOutput
    from training.lib.structured_generation import OUTLINES_REPETITION_PENALTY

    class FakeGenerator:
        def __init__(self, outlines_model, schema_target):
            self.outlines_model = outlines_model
            self.schema_target = schema_target

        def __call__(self, prompt_text, **kwargs):
            assert prompt_text == "prompt"
            assert kwargs["max_new_tokens"] == 256
            assert kwargs["repetition_penalty"] == OUTLINES_REPETITION_PENALTY
            return {
                "text_ko": "앞장선다.",
                "text_en": "They lead.",
                "register": "haera",
                "dominant_trait": "harm_avoidance",
                "temperament_expressed": "steady",
            }

    class FakeOutlines:
        Generator = FakeGenerator

        @staticmethod
        def json_schema(schema):
            return {"schema": schema.__name__}

    monkeypatch.setattr(qlora_smoke, "_outlines_module", FakeOutlines)

    result = qlora_smoke._generate_sample_outlines(
        outlines_model={"wrapped": True},
        schema=TaskAOutput,
        prompt_text="prompt",
        max_new_tokens=256,
    )

    payload = json.loads(result)
    assert payload["register"] == "haera"


def test_generate_sample_outlines_serializes_pydantic_model(monkeypatch) -> None:
    from training.lib import qlora_smoke
    from training.lib.output_schema import TaskAOutput

    model_instance = TaskAOutput(
        text_ko="경계를 늦추지 않는다.",
        text_en="They stay alert.",
        register="hao",
        dominant_trait="harm_avoidance",
        temperament_expressed="melancholic",
    )

    class FakeGenerator:
        def __init__(self, outlines_model, schema_target):
            self.outlines_model = outlines_model
            self.schema_target = schema_target

        def __call__(self, prompt_text, **kwargs):
            return model_instance

    class FakeOutlines:
        Generator = FakeGenerator

    monkeypatch.setattr(qlora_smoke, "_outlines_module", FakeOutlines)

    result = qlora_smoke._generate_sample_outlines(
        outlines_model={"wrapped": True},
        schema=TaskAOutput,
        prompt_text="prompt",
        max_new_tokens=256,
    )

    payload = json.loads(result)
    assert payload["register"] == "hao"


def test_generate_samples_falls_back_when_outlines_generation_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    from training.lib import qlora_smoke
    from training.lib.output_schema import TaskAOutput
    from training.lib.qlora_smoke import RuntimeConfig
    from training.lib.structured_generation import StructuredGenerationResult
    from training.lib.structured_metrics import GenerationAttemptMetrics

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(use_cache=False)

        def eval(self) -> None:
            return None

    monkeypatch.setattr(qlora_smoke, "_create_outlines_model", lambda model, tokenizer: object())
    monkeypatch.setattr(qlora_smoke, "_generate_sample_outlines", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad outlines")))
    monkeypatch.setattr(qlora_smoke, "_build_sample_prompt_messages", lambda row: [{"role": "user", "content": "prompt"}])
    monkeypatch.setattr(qlora_smoke, "_sample_generation_assistant_prefix", lambda task: "")
    monkeypatch.setattr(qlora_smoke, "_sample_generation_output_schema", lambda task: TaskAOutput)
    monkeypatch.setattr(qlora_smoke, "render_conversation", lambda tokenizer, messages, add_generation_prompt: "prompt")
    monkeypatch.setattr(qlora_smoke, "write_jsonl", lambda path, rows: None)

    def fake_generate_structured(llm, prompt, schema, **kwargs):
        metrics_collector = kwargs["metrics_collector"]
        metrics_collector.record(
            GenerationAttemptMetrics(
                task_id="A",
                attempt_number=1,
                raw_length=64,
                json_parse_success=True,
                schema_validation_success=True,
                overall_success=True,
            )
        )
        validated = TaskAOutput(
            text_ko="앞을 본다.",
            text_en="They look ahead.",
            register="haera",
            dominant_trait="reward_dependence",
            temperament_expressed="sanguine",
        )
        payload = validated.model_dump(mode="json", by_alias=True)
        encoded = json.dumps(payload, ensure_ascii=False)
        return StructuredGenerationResult(
            model=validated,
            payload=payload,
            raw_output=encoded,
            candidate_output=encoded,
            attempts=[],
            attempt_count=1,
            last_error_kind=None,
            structured_decoding={"requested_mode": "json_schema", "used_mode": "none", "enabled": False, "supported": False},
            attempt_metrics=[],
        )

    monkeypatch.setattr(qlora_smoke, "generate_structured", fake_generate_structured)

    rows = [
        {
            "task": "A",
            "source_split": "eval",
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "{}"},
            ],
        }
    ]

    samples, metrics = qlora_smoke._generate_samples(
        FakeModel(),
        object(),
        rows,
        tmp_path / "sample_generations.jsonl",
        RuntimeConfig(device="cpu", use_qlora=False, fallback_reason=None, torch_dtype="float32"),
        SimpleNamespace(),
    )

    assert len(samples) == 1
    captured = capsys.readouterr()
    assert "[outlines] Constrained decoding ENABLED" in captured.out
    assert samples[0]["structured_decoding"]["used_mode"] == "repair_sanitize_fallback"
    assert "bad outlines" in samples[0]["structured_decoding"]["reason"]
    assert metrics["total_successes"] == 1
    assert metrics["per_sample_success_rate"] == 1.0


def test_post_sanitize_recovers_enum_drift_after_outlines() -> None:
    from training.lib.json_sanitize import sanitize_json_output
    from training.lib.output_schema import TaskBOutput

    outlines_output = {
        "text_ko": "떨리는 손으로 뒤를 살핀다.",
        "text_en": "With trembling hands, they look behind.",
        "register": "haera",
        "emotion_expressed": "sorrow",
        "intensity": 0.7,
        "mimetics": ["떨리는"],
        "temperament_influence": "melancholic caution",
    }

    sanitized, actions = sanitize_json_output(outlines_output, "B")
    validated = TaskBOutput.model_validate(sanitized)

    assert validated.emotion_expressed == "sadness"
    assert len(actions) > 0


def test_world_context_labels_are_in_leaky_strip_list() -> None:
    from training.lib.qlora_smoke import LEAKY_GENERATION_SECTION_LABELS
    from training.lib.structured_generation import OUTLINES_REPETITION_PENALTY, STRUCTURED_GENERATION_DEFAULTS

    assert "repetition_penalty" not in STRUCTURED_GENERATION_DEFAULTS
    assert OUTLINES_REPETITION_PENALTY == 1.2
    assert "세계관" in LEAKY_GENERATION_SECTION_LABELS
    assert "WORLD" in LEAKY_GENERATION_SECTION_LABELS
    assert "WORLD_DESC" in LEAKY_GENERATION_SECTION_LABELS
    assert "WORLD_VOCAB" in LEAKY_GENERATION_SECTION_LABELS
