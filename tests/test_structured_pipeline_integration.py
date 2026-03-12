from __future__ import annotations

from training.lib.output_schema import TaskAOutput, TaskFOutput
from training.lib.structured_generation import StructuredGenerationError, generate_structured
from training.lib.structured_metrics import BatchMetrics


def test_structured_pipeline_valid_json_succeeds_first_try() -> None:
    metrics = BatchMetrics()
    result = generate_structured(
        lambda _prompt: '{"text_ko":"신중하다","text_en":"Careful.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}',
        "prompt",
        TaskAOutput,
        task_id="A",
        metrics_collector=metrics,
    )

    assert result.attempt_count == 1
    assert result.payload["register"] == "haera"
    assert metrics.total_successes == 1


def test_structured_pipeline_repairs_fenced_json() -> None:
    metrics = BatchMetrics()
    result = generate_structured(
        lambda _prompt: '```json\n{"text_ko":"신중하다","text_en":"Careful.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}\n```',
        "prompt",
        TaskAOutput,
        task_id="A",
        metrics_collector=metrics,
    )

    assert result.attempt_count == 1
    assert "fence_strip" in result.attempt_metrics[0].repairs_applied
    assert metrics.repair_applied_count == 1


def test_structured_pipeline_sanitizes_extra_keys() -> None:
    metrics = BatchMetrics()
    result = generate_structured(
        lambda _prompt: '{"text_ko":"신중하다","text_en":"Careful.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic","schema_explanation":"leaked"}',
        "prompt",
        TaskAOutput,
        task_id="A",
        metrics_collector=metrics,
    )

    assert "schema_explanation" not in result.payload
    assert result.attempt_metrics[0].keys_removed == ["schema_explanation"]
    assert metrics.keys_removed_count == 1


def test_structured_pipeline_normalizes_enum_case() -> None:
    metrics = BatchMetrics()
    result = generate_structured(
        lambda _prompt: '{"emotion":"Fear","intensity":0.8,"cause_ko":"겁에 질렸다","cause_en":"Fear struck.","previous_emotion":"Trust","transition_type":" Sudden ","temperament_amplifier":"high_HA"}',
        "prompt",
        TaskFOutput,
        task_id="F",
        metrics_collector=metrics,
    )

    assert result.payload["emotion"] == "fear"
    assert result.payload["previous_emotion"] == "trust"
    assert result.payload["transition_type"] == "sudden"
    assert metrics.enums_normalized_count == 1


def test_structured_pipeline_retries_on_garbage_and_records_metrics() -> None:
    metrics = BatchMetrics()
    prompts: list[str] = []
    responses = iter(
        [
            "not json at all",
            '{"text_ko":"신중하다","text_en":"Careful.","register":"haera","dominant_trait":"harm_avoidance","temperament_expressed":"melancholic"}',
        ]
    )

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    result = generate_structured(
        fake_llm,
        "prompt",
        TaskAOutput,
        task_id="A",
        metrics_collector=metrics,
    )

    assert result.attempt_count == 2
    assert metrics.total_attempts == 2
    assert metrics.required_retry == 1
    assert "The previous output failed validation." in prompts[1]


def test_structured_pipeline_raises_after_retry_exhaustion() -> None:
    metrics = BatchMetrics()

    try:
        generate_structured(
            lambda _prompt: "not json",
            "prompt",
            TaskAOutput,
            task_id="A",
            metrics_collector=metrics,
            max_retry=1,
        )
    except StructuredGenerationError as exc:
        assert exc.attempt_count == 2
        assert metrics.total_attempts == 2
        assert metrics.max_retries_exhausted == 1
    else:
        raise AssertionError("Expected garbage output to exhaust retries")
