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


def test_detect_runtime_raises_clear_error_when_bitsandbytes_import_fails(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from training.run_qlora_smoke import detect_runtime

    monkeypatch.setattr(
        "training.run_qlora_smoke._bitsandbytes_status",
        lambda: (False, "bitsandbytes import failed: ModuleNotFoundError: No module named 'bitsandbytes'"),
    )

    try:
        detect_runtime(prefer_qlora=True, require_qlora=True)
    except RuntimeError as exc:
        assert "bitsandbytes import failed" in str(exc)
    else:
        raise AssertionError("Expected missing bitsandbytes to hard-fail when QLoRA is required")


def test_model_is_4bit_quantized_detects_wrapped_peft_model() -> None:
    from training.run_qlora_smoke import _model_is_4bit_quantized

    model = SimpleNamespace(base_model=SimpleNamespace(model=SimpleNamespace(is_loaded_in_4bit=True)))

    assert _model_is_4bit_quantized(model) is True
