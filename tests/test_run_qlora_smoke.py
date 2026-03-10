from __future__ import annotations

import json
from pathlib import Path


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


def test_notebook_uses_shared_training_module() -> None:
    notebook_path = Path("notebooks/dgx_spark_qlora_smoke.ipynb")
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))

    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )

    assert "from training.lib.qlora_smoke import" in source
