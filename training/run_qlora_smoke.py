#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import ensure_directory, read_jsonl, write_jsonl
from scripts.prepare_dataset import _validate_messages_row


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_TASKS = ("A", "B", "C", "E", "F", "G", "H")


@dataclass
class RuntimeConfig:
    device: str
    use_qlora: bool
    fallback_reason: str | None
    torch_dtype: str


def load_message_rows(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    validated: list[dict] = []
    for index, row in enumerate(rows, start=1):
        try:
            _validate_messages_row(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{index}: {exc}") from exc
        messages = row["messages"]
        if messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
            raise ValueError(f"{path}:{index}: assistant target must be present and non-empty")
        validated.append(row)
    return validated


def render_conversation(tokenizer: Any, messages: list[dict], *, add_generation_prompt: bool) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    rendered: list[str] = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def _load_training_libraries():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def _bitsandbytes_status() -> tuple[bool, str | None]:
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return False, f"bitsandbytes import failed: {type(exc).__name__}: {exc}"
    return True, None


def _model_is_4bit_quantized(model: Any) -> bool:
    pending = [model]
    visited: set[int] = set()

    while pending:
        candidate = pending.pop()
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in visited:
            continue
        visited.add(candidate_id)

        if getattr(candidate, "is_loaded_in_4bit", False):
            return True

        pending.append(getattr(candidate, "model", None))
        pending.append(getattr(candidate, "base_model", None))

    return False


def detect_runtime(prefer_qlora: bool, require_qlora: bool) -> RuntimeConfig:
    import torch

    device = "cpu"
    fallback_reason: str | None = None
    torch_dtype = "float32"
    use_qlora = False

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        bitsandbytes_available, bitsandbytes_error = _bitsandbytes_status()
        if prefer_qlora and bitsandbytes_available:
            use_qlora = True
        elif prefer_qlora:
            fallback_reason = bitsandbytes_error or "bitsandbytes is unavailable in this environment"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = "float32"
        if prefer_qlora:
            fallback_reason = "Apple Silicon MPS does not support bitsandbytes 4-bit QLoRA"
    elif prefer_qlora:
        fallback_reason = "CUDA is unavailable; true QLoRA requires CUDA + bitsandbytes"

    if require_qlora and not use_qlora:
        raise RuntimeError(f"QLoRA was required but is unavailable: {fallback_reason or 'unknown reason'}")

    return RuntimeConfig(
        device=device,
        use_qlora=use_qlora,
        fallback_reason=fallback_reason,
        torch_dtype=torch_dtype,
    )


def _torch_dtype(runtime: RuntimeConfig, torch: Any) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[runtime.torch_dtype]


def _tokenize_rows(rows: list[dict], tokenizer: Any, max_length: int, Dataset: Any) -> Any:
    formatted = [{"task": row.get("task"), "messages": row["messages"], "text": render_conversation(tokenizer, row["messages"], add_generation_prompt=False)} for row in rows]
    dataset = Dataset.from_list(formatted)

    def tokenize_batch(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized

    return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)


def _load_model_and_tokenizer(args: argparse.Namespace, runtime: RuntimeConfig, libs: dict[str, Any]) -> tuple[Any, Any]:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForCausalLM = libs["AutoModelForCausalLM"]
    BitsAndBytesConfig = libs["BitsAndBytesConfig"]
    prepare_model_for_kbit_training = libs["prepare_model_for_kbit_training"]
    get_peft_model = libs["get_peft_model"]
    LoraConfig = libs["LoraConfig"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if runtime.use_qlora:
        quant_dtype = _torch_dtype(runtime, torch)
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=quant_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = _torch_dtype(runtime, torch)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except Exception as exc:  # noqa: BLE001
        if runtime.use_qlora:
            raise RuntimeError(
                "True QLoRA 4-bit model load failed on CUDA via bitsandbytes: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        raise

    if runtime.use_qlora and not _model_is_4bit_quantized(model):
        raise RuntimeError(
            "True QLoRA was requested, but the loaded model is not marked as 4-bit bitsandbytes quantized."
        )

    if runtime.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model, tokenizer


def pick_rows(rows: list[dict], limit: int, seed: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    task_first: list[dict] = []
    seen_tasks: set[str] = set()
    for row in rows:
        task = str(row.get("task", "unknown"))
        if task not in seen_tasks:
            task_first.append(row)
            seen_tasks.add(task)
    if len(task_first) >= limit:
        return task_first[:limit]

    remaining = [row for row in rows if row not in task_first]
    rng.shuffle(remaining)
    return task_first + remaining[: limit - len(task_first)]


def _count_tasks(rows: list[dict]) -> dict[str, int]:
    counter = Counter(row.get("task", "unknown") for row in rows)
    return dict(sorted(counter.items()))


def _resolve_output_dir(base_output_dir: Path | None) -> Path:
    if base_output_dir is not None:
        return ensure_directory(base_output_dir)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return ensure_directory(Path("outputs") / "smoke" / "worldsim-v31-mix-v1" / timestamp)


def build_training_arguments_kwargs(
    runtime: RuntimeConfig,
    *,
    available_parameters: set[str],
    output_dir: str,
    max_steps: int,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "logging_steps": 1,
        "report_to": [],
        "seed": seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }
    if "eval_strategy" in available_parameters:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in available_parameters:
        kwargs["evaluation_strategy"] = "no"
    if "save_strategy" in available_parameters:
        kwargs["save_strategy"] = "no"
    if runtime.device == "cpu" and "use_cpu" in available_parameters:
        kwargs["use_cpu"] = True
    elif runtime.device == "cpu" and "no_cuda" in available_parameters:
        kwargs["no_cuda"] = True
    elif runtime.device == "mps" and "use_mps_device" in available_parameters:
        kwargs["use_mps_device"] = True
    return kwargs


def build_trainer_kwargs(
    *,
    available_parameters: set[str],
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    data_collator: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    if "processing_class" in available_parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in available_parameters:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def _select_generation_rows(train_rows: list[dict], eval_rows: list[dict]) -> list[dict]:
    candidates = eval_rows + train_rows
    picked: list[dict] = []
    seen: set[str] = set()
    for task in DEFAULT_TASKS:
        for row in candidates:
            if row.get("task") == task and task not in seen:
                picked.append(row)
                seen.add(task)
                break
    return picked


def _generate_samples(model: Any, tokenizer: Any, rows: list[dict], output_path: Path, runtime: RuntimeConfig, torch: Any) -> list[dict]:
    if not rows:
        write_jsonl(output_path, [])
        return []

    samples: list[dict] = []
    model.eval()
    model.config.use_cache = True
    device = runtime.device

    for row in rows:
        prompt_messages = row["messages"][:-1]
        prompt_text = render_conversation(tokenizer, prompt_messages, add_generation_prompt=True)
        encoded = tokenizer(prompt_text, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=160,
                do_sample=False,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_tokens = generated[0][encoded["input_ids"].shape[1] :]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        parse_error = None
        try:
            json.loads(generated_text)
        except Exception as exc:  # noqa: BLE001
            parse_error = type(exc).__name__
        samples.append(
            {
                "task": row.get("task"),
                "source_split": row.get("source_split"),
                "prompt_messages": prompt_messages,
                "expected_assistant": row["messages"][-1]["content"],
                "generated_assistant": generated_text,
                "json_parse_error": parse_error,
            }
        )

    write_jsonl(output_path, samples)
    return samples


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = load_message_rows(args.train_file)
    eval_rows = load_message_rows(args.dev_file)
    train_rows = pick_rows(train_rows, args.max_train_samples, args.seed)
    eval_rows = pick_rows(eval_rows, args.max_eval_samples, args.seed + 1)

    runtime = detect_runtime(prefer_qlora=not args.disable_qlora, require_qlora=args.require_qlora)
    output_dir = _resolve_output_dir(args.output_dir)
    config_snapshot_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "sample_generations.jsonl"
    metrics_path = output_dir / "metrics.json"
    adapter_dir = output_dir / "adapter"

    config_snapshot = {
        "model_name": args.model_name,
        "train_file": str(args.train_file),
        "dev_file": str(args.dev_file),
        "max_steps": args.max_steps,
        "max_train_samples": len(train_rows),
        "max_eval_samples": len(eval_rows),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "seed": args.seed,
        "runtime": asdict(runtime),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    config_snapshot_path.write_text(json.dumps(config_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        summary = {
            "status": "dry_run",
            "runtime": asdict(runtime),
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "train_task_counts": _count_tasks(train_rows),
            "eval_task_counts": _count_tasks(eval_rows),
            "output_dir": str(output_dir),
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    libs = _load_training_libraries()
    torch = libs["torch"]
    set_seed = libs["set_seed"]
    Dataset = libs["Dataset"]
    DataCollatorForLanguageModeling = libs["DataCollatorForLanguageModeling"]
    TrainingArguments = libs["TrainingArguments"]
    Trainer = libs["Trainer"]

    set_seed(args.seed)
    model, tokenizer = _load_model_and_tokenizer(args, runtime, libs)
    qlora_verified = runtime.use_qlora and _model_is_4bit_quantized(model)
    train_dataset = _tokenize_rows(train_rows, tokenizer, args.max_length, Dataset)
    eval_dataset = _tokenize_rows(eval_rows, tokenizer, args.max_length, Dataset)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args_kwargs = build_training_arguments_kwargs(
        runtime,
        available_parameters=set(inspect.signature(TrainingArguments.__init__).parameters),
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        train_batch_size=args.per_device_train_batch_size,
        eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        **build_trainer_kwargs(
            available_parameters=set(inspect.signature(Trainer.__init__).parameters),
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
        )
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    ensure_directory(adapter_dir)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    sample_rows = _select_generation_rows(train_rows, eval_rows)
    samples = _generate_samples(model, tokenizer, sample_rows, sample_path, runtime, torch)

    train_loss = float(train_result.training_loss)
    eval_loss = float(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None
    finite_losses = math.isfinite(train_loss) and (eval_loss is None or math.isfinite(eval_loss))

    metrics_payload = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "finite_losses": finite_losses,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "status": "ok",
        "runtime": asdict(runtime),
        "qlora_verified": qlora_verified,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "train_task_counts": _count_tasks(train_rows),
        "eval_task_counts": _count_tasks(eval_rows),
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "finite_losses": finite_losses,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "sample_generations": len(samples),
        "config_snapshot": str(config_snapshot_path),
        "metrics_path": str(metrics_path),
        "sample_path": str(sample_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal WorldSim QLoRA smoke training job.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base model to adapt. Defaults to a small public Qwen instruct model for smoke validation.")
    parser.add_argument("--train-file", type=Path, default=Path("data/training/worldsim-v31-mix-v1/train_converted.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/training/worldsim-v31-mix-v1/dev_converted.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output directory. Defaults to outputs/smoke/worldsim-v31-mix-v1/<timestamp>.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=32)
    parser.add_argument("--max-eval-samples", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-qlora", action="store_true", help="Force plain LoRA even if CUDA + bitsandbytes are available.")
    parser.add_argument("--require-qlora", action="store_true", help="Fail instead of falling back when true 4-bit QLoRA is unavailable.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and runtime selection without loading model weights.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_smoke(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
