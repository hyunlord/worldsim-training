from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.common import ensure_directory, read_jsonl, write_jsonl
from scripts.prepare_dataset import _validate_messages_row


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_TASKS = ("A", "B", "C", "E", "F", "G", "H")


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    use_qlora: bool
    fallback_reason: str | None
    torch_dtype: str


@dataclass(slots=True)
class SmokeRunConfig:
    model_name: str = DEFAULT_MODEL_NAME
    train_file: Path = Path("data/training/worldsim-v31-mix-v1/train_converted.jsonl")
    dev_file: Path = Path("data/training/worldsim-v31-mix-v1/dev_converted.jsonl")
    output_dir: Path | None = None
    max_steps: int = 5
    max_train_samples: int = 32
    max_eval_samples: int = 16
    max_length: int = 512
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_TARGET_MODULES))
    seed: int = 42
    trust_remote_code: bool = False
    disable_qlora: bool = False
    require_qlora: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_file"] = str(self.train_file)
        payload["dev_file"] = str(self.dev_file)
        payload["output_dir"] = str(self.output_dir) if self.output_dir is not None else None
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass(slots=True)
class SmokeRunResult:
    success: bool
    status: str
    used_true_qlora: bool
    runtime: dict[str, Any] | None
    environment: dict[str, Any]
    output_dir: str
    summary_path: str
    config_snapshot: str | None
    metrics_path: str | None
    sample_path: str | None
    adapter_dir: str | None
    train_rows: int
    eval_rows: int
    train_task_counts: dict[str, int]
    eval_task_counts: dict[str, int]
    train_loss: float | None
    eval_loss: float | None
    finite_losses: bool | None
    blocker_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmokeRunBlockedError(RuntimeError):
    """Raised when a smoke run cannot proceed under the requested constraints."""


def coerce_smoke_config(value: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunConfig:
    if isinstance(value, SmokeRunConfig):
        return SmokeRunConfig(
            model_name=str(value.model_name),
            train_file=Path(value.train_file),
            dev_file=Path(value.dev_file),
            output_dir=Path(value.output_dir) if value.output_dir is not None else None,
            max_steps=int(value.max_steps),
            max_train_samples=int(value.max_train_samples),
            max_eval_samples=int(value.max_eval_samples),
            max_length=int(value.max_length),
            per_device_train_batch_size=int(value.per_device_train_batch_size),
            per_device_eval_batch_size=int(value.per_device_eval_batch_size),
            gradient_accumulation_steps=int(value.gradient_accumulation_steps),
            learning_rate=float(value.learning_rate),
            lora_r=int(value.lora_r),
            lora_alpha=int(value.lora_alpha),
            lora_dropout=float(value.lora_dropout),
            target_modules=tuple(str(module) for module in value.target_modules),
            seed=int(value.seed),
            trust_remote_code=bool(value.trust_remote_code),
            disable_qlora=bool(value.disable_qlora),
            require_qlora=bool(value.require_qlora),
            dry_run=bool(value.dry_run),
        )

    if isinstance(value, argparse.Namespace):
        payload = vars(value)
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"Unsupported smoke config input: {type(value).__name__}")

    target_modules = payload.get("target_modules", DEFAULT_TARGET_MODULES)
    output_dir = payload.get("output_dir")
    return SmokeRunConfig(
        model_name=str(payload.get("model_name", DEFAULT_MODEL_NAME)),
        train_file=Path(payload.get("train_file", SmokeRunConfig.train_file)),
        dev_file=Path(payload.get("dev_file", SmokeRunConfig.dev_file)),
        output_dir=Path(output_dir) if output_dir is not None else None,
        max_steps=int(payload.get("max_steps", 5)),
        max_train_samples=int(payload.get("max_train_samples", 32)),
        max_eval_samples=int(payload.get("max_eval_samples", 16)),
        max_length=int(payload.get("max_length", 512)),
        per_device_train_batch_size=int(payload.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(payload.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(payload.get("gradient_accumulation_steps", 1)),
        learning_rate=float(payload.get("learning_rate", 2e-4)),
        lora_r=int(payload.get("lora_r", 16)),
        lora_alpha=int(payload.get("lora_alpha", 32)),
        lora_dropout=float(payload.get("lora_dropout", 0.05)),
        target_modules=tuple(str(module) for module in target_modules),
        seed=int(payload.get("seed", 42)),
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        disable_qlora=bool(payload.get("disable_qlora", False)),
        require_qlora=bool(payload.get("require_qlora", False)),
        dry_run=bool(payload.get("dry_run", False)),
    )


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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def detect_runtime(prefer_qlora: bool, require_qlora: bool) -> RuntimeConfig:
    import torch

    device = "cpu"
    fallback_reason: str | None = None
    torch_dtype = "float32"
    use_qlora = False

    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        if prefer_qlora and _bitsandbytes_available():
            use_qlora = True
        elif prefer_qlora:
            fallback_reason = "bitsandbytes is unavailable in this environment"
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


def get_environment_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cwd": str(Path.cwd()),
    }

    try:
        import torch

        torch_info: dict[str, Any] = {
            "available": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            torch_info["cuda_device_count"] = torch.cuda.device_count()
            torch_info["cuda_device_name"] = torch.cuda.get_device_name(0)
            torch_info["cuda_bf16_supported"] = torch.cuda.is_bf16_supported()
        summary["torch"] = torch_info
    except Exception as exc:  # noqa: BLE001
        summary["torch"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    for module_name in ("transformers", "datasets", "peft", "accelerate", "bitsandbytes"):
        try:
            module = __import__(module_name)
            summary[module_name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:  # noqa: BLE001
            summary[module_name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return summary


def _torch_dtype(runtime: RuntimeConfig, torch: Any) -> Any:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[runtime.torch_dtype]


def _tokenize_rows(rows: list[dict], tokenizer: Any, max_length: int, Dataset: Any) -> Any:
    texts = [render_conversation(tokenizer, row["messages"], add_generation_prompt=False) for row in rows]
    encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return Dataset.from_dict(dict(encoded))


def _load_model_and_tokenizer(config: SmokeRunConfig, runtime: RuntimeConfig, libs: dict[str, Any]) -> tuple[Any, Any]:
    torch = libs["torch"]
    AutoTokenizer = libs["AutoTokenizer"]
    AutoModelForCausalLM = libs["AutoModelForCausalLM"]
    BitsAndBytesConfig = libs["BitsAndBytesConfig"]
    prepare_model_for_kbit_training = libs["prepare_model_for_kbit_training"]
    get_peft_model = libs["get_peft_model"]
    LoraConfig = libs["LoraConfig"]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
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

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if runtime.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
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


def preview_metrics(output_dir: Path | str) -> dict[str, Any]:
    metrics_path = Path(output_dir) / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_sample_generations(output_dir: Path | str) -> list[dict]:
    sample_path = Path(output_dir) / "sample_generations.jsonl"
    return read_jsonl(sample_path) if sample_path.exists() else []


def count_parseable_json_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = len(samples)
    parseable = sum(1 for row in samples if not row.get("json_parse_error"))
    return {
        "total": total,
        "parseable_json": parseable,
        "failed_json": total - parseable,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_blocked_result(
    *,
    output_dir: Path,
    summary_path: Path,
    config_snapshot_path: Path | None,
    metrics_path: Path | None,
    sample_path: Path | None,
    adapter_dir: Path | None,
    environment: dict[str, Any],
    runtime: RuntimeConfig | None,
    train_rows: list[dict],
    eval_rows: list[dict],
    blocker_reason: str,
) -> SmokeRunResult:
    return SmokeRunResult(
        success=False,
        status="blocked",
        used_true_qlora=bool(runtime and runtime.use_qlora),
        runtime=asdict(runtime) if runtime else None,
        environment=environment,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        config_snapshot=str(config_snapshot_path) if config_snapshot_path else None,
        metrics_path=str(metrics_path) if metrics_path else None,
        sample_path=str(sample_path) if sample_path else None,
        adapter_dir=str(adapter_dir) if adapter_dir else None,
        train_rows=len(train_rows),
        eval_rows=len(eval_rows),
        train_task_counts=_count_tasks(train_rows),
        eval_task_counts=_count_tasks(eval_rows),
        train_loss=None,
        eval_loss=None,
        finite_losses=None,
        blocker_reason=blocker_reason,
    )


def run_smoke(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    config = coerce_smoke_config(config_input)
    output_dir = _resolve_output_dir(config.output_dir)
    config_snapshot_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    sample_path = output_dir / "sample_generations.jsonl"
    metrics_path = output_dir / "metrics.json"
    adapter_dir = output_dir / "adapter"
    environment = get_environment_summary()
    runtime: RuntimeConfig | None = None
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    try:
        train_rows = pick_rows(load_message_rows(config.train_file), config.max_train_samples, config.seed)
        eval_rows = pick_rows(load_message_rows(config.dev_file), config.max_eval_samples, config.seed + 1)
        runtime = detect_runtime(prefer_qlora=not config.disable_qlora, require_qlora=config.require_qlora)

        _write_json(
            config_snapshot_path,
            {
                **config.to_dict(),
                "generated_at": datetime.now(UTC).isoformat(),
                "runtime": asdict(runtime),
                "environment": environment,
                "max_train_samples": len(train_rows),
                "max_eval_samples": len(eval_rows),
            },
        )

        if config.dry_run:
            result = SmokeRunResult(
                success=True,
                status="dry_run",
                used_true_qlora=runtime.use_qlora,
                runtime=asdict(runtime),
                environment=environment,
                output_dir=str(output_dir),
                summary_path=str(summary_path),
                config_snapshot=str(config_snapshot_path),
                metrics_path=None,
                sample_path=None,
                adapter_dir=None,
                train_rows=len(train_rows),
                eval_rows=len(eval_rows),
                train_task_counts=_count_tasks(train_rows),
                eval_task_counts=_count_tasks(eval_rows),
                train_loss=None,
                eval_loss=None,
                finite_losses=None,
            )
            _write_json(summary_path, result.to_dict())
            return result

        libs = _load_training_libraries()
        torch = libs["torch"]
        set_seed = libs["set_seed"]
        Dataset = libs["Dataset"]
        DataCollatorForLanguageModeling = libs["DataCollatorForLanguageModeling"]
        TrainingArguments = libs["TrainingArguments"]
        Trainer = libs["Trainer"]

        set_seed(config.seed)
        model, tokenizer = _load_model_and_tokenizer(config, runtime, libs)
        train_dataset = _tokenize_rows(train_rows, tokenizer, config.max_length, Dataset)
        eval_dataset = _tokenize_rows(eval_rows, tokenizer, config.max_length, Dataset)
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args_kwargs = build_training_arguments_kwargs(
            runtime,
            available_parameters=set(inspect.signature(TrainingArguments.__init__).parameters),
            output_dir=str(output_dir / "checkpoints"),
            max_steps=config.max_steps,
            train_batch_size=config.per_device_train_batch_size,
            eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            seed=config.seed,
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

        _write_json(
            metrics_path,
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "finite_losses": finite_losses,
            },
        )

        result = SmokeRunResult(
            success=True,
            status="ok",
            used_true_qlora=runtime.use_qlora,
            runtime=asdict(runtime),
            environment=environment,
            output_dir=str(output_dir),
            summary_path=str(summary_path),
            config_snapshot=str(config_snapshot_path),
            metrics_path=str(metrics_path),
            sample_path=str(sample_path),
            adapter_dir=str(adapter_dir),
            train_rows=len(train_rows),
            eval_rows=len(eval_rows),
            train_task_counts=_count_tasks(train_rows),
            eval_task_counts=_count_tasks(eval_rows),
            train_loss=train_loss,
            eval_loss=eval_loss,
            finite_losses=finite_losses,
        )
        _write_json(summary_path, result.to_dict())
        return result
    except Exception as exc:  # noqa: BLE001
        blocker_reason = f"{type(exc).__name__}: {exc}"
        if not config_snapshot_path.exists():
            _write_json(
                config_snapshot_path,
                {
                    **config.to_dict(),
                    "generated_at": datetime.now(UTC).isoformat(),
                    "runtime": asdict(runtime) if runtime else None,
                    "environment": environment,
                    "max_train_samples": len(train_rows),
                    "max_eval_samples": len(eval_rows),
                },
            )
        result = _build_blocked_result(
            output_dir=output_dir,
            summary_path=summary_path,
            config_snapshot_path=config_snapshot_path,
            metrics_path=metrics_path if metrics_path.exists() else None,
            sample_path=sample_path if sample_path.exists() else None,
            adapter_dir=adapter_dir if adapter_dir.exists() else None,
            environment=environment,
            runtime=runtime,
            train_rows=train_rows,
            eval_rows=eval_rows,
            blocker_reason=blocker_reason,
        )
        _write_json(summary_path, result.to_dict())
        return result


def run_smoke_or_raise(config_input: SmokeRunConfig | argparse.Namespace | Mapping[str, Any]) -> SmokeRunResult:
    result = run_smoke(config_input)
    if not result.success:
        raise SmokeRunBlockedError(result.blocker_reason or "Smoke run failed")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_smoke(parse_args(argv))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.success else 1
