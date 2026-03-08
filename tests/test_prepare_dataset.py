import json
from pathlib import Path

import pytest
import yaml

from scripts.prepare_dataset import prepare_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_dataset_builds_final_dataset_and_manifest(tmp_path: Path) -> None:
    passed_file = tmp_path / "data" / "validated" / "passed.jsonl"
    negatives_file = tmp_path / "data" / "samples" / "negative_examples.jsonl"
    general_file = tmp_path / "data" / "samples" / "general_korean.jsonl"
    output_file = tmp_path / "data" / "final" / "training_dataset.jsonl"
    manifest_file = tmp_path / "artifacts" / "manifests" / "training_dataset_manifest.yaml"

    write_jsonl(
        passed_file,
        [{"task": "A", "output": "곧은 마음으로 앞장섰다.", "register": "haera"}],
    )
    write_jsonl(
        negatives_file,
        [{"task": "NEG", "output": "이 사람은 이다. 이 사람은 이다.", "label": "reject"}],
    )
    write_jsonl(
        general_file,
        [{"task": "GEN", "output": "강가에 물안개가 내려앉았다.", "label": "retain"}],
    )

    manifest = prepare_dataset(
        passed_file=passed_file,
        negative_samples_file=negatives_file,
        general_samples_file=general_file,
        output_file=output_file,
        manifest_file=manifest_file,
    )

    assert manifest["counts"]["validated"] == 1
    assert manifest["counts"]["negative"] == 1
    assert manifest["counts"]["general"] == 1
    assert manifest["counts"]["total"] == 3
    assert output_file.exists()
    assert manifest_file.exists()
    manifest_payload = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    assert manifest_payload["counts"] == {
        "validated": 1,
        "negative": 1,
        "general": 1,
        "total": 3,
    }


def test_prepare_dataset_honors_dataset_mix_even_with_explicit_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
dataset_mix:
  include_negative_samples: false
  include_general_samples: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    passed_file = tmp_path / "data" / "validated" / "passed.jsonl"
    negatives_file = tmp_path / "data" / "samples" / "negative_examples.jsonl"
    general_file = tmp_path / "data" / "samples" / "general_korean.jsonl"
    output_file = tmp_path / "data" / "final" / "training_dataset.jsonl"
    manifest_file = tmp_path / "artifacts" / "manifests" / "training_dataset_manifest.yaml"

    write_jsonl(passed_file, [{"task": "A", "output": "곧은 마음으로 앞장섰다.", "register": "haera"}])
    write_jsonl(negatives_file, [{"task": "NEG", "output": "이 사람은 이다. 이 사람은 이다.", "label": "reject"}])
    write_jsonl(general_file, [{"task": "GEN", "output": "강가에 물안개가 내려앉았다.", "label": "retain"}])

    result = prepare_dataset(
        repo_root=tmp_path,
        passed_file=passed_file,
        negative_samples_file=negatives_file,
        general_samples_file=general_file,
        output_file=output_file,
        manifest_file=manifest_file,
    )

    manifest_payload = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["counts"] == {
        "validated": 1,
        "negative": 0,
        "general": 1,
        "total": 2,
    }


def test_prepare_dataset_converts_layer3_and_layer4_rows_to_messages(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    prompts_dir = tmp_path / "prompts" / "training"
    config_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
prompts:
  training:
    layer3_system: prompts/training/layer3_system.txt
    layer4_system: prompts/training/layer4_system.txt
dataset_mix:
  include_negative_samples: false
  include_general_samples: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (prompts_dir / "layer3_system.txt").write_text("너는 석기시대 서사 도우미다. JSON으로만 답하라.", encoding="utf-8")
    (prompts_dir / "layer4_system.txt").write_text("너는 석기시대 서사 도우미다. 지시된 형식과 길이를 지켜라. 순우리말만 써라.", encoding="utf-8")

    passed_file = tmp_path / "data" / "validated" / "passed.jsonl"
    write_jsonl(
        passed_file,
        [
            {
                "task": "B",
                "layer": "L4",
                "prompt": "[TASK] B\n[PERS] 겁많음\n[SITU] 짐승발견",
                "output": "풀숲이 흔들렸다. 온몸이 오들오들 떨렸다.",
            },
            {
                "task": "E",
                "layer": "L3",
                "prompt": "[TASK] E\n[PERS] 겁많음\n[SITU] 짐승발견",
                "output": "{\"action_id\": 0, \"confidence\": 0.9, \"hint\": \"곧바로 달아났다\"}",
            },
        ],
    )
    write_jsonl(tmp_path / "data" / "samples" / "negative_examples.jsonl", [])
    write_jsonl(tmp_path / "data" / "samples" / "general_korean.jsonl", [])

    result = prepare_dataset(repo_root=tmp_path, dataset_name="worldsim-chat")
    rows = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert rows[0]["messages"][0]["content"] == "너는 석기시대 서사 도우미다. 지시된 형식과 길이를 지켜라. 순우리말만 써라."
    assert rows[1]["messages"][0]["content"] == "너는 석기시대 서사 도우미다. JSON으로만 답하라."
    assert rows[0]["messages"][1]["content"].startswith("[TASK] B")
    assert rows[1]["messages"][2]["content"].startswith("{\"action_id\"")


def test_prepare_dataset_rejects_unsafe_dataset_name(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
dataset_mix:
  include_negative_samples: false
  include_general_samples: false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(tmp_path / "data" / "validated" / "passed.jsonl", [])
    write_jsonl(tmp_path / "data" / "samples" / "negative_examples.jsonl", [])
    write_jsonl(tmp_path / "data" / "samples" / "general_korean.jsonl", [])

    with pytest.raises(ValueError, match="dataset_name"):
        prepare_dataset(repo_root=tmp_path, dataset_name="../../../tmp/pwn")


def test_prepare_dataset_converts_negative_and_general_rows_to_messages(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
dataset_mix:
  include_negative_samples: true
  include_general_samples: true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(
        tmp_path / "data" / "validated" / "passed.jsonl",
        [{"task": "A", "layer": "L4", "prompt": "[TASK] A", "output": "곧은 마음으로 앞장섰다."}],
    )
    write_jsonl(
        tmp_path / "data" / "samples" / "negative_examples.jsonl",
        [{"task": "NEG", "label": "reject", "output": "이 사람은 이다.", "reason": "repetition_loop"}],
    )
    write_jsonl(
        tmp_path / "data" / "samples" / "general_korean.jsonl",
        [{"task": "GEN", "label": "retain", "output": "강가에 물안개가 내려앉았다."}],
    )

    result = prepare_dataset(repo_root=tmp_path, dataset_name="worldsim-chat")
    rows = [json.loads(line) for line in result.dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert all("messages" in row for row in rows)
    negative_row = next(row for row in rows if row["task"] == "NEG")
    general_row = next(row for row in rows if row["task"] == "GEN")
    assert negative_row["messages"][1]["content"].endswith("이 사람은 이다.")
    assert negative_row["messages"][2]["content"] == "reject"
    assert general_row["messages"][2]["content"] == "강가에 물안개가 내려앉았다."


def test_prepare_dataset_requires_input_files_to_exist(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "generation.yaml").write_text(
        """
paths:
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
  negative_samples_file: data/samples/negative_examples.jsonl
  general_samples_file: data/samples/general_korean.jsonl
dataset_mix:
  include_negative_samples: true
  include_general_samples: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Required validated file does not exist"):
        prepare_dataset(repo_root=tmp_path)
