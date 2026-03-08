import json
from pathlib import Path

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
