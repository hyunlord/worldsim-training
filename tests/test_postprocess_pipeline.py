from __future__ import annotations

import json
from pathlib import Path

from scripts.common import read_jsonl, write_jsonl


def compact_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def bootstrap_postprocess_repo(tmp_path: Path) -> Path:
    for rel in [
        "config",
        "data/raw",
        "data/validated",
        "data/final",
        "artifacts/manifests",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)

    (tmp_path / "config/generation.yaml").write_text(
        """
paths:
  raw_dir: data/raw
  validated_dir: data/validated
  final_dir: data/final
  manifest_dir: artifacts/manifests
validation:
  forbidden_words: [식량, 전투]
  register_endings:
    haera: ['다[.\\s!?]?$', '는다[.\\s!?]?$', '했다[.\\s!?]?$', '쳤다[.\\s!?]?$']
    hao: ['오[.\\s!?]?$', '소[.\\s!?]?$', '시오[.\\s!?]?$']
    hae: ['해[.\\s!?]?$', '야[.\\s!?]?$', '지[.\\s!?]?$', '어[.\\s!?]?$']
  task_limits:
    A: {min_chars: 20, max_chars: 80, sentences: 1}
    B: {min_chars: 20, max_chars: 120, sentences: 2}
    C: {min_chars: 10, max_chars: 60, sentences: 1}
v31_context:
  temperament_axes:
    NS: novelty_seeking
    HA: harm_avoidance
    RD: reward_dependence
    P: persistence
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_normalize_emotion_and_register_variants() -> None:
    from scripts.lib.normalize import normalize_emotion, normalize_register

    assert normalize_emotion(" 공포 ") == "fear"
    assert normalize_emotion("Fear.") == "fear"
    assert normalize_emotion("EXPECTATION") == "anticipation"
    assert normalize_emotion("미지감정") is None

    assert normalize_register("해라체") == "haera"
    assert normalize_register(" HAO ") == "hao"
    assert normalize_register("hae.") == "hae"
    assert normalize_register("모름") is None


def test_classify_record_applies_canonical_task_policies(tmp_path: Path) -> None:
    repo_root = bootstrap_postprocess_repo(tmp_path)

    from scripts.lib.postprocess import classify_record, load_postprocess_policy

    policy = load_postprocess_policy(repo_root / "config")

    task_a_pass = {
        "task": "A",
        "register": "haera",
        "output": compact_json(
            {
                "text_ko": "뒤를 살피며 조심조심 걷는, 겁 많지만 빈틈없는 이다.",
                "text_en": "Walks cautiously looking behind, fearful but meticulous.",
                "register": "haera",
                "dominant_trait": "harm_avoidance",
                "temperament_expressed": "melancholic",
            }
        ),
    }
    task_a_legacy = {
        "task": "A",
        "register": "haera",
        "output": compact_json(
            {
                "text_ko": "뒤를 살피며 조심조심 걷는, 겁 많지만 빈틈없는 이다.",
                "text_en": "Walks cautiously looking behind, fearful but meticulous.",
                "register": "haera",
                "dominant_trait": "conscientiousness",
                "temperament_expressed": "melancholic",
            }
        ),
    }
    task_b_recoverable = {
        "task": "B",
        "register": "haera",
        "emotion_id": "joy",
        "situation_id": "predator",
        "output": compact_json(
            {
                "text_ko": "싱글벙글 웃으며 날랜 짐승을 빤히 노려본다. 겁내기보다 먼저 덤빌 때를 재어 본다.",
                "text_en": "Grinning, they stare the swift beast down. Rather than cower, they size up the moment to lunge first.",
                "register": "haera",
                "emotion_expressed": "anticipation",
                "intensity": 0.57,
                "mimetics": ["싱글벙글", "빤히"],
                "temperament_influence": "bold_impulsive_eagerness_overrides_fear",
            }
        ),
    }
    task_b_failed = {
        "task": "B",
        "register": "haera",
        "emotion_id": "joy",
        "situation_id": "predator",
        "output": compact_json(
            {
                "text_ko": "방긋 웃지만 온몸이 벌벌 떨린다. 겁에 질려 뒷걸음질쳤다.",
                "text_en": "Smiles brightly but trembles all over. Terrified, they step backward.",
                "register": "haera",
                "emotion_expressed": "joy",
                "intensity": 0.8,
                "mimetics": ["방긋", "벌벌"],
                "temperament_influence": "surface_smile_masks_fear",
            }
        ),
    }
    task_c_recoverable = {
        "task": "C",
        "register": "haera",
        "speaker_role": "chief",
        "output": compact_json(
            {
                "speech_ko": "당장 앞으로 나서라.",
                "speech_en": "Step forward right now.",
                "register": "해라체",
                "emotion_expressed": "ANGER",
                "speaker_role": "chief",
                "temperament_tone": "choleric_directness",
            }
        ),
    }

    result_a_pass = classify_record(task_a_pass, policy)
    result_a_legacy = classify_record(task_a_legacy, policy)
    result_b_recoverable = classify_record(task_b_recoverable, policy)
    result_b_failed = classify_record(task_b_failed, policy)
    result_c_recoverable = classify_record(task_c_recoverable, policy)

    assert result_a_pass.disposition == "passed"
    assert result_a_legacy.disposition == "review"
    assert "legacy_trait_schema_mismatch" in result_a_legacy.semantic_issues
    assert result_b_recoverable.disposition == "passed"
    assert result_b_failed.disposition == "failed"
    assert "emotion_text_contradiction" in result_b_failed.semantic_issues
    assert result_c_recoverable.disposition == "recoverable"
    assert result_c_recoverable.normalized_output["register"] == "haera"
    assert result_c_recoverable.normalized_output["emotion_expressed"] == "anger"


def test_create_snapshot_copies_inputs_and_writes_metadata(tmp_path: Path) -> None:
    repo_root = bootstrap_postprocess_repo(tmp_path / "repo")
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_file = source_dir / "generated.jsonl"
    skipped_file = source_dir / "skipped.jsonl"
    passed_file = source_dir / "passed.jsonl"
    failed_file = source_dir / "failed.jsonl"
    report_file = source_dir / "report.json"

    write_jsonl(raw_file, [{"task": "A", "output": compact_json({"text_ko": "가", "text_en": "a", "register": "haera", "dominant_trait": "harm_avoidance", "temperament_expressed": "melancholic"})}])
    write_jsonl(skipped_file, [{"task": "B", "skip_reason": "generation_validation_failed:invalid_emotion"}])
    write_jsonl(passed_file, [{"task": "A"}])
    write_jsonl(failed_file, [{"task": "B"}])
    report_file.write_text(json.dumps({"passed": 1, "failed": 1}, ensure_ascii=False), encoding="utf-8")

    from scripts.create_dataset_snapshot import create_snapshot

    result = create_snapshot(
        repo_root=repo_root,
        raw_file=raw_file,
        skipped_file=skipped_file,
        passed_file=passed_file,
        failed_file=failed_file,
        report_file=report_file,
        output_dir=repo_root / "artifacts" / "manifests" / "snapshots" / "run-01",
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.raw_snapshot.exists()
    assert result.skipped_snapshot.exists()
    assert read_jsonl(result.raw_snapshot)[0]["task"] == "A"
    assert metadata["source_files"]["raw_generated_file"] == str(raw_file)
    assert metadata["snapshot_files"]["raw_generated_file"] == str(result.raw_snapshot)
    assert metadata["postprocess_version"]


def test_recover_skipped_recovers_valid_rows_and_separates_unrecoverable(tmp_path: Path) -> None:
    repo_root = bootstrap_postprocess_repo(tmp_path / "repo")
    skipped_file = repo_root / "data" / "raw" / "skipped.jsonl"
    write_jsonl(
        skipped_file,
        [
            {
                "task": "B",
                "register": "haera",
                "emotion_id": "joy",
                "situation_id": "predator",
                "skip_reason": "generation_validation_failed:invalid_emotion",
                "output": compact_json(
                    {
                        "text_ko": "싱글벙글 웃으며 날랜 짐승을 빤히 노려본다. 겁내기보다 먼저 덤빌 때를 재어 본다.",
                        "text_en": "Grinning, they stare the swift beast down. Rather than cower, they size up the moment to lunge first.",
                        "register": "haera",
                        "emotion_expressed": "anticipation",
                        "intensity": 0.57,
                        "mimetics": ["싱글벙글", "빤히"],
                        "temperament_influence": "bold_impulsive_eagerness_overrides_fear",
                    }
                ),
            },
            {
                "task": "C",
                "register": "haera",
                "speaker_role": "chief",
                "skip_reason": "generation_validation_failed:not_json",
                "output": "not-json",
            },
        ],
    )

    from scripts.recover_skipped import recover_skipped

    result = recover_skipped(
        repo_root=repo_root,
        skipped_file=skipped_file,
        output_dir=repo_root / "data" / "validated" / "recovery",
    )

    recovered_rows = read_jsonl(result.recovered_path)
    unrecoverable_rows = read_jsonl(result.unrecoverable_path)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert len(recovered_rows) == 1
    assert recovered_rows[0]["postprocess"]["disposition"] in {"passed", "recoverable"}
    assert len(unrecoverable_rows) == 1
    assert report["counts_by_final_disposition"]["recovered"] == 1
    assert report["counts_by_final_disposition"]["unrecoverable"] == 1


def test_sample_for_review_writes_diverse_review_files(tmp_path: Path) -> None:
    repo_root = bootstrap_postprocess_repo(tmp_path / "repo")
    postprocess_dir = repo_root / "data" / "validated" / "postprocess"
    recovery_dir = repo_root / "data" / "validated" / "recovery"
    write_jsonl(
        postprocess_dir / "passed.jsonl",
        [
            {"task": "A", "personality_id": "p1", "world_id": "default", "postprocess": {"disposition": "passed"}, "output": compact_json({"text_ko": "하나", "text_en": "one", "register": "haera", "dominant_trait": "harm_avoidance", "temperament_expressed": "melancholic"})},
            {"task": "A", "personality_id": "p2", "world_id": "ocean", "postprocess": {"disposition": "passed"}, "output": compact_json({"text_ko": "둘둘둘둘둘둘둘둘둘둘", "text_en": "two", "register": "haera", "dominant_trait": "persistence", "temperament_expressed": "phlegmatic"})},
            {"task": "B", "personality_id": "p1", "world_id": "default", "situation_id": "predator", "postprocess": {"disposition": "passed"}, "output": compact_json({"text_ko": "오들오들 떨며 물러섰다. 곧 숨을 죽이고 둘레를 훑었다.", "text_en": "They trembled and stepped back. Then they watched quietly.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.7, "mimetics": ["오들오들"], "temperament_influence": "fear_sharpens_watchfulness"})},
            {"task": "B", "personality_id": "p2", "world_id": "winter", "situation_id": "storm", "postprocess": {"disposition": "review"}, "output": compact_json({"text_ko": "눈보라 속에서 이를 악물었다. 한걸음도 물러서지 않았다.", "text_en": "They clenched their teeth in the blizzard. They refused to step back.", "register": "haera", "emotion_expressed": "anger", "intensity": 0.6, "mimetics": [], "temperament_influence": "anger_overrides_fatigue"})},
            {"task": "C", "personality_id": "p3", "world_id": "default", "speaker_role": "chief", "postprocess": {"disposition": "passed"}, "output": compact_json({"speech_ko": "다들 앞으로 나서라.", "speech_en": "Everyone, step forward.", "register": "haera", "emotion_expressed": "anger", "speaker_role": "chief", "temperament_tone": "direct_command"})},
        ],
    )
    write_jsonl(
        recovery_dir / "recovered.jsonl",
        [
            {"task": "B", "personality_id": "p4", "world_id": "dungeon", "situation_id": "predator", "postprocess": {"disposition": "recoverable"}, "output": compact_json({"text_ko": "싱글벙글 웃으며 덤빌 때를 쟀다. 날랜 짐승 둘레를 천천히 훑었다.", "text_en": "They grinned and timed their strike. They slowly scanned around the swift beast.", "register": "haera", "emotion_expressed": "anticipation", "intensity": 0.5, "mimetics": ["싱글벙글"], "temperament_influence": "eager_patience"})}
        ],
    )

    from scripts.sample_for_review import sample_for_review

    result = sample_for_review(
        repo_root=repo_root,
        postprocess_dir=postprocess_dir,
        recovery_dir=recovery_dir,
        output_dir=repo_root / "data" / "validated" / "review_samples",
        target_a=2,
        target_b=2,
        target_c=1,
        target_recovered=1,
    )

    task_b_rows = read_jsonl(result["review_task_b"])
    recovered_rows = read_jsonl(result["review_recovered"])

    assert len(task_b_rows) == 2
    assert len(recovered_rows) == 1
    assert all("sample_reason" in row for row in task_b_rows)
    assert all("sample_reason" in row for row in recovered_rows)


def test_assemble_final_dataset_builds_conservative_train_dev_and_excluded(tmp_path: Path) -> None:
    repo_root = bootstrap_postprocess_repo(tmp_path / "repo")
    passed_file = repo_root / "data" / "validated" / "postprocess" / "passed.jsonl"
    recovered_file = repo_root / "data" / "validated" / "recovery" / "recovered.jsonl"
    review_file = repo_root / "data" / "validated" / "postprocess" / "review.jsonl"
    approved_review_file = repo_root / "data" / "validated" / "review_samples" / "approved.jsonl"
    unrecoverable_file = repo_root / "data" / "validated" / "recovery" / "unrecoverable.jsonl"

    write_jsonl(
        passed_file,
        [
            {"task": "A", "prompt": "[TASK] A", "output": compact_json({"text_ko": "뒤를 살피며 조심조심 걷는, 겁 많지만 빈틈없는 이다.", "text_en": "Walks cautiously looking behind, fearful but meticulous.", "register": "haera", "dominant_trait": "harm_avoidance", "temperament_expressed": "melancholic"}), "postprocess": {"disposition": "passed"}},
            {"task": "B", "prompt": "[TASK] B", "output": compact_json({"text_ko": "오들오들 떨며 물러섰다. 곧 숨을 죽이고 둘레를 훑었다.", "text_en": "They trembled and stepped back. Then they watched quietly.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.7, "mimetics": ["오들오들"], "temperament_influence": "fear_sharpens_watchfulness"}), "postprocess": {"disposition": "passed"}},
            {"task": "C", "prompt": "[TASK] C", "output": compact_json({"speech_ko": "다들 앞으로 나서라.", "speech_en": "Everyone, step forward.", "register": "haera", "emotion_expressed": "anger", "speaker_role": "chief", "temperament_tone": "direct_command"}), "postprocess": {"disposition": "passed"}},
        ],
    )
    write_jsonl(
        recovered_file,
        [
            {"task": "B", "prompt": "[TASK] B", "output": compact_json({"text_ko": "싱글벙글 웃으며 덤빌 때를 쟀다. 날랜 짐승 둘레를 천천히 훑었다.", "text_en": "They grinned and timed their strike. They slowly scanned around the swift beast.", "register": "haera", "emotion_expressed": "anticipation", "intensity": 0.5, "mimetics": ["싱글벙글"], "temperament_influence": "eager_patience"}), "postprocess": {"disposition": "recoverable"}},
            {"task": "B", "prompt": "[TASK] B", "output": compact_json({"text_ko": "오들오들 떨며 물러섰다. 곧 숨을 죽이고 둘레를 훑었다.", "text_en": "They trembled and stepped back. Then they watched quietly.", "register": "haera", "emotion_expressed": "fear", "intensity": 0.7, "mimetics": ["오들오들"], "temperament_influence": "fear_sharpens_watchfulness"}), "postprocess": {"disposition": "recoverable"}},
        ],
    )
    write_jsonl(review_file, [{"task": "A", "prompt": "[TASK] A", "output": compact_json({"text_ko": "옛 틀의 낱말로 성격을 적었다.", "text_en": "Used a legacy trait label.", "register": "haera", "dominant_trait": "conscientiousness", "temperament_expressed": "melancholic"}), "postprocess": {"disposition": "review", "semantic_issues": ["legacy_trait_schema_mismatch"]}}])
    write_jsonl(approved_review_file, [{"task": "A", "prompt": "[TASK] A", "output": compact_json({"text_ko": "발걸음은 느려도 끝까지 살핀다.", "text_en": "Though slow to move, they inspect things to the end.", "register": "haera", "dominant_trait": "persistence", "temperament_expressed": "phlegmatic"}), "postprocess": {"disposition": "approved_review"}}])
    write_jsonl(unrecoverable_file, [{"task": "C", "output": "not-json", "postprocess": {"disposition": "failed", "structural_issues": ["not_json"]}}])

    from scripts.assemble_final_dataset import assemble_final_dataset

    result = assemble_final_dataset(
        repo_root=repo_root,
        passed_file=passed_file,
        recovered_file=recovered_file,
        approved_review_file=approved_review_file,
        review_file=review_file,
        unrecoverable_file=unrecoverable_file,
        output_dir=repo_root / "data" / "final" / "abc-final",
        dataset_name="abc-final",
        dev_fraction=0.25,
        seed=7,
    )

    train_rows = read_jsonl(result["train_path"])
    dev_rows = read_jsonl(result["dev_path"])
    excluded_rows = read_jsonl(result["excluded_path"])
    manifest = json.loads(result["manifest_path"].read_text(encoding="utf-8"))

    train_outputs = {row["output"] for row in train_rows}
    dev_outputs = {row["output"] for row in dev_rows}

    assert train_outputs.isdisjoint(dev_outputs)
    assert any(row["exclusion_reason"] == "duplicate_content" for row in excluded_rows)
    assert any(row["exclusion_reason"] == "review_not_approved" for row in excluded_rows)
    assert any(row["exclusion_reason"] == "unrecoverable_source" for row in excluded_rows)
    assert manifest["normalization_version"]
    assert manifest["validator_version"]
    assert manifest["included_recovered_rows"] == 1
