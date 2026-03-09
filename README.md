# WorldSim Training

WorldSim 전용 파인튜닝 데이터 생성, 검증, 학습, 평가를 위한 독립 레포다. 게임 런타임 코드는 이 레포에 두지 않는다.

## Workflow

1. `docs/design/`와 `docs/plans/`에서 설계와 실행 계획을 고정한다.
2. `config/`와 `prompts/`에서 생성 규칙과 프롬프트 자산을 관리한다.
3. `scripts/`로 `data/raw -> data/validated -> data/final` 흐름을 실행한다.
4. `training/`과 `eval/`에서 학습/평가 설정과 결과를 관리한다.
5. `artifacts/`에 병합 모델, GGUF, manifest를 남긴다.

## Key Paths

- `config/*.yaml`: 사람이 검토하는 선언형 설정
- `prompts/teacher/*`: 데이터 생성용 프롬프트 자산
- `scripts/generate_data.py`: config + prompts -> raw jsonl
- `scripts/validate_data.py`: raw jsonl -> validated jsonl
- `scripts/prepare_dataset.py`: validated + sample sets -> final dataset + manifest
- `data/samples/*.jsonl`: 부정 예시, 일반 한국어 샘플
- `artifacts/manifests/*.yaml`: 산출물 추적 정보

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 -m pytest
python3 scripts/generate_data.py --dry-run
```

## Usage

기본 파이프라인은 `generate_data.py -> validate_data.py -> prepare_dataset.py` 순서다. Task `A/B/C` 후처리 경로를 쓸 때는 `validate_data.py`와 최종 조립 사이에 보수적인 후처리 단계를 끼운다.

1. 생성이 끝나면 `create_dataset_snapshot.py`로 raw 결과와 `skipped` 결과를 함께 보존하고 `dataset_snapshot.json`을 남긴다.
2. `validate_postprocess.py`로 Task `A/B/C`를 다시 분류해 `passed`, `recoverable`, `review`, `failed` 버킷을 만든다.
3. `recover_skipped.py`로 `skipped.jsonl`에서 살릴 수 있는 행만 `recovered`로 복구하고, 나머지는 `needs_review` 또는 `unrecoverable`로 분리한다.
4. `sample_for_review.py`로 A/B/C와 recovered 버킷에서 사람 검토용 샘플을 뽑는다.
5. 사람 승인된 행만 `approved.jsonl`로 모은 뒤 `assemble_final_dataset.py`로 `train.jsonl`, `dev.jsonl`, `excluded.jsonl`, `dataset_manifest.json`을 만든다.
6. 최종 묶음이 확정되면 필요할 때 기존 `prepare_dataset.py`를 finalized corpora 기준으로 다시 실행한다.

세부 정책과 예시는 [docs/dataset_abc_postprocess_spec.md](/Users/rexxa/.config/superpowers/worktrees/worldsim-training/postprocess-abc-finalization/docs/dataset_abc_postprocess_spec.md)를 따른다.

세부 규칙은 각 디렉터리의 `AGENTS.md`를 따른다.
