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

세부 규칙은 각 디렉터리의 `AGENTS.md`를 따른다.
