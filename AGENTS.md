# WorldSim Training Instructions

이 레포는 WorldSim 학습 데이터 생성, 검증, 학습, 평가만 다룬다. 게임 런타임 코드나 게임 레포 전용 파일은 추가하지 않는다.

## Core Rules

- 모든 자동화 작업은 먼저 현재 디렉터리와 더 깊은 하위 디렉터리의 `AGENTS.md`를 확인한다.
- 기본 흐름은 `docs/design -> docs/plans -> config/prompts -> scripts -> data -> training/eval/artifacts` 순서를 따른다.
- 설정값, 프롬프트, 검증 규칙은 각각 `config/`와 `prompts/`에 둔다. 스크립트에 하드코딩하지 않는다.
- 산출물은 정해진 디렉터리에만 쓴다. 임의의 새 출력 디렉터리를 만들지 않는다.
- 대용량 생성물은 `data/raw`, `data/validated`, `data/final`, `artifacts/merged`, `artifacts/gguf` 아래에만 둔다.

## Repo Boundaries

- 허용: 데이터 생성, 데이터 QA, 데이터셋 조립, 학습 설정, 평가 리포트, 산출물 manifest
- 금지: 게임 레포 코드, 런타임 프롬프트 빌더 구현, 엔진/클라이언트 자산, 임의 실험 파일 난립
