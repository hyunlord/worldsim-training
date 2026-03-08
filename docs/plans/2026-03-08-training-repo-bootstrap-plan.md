# WorldSim Training Repo Bootstrap Plan

## Summary

- 학습 전용 라이프사이클 레포를 기준으로 스켈레톤을 만든다.
- 공통 규칙은 루트 `AGENTS.md`, 디렉터리별 규칙은 하위 `AGENTS.md`로 분리한다.
- `generate -> validate -> prepare` 파이프라인과 그 입력 계약을 먼저 고정한다.

## Decisions

- 게임 런타임 코드는 이 레포에 넣지 않는다.
- 설정과 프롬프트 자산은 `config/`, `prompts/`에서만 관리한다.
- 생성물은 `data/`, 산출물 메타데이터는 `artifacts/manifests/`로 고정한다.
- 첨부된 설계 문서는 원문을 날짜형 이름으로 `docs/design/`과 `docs/references/`에 보관한다.
