# WorldSim Training Phase A Update: Layer 3 + Layer 4

## Summary

- 기존 Layer 4 텍스트 task A-D에 Layer 3 JSON task E-F를 추가한다.
- Task E는 상황별 닫힌 행동 선택, Task F는 감정 전이를 구조화 JSON으로 학습한다.
- 검증 단계는 금지어 자동 치환과 Layer 3 JSON 스키마 검증을 함께 수행한다.

## Asset Changes

- `config/generation.yaml`에 task E/F 메타데이터, JSON 검증 규칙, 자동 치환 규칙, grammar 경로를 추가한다.
- `config/situations.yaml`에 상황별 `action_options`를 추가해 스크립트가 닫힌 선택지를 설정에서 읽게 한다.
- `prompts/teacher/task_e.txt`, `prompts/teacher/task_f.txt`를 추가해 teacher 생성 입력을 분리한다.
- `prompts/training/layer3_system.txt`, `prompts/training/layer4_system.txt`를 추가해 prepare 단계가 layer별 system prompt를 선택할 수 있게 한다.
- `grammars/task_e_action.gbnf`, `grammars/task_f_emotion.gbnf`를 추가해 추론 단계 제약 자산을 함께 보관한다.

## Counts

- Task A: 10 성격 x 5 변형
- Task B: 10 상황 x 8 감정 x 10 성격 x 2 변형
- Task C: 10 상황 x 10 성격 x 3 변형
- Task D: 10 상황 x 5 변형
- Task E: 10 상황 x 10 성격 x 3 변형
- Task F: 10 상황 x 8 현재감정 x 10 성격 x 1 변형

Task B 변형 수는 Layer 3 추가분을 흡수하기 위해 3에서 2로 낮춘다.
