# Dataset ABC Postprocess Spec

## Goal

Task A/B/C 산출물은 생성 직후의 1차 검증 결과만으로 최종 채택하지 않는다. 이 문서는 `A`, `B`, `C` 전용 후처리 단계에서 무엇을 정규화하고, 무엇을 자동 통과시키며, 무엇을 사람 검토나 치명 실패로 보내는지에 대한 정본 정책을 정의한다.

핵심 원칙은 보수성이다.

- 후처리는 `register`, `emotion_expressed`, `dominant_trait` 같은 열거형 필드를 결정적으로 정규화한다.
- 후처리는 검증 단계와 같은 금칙어 치환, 바깥 따옴표 제거, 공백/구두점 정리처럼 손실 없는 표면 수리만 허용한다.
- 후처리는 본문 의미를 다시 창작하거나, 자유 텍스트를 새로 쓰는 식의 보정을 하지 않는다.
- 의미를 바꾸는 추정 보정은 하지 않고, 애매하면 `review`, 구조가 깨졌거나 모순이면 `fatal`로 보낸다.

## Scope

- 대상 태스크: `A`, `B`, `C`
- 입력: 생성 직후 `data/raw/*.jsonl`, 1차 검증에서 밀려난 `skipped.jsonl`
- 후처리 버킷:
  - `passed`: 이미 정본 정책을 만족한다.
  - `recoverable`: 결정적 정규화만으로 안전하게 살릴 수 있다.
  - `review`: 의미상 그럴듯하지만 자동 보정이 위험하다.
  - `fatal`: 구조가 깨졌거나 의미 충돌이 있어 신뢰할 수 없다.

실제 파일 단위에서는 `fatal`이 분류기 단계에서 `failed`, 복구 단계에서 `unrecoverable`로 드러날 수 있다. 분류 명칭은 달라도 정책상 같은 치명 실패 버킷으로 본다.

## Canonical Policy By Task

| Task | Canonical fields | Pass criteria | Review trigger | Fatal trigger |
| --- | --- | --- | --- | --- |
| `A` | `text_ko`, `text_en`, `register`, `dominant_trait`, `temperament_expressed` | `dominant_trait`가 TCI 4축 중 하나이고, `register`와 `temperament_expressed`가 요청 문맥과 맞는다. | 레거시 성격 축처럼 사람이 의미를 재판단해야 하는 경우 | JSON 파싱 실패, 필수 필드 누락, 문맥과 양립 불가한 값 |
| `B` | `text_ko`, `text_en`, `register`, `emotion_expressed`, `intensity`, `mimetics`, `temperament_influence` | 감정 라벨, 한국어 본문, 의성어/의태어, 기질 영향 설명이 서로 일관된다. | 의미가 크게 틀리진 않지만 자동 판정이 확신되지 않는 경우 | 감정 라벨과 본문이 정면으로 충돌하거나 구조가 깨진 경우 |
| `C` | `speech_ko`, `speech_en`, `register`, `emotion_expressed`, `speaker_role`, `temperament_tone` | 직접화법 1문장이고, 역할/감정/어체가 정규화 후 일관된다. | 의미는 통하지만 역할 또는 톤 해석을 사람이 확인해야 하는 경우 | JSON 파싱 실패, 필수 필드 누락, 정규화 후에도 어체/감정을 확정할 수 없는 경우 |

## Normalization Rules

후처리 정규화는 아래처럼 손실 없는 규칙만 허용한다.

1. 문자열 앞뒤 공백을 제거한다.
2. 열거형 문자열 끝의 마침표 같은 사소한 구두점을 제거한다.
3. ASCII 기반 코드값은 소문자로 맞춘다.
4. `register`는 다음 정본 코드로만 보관한다.
   - `해라`, `해라체`, `haera` -> `haera`
   - `하오`, `하오체`, `hao` -> `hao`
   - `해`, `해체`, `hae` -> `hae`
5. `emotion_expressed`는 동의어와 표기 흔들림을 정본 감정 코드로 맞춘다.
   - `공포`, `Fear.`, `fear` -> `fear`
   - `EXPECTATION`, `expectation`, `anticipation` -> `anticipation`
   - 이 밖에도 미리 허용한 감정 코드 집합으로만 수렴한다.
6. Task A의 `dominant_trait`는 TCI 약어와 동의 표기를 정본 값으로 맞춘다.
   - `NS`, `novelty seeking` -> `novelty_seeking`
   - `HA`, `harm avoidance` -> `harm_avoidance`
   - `RD`, `reward dependence` -> `reward_dependence`
   - `P` -> `persistence`
7. Task C의 `speech_ko`, `speech_en`은 바깥 큰따옴표/작은따옴표만 제거할 수 있다.
8. `mimetics` 배열은 공백 정리와 중복 제거만 허용한다.
9. `text_ko`와 `speech_ko`는 기존 검증 규칙과 같은 금칙어 자동 치환을 적용할 수 있다. 이는 의미를 바꾸기 위한 재서술이 아니라, 이미 선언된 치환 사전을 따른 표면 수리다.
10. 출력이 문자열 JSON이면 객체로 파싱해 검사한다. 루트가 객체가 아니거나 파싱이 실패하면 `fatal`이다.
11. 본문 의미를 재해석해 새 문장을 쓰는 보정은 하지 않는다.

## Disposition Policy

### `passed`

- 정규화 전후 의미가 동일하고, 추가 조치 없이 바로 채택 가능한 경우
- 예: Task B에서 요청 감정과 최종 `emotion_expressed`가 다르더라도, 실제 본문과 의성어/의태어가 최종 라벨과 일관되면 통과 가능

### `recoverable`

- 열거형 표기 흔들림, 대소문자, 공백, 구두점, 금칙어 표면 치환처럼 손실 없는 결정적 보정만 필요한 경우
- 정규화된 결과를 저장하되, 의미를 새로 쓰는 재서술은 하지 않는다
- 최종 데이터셋에는 포함 가능하지만, 별도 샘플링으로 QA를 거친다

### `review`

- 자동 보정이 의미를 바꿀 수 있는 경우
- 사람 승인 전에는 최종 데이터셋에 포함하지 않는다
- 승인되면 `approved_review`로 승격해 최종 조립에 포함한다

### `fatal`

- 구조가 깨졌거나, 라벨과 본문이 정면 충돌해 자동 신뢰가 불가능한 경우
- 최종 조립에서는 `unrecoverable_source` 또는 동등한 제외 사유로 빠진다

## Task A TCI Note

Task A의 `dominant_trait`는 v3.1 기준으로 TCI 4축만 정본으로 인정한다.

- `novelty_seeking`
- `harm_avoidance`
- `reward_dependence`
- `persistence`

`NS`, `HA`, `RD`, `P` 같은 약어는 정본 값으로 정규화할 수 있다. 반면 레거시 축 이름인 `conscientiousness`, `openness`, `extraversion` 같은 HEXACO/Big Five 계열 값은 자동 매핑하지 않는다. 이 값들은 비슷해 보여도 TCI 4축과 일대일 대응이 아니므로, 기계적으로 바꾸면 의미 오염이 생긴다. 따라서 Task A에서 레거시 축이 나오면 기본 처리는 `review`다.

## Acceptable Examples

### Task A acceptable

```json
{
  "task": "A",
  "output": {
    "text_ko": "뒤를 살피며 조심조심 걷는, 겁 많지만 빈틈없는 이다.",
    "text_en": "Walks cautiously looking behind, fearful but meticulous.",
    "register": "haera",
    "dominant_trait": "harm_avoidance",
    "temperament_expressed": "melancholic"
  }
}
```

- 이유: `dominant_trait`가 TCI 정본 축이고, 출력 필드가 모두 정본 계약과 맞는다.

### Task B acceptable

```json
{
  "task": "B",
  "emotion_id": "joy",
  "output": {
    "text_ko": "싱글벙글 웃으며 날랜 짐승을 빤히 노려본다. 겁내기보다 먼저 덤빌 때를 재어 본다.",
    "text_en": "Grinning, they stare the swift beast down. Rather than cower, they size up the moment to lunge first.",
    "register": "haera",
    "emotion_expressed": "anticipation",
    "intensity": 0.57,
    "mimetics": ["싱글벙글", "빤히"],
    "temperament_influence": "bold_impulsive_eagerness_overrides_fear"
  }
}
```

- 이유: 요청 감정(`joy`)과 다르더라도, 실제 본문과 의성어/의태어는 `anticipation`에 일관되므로 정본 정책상 채택 가능하다.

### Task C acceptable after normalization

```json
{
  "task": "C",
  "speaker_role": "chief",
  "output": {
    "speech_ko": "당장 앞으로 나서라.",
    "speech_en": "Step forward right now.",
    "register": "해라체",
    "emotion_expressed": "ANGER",
    "speaker_role": "chief",
    "temperament_tone": "choleric_directness"
  }
}
```

- 이유: 본문 의미는 그대로 두고 `register -> haera`, `emotion_expressed -> anger`로 결정적 정규화가 가능하다.

## Unacceptable Examples

### Task A unacceptable for automatic pass

```json
{
  "task": "A",
  "output": {
    "text_ko": "뒤를 살피며 조심조심 걷는, 겁 많지만 빈틈없는 이다.",
    "text_en": "Walks cautiously looking behind, fearful but meticulous.",
    "register": "haera",
    "dominant_trait": "conscientiousness",
    "temperament_expressed": "melancholic"
  }
}
```

- 처리: `review`
- 이유: `conscientiousness`는 레거시 축이며 TCI 정본 축으로 자동 치환할 수 없다.

### Task B unacceptable

```json
{
  "task": "B",
  "emotion_id": "joy",
  "output": {
    "text_ko": "방긋 웃지만 온몸이 벌벌 떨린다. 겁에 질려 뒷걸음질쳤다.",
    "text_en": "Smiles brightly but trembles all over. Terrified, they step backward.",
    "register": "haera",
    "emotion_expressed": "joy",
    "intensity": 0.8,
    "mimetics": ["방긋", "벌벌"],
    "temperament_influence": "surface_smile_masks_fear"
  }
}
```

- 처리: `fatal`
- 이유: 본문이 공포 반응을 직접 묘사하는데 감정 라벨은 `joy`여서 `emotion_text_contradiction`에 해당한다.

### Task C unacceptable

```json
{
  "task": "C",
  "output": "not-json"
}
```

- 처리: `fatal`
- 이유: 구조가 깨져 있어 정규화 이전에 JSON 계약을 만족하지 못한다.

## Concise Post-Generation Flow

1. 생성 직후 raw 결과와 `skipped` 결과를 분리해 스냅샷으로 남긴다.
2. Task A/B/C 행을 정본 정책으로 재분류해 `passed`, `recoverable`, `review`, `fatal` 버킷으로 나눈다.
3. `review`는 사람이 승인한 행만 `approved_review`로 승격한다. `recoverable`은 샘플링 QA 대상으로 별도 확인한다.
4. 최종 조립은 `passed + recovered + approved_review`만 포함하고, `review_not_approved`, `duplicate_content`, `unrecoverable_source` 같은 제외 사유는 별도 기록한다.

이 흐름은 생성 단계의 느슨한 거르기보다 보수적이며, 최종 `train/dev/excluded` 묶음이 어떤 근거로 만들어졌는지 추적 가능하게 유지하는 데 목적이 있다.
