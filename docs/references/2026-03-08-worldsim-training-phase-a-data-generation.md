# WorldSim Training Data Generation — Phase A

> **레포**: `worldsim-training` (별도 레포)
> **목적**: 특화형 파인튜닝용 5,500개 학습 데이터 생성
> **API**: OpenRouter (Claude Sonnet 권장)
> **비용**: ~$22 (5,500 예시 × 평균 500토큰)

---

## 1. 레포 초기 구조

```bash
mkdir worldsim-training && cd worldsim-training
git init

mkdir -p config scripts data/{raw,validated,final} prompts
touch .env README.md requirements.txt
```

### requirements.txt
```
openai>=1.0.0        # OpenRouter는 OpenAI SDK 호환
pyyaml>=6.0
tqdm>=4.65
jsonlines>=4.0
```

### .env
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514
```

---

## 2. Config 파일들

### config/situations.yaml
```yaml
# 10개 핵심 상황 — 석기시대 서사에서 가장 빈번한 이벤트
situations:
  - id: predator
    ko: 짐승발견
    desc: "날랜 짐승이 무리 근처에 나타났다"
    typical_actions: ["도망", "숨기", "맞서기", "경고"]

  - id: food_found
    ko: 먹거리발견
    desc: "먹을 것을 찾았다"
    typical_actions: ["채집", "나누기", "숨기기", "알리기"]

  - id: storm
    ko: 비바람
    desc: "거센 비바람이 몰아치고 있다"
    typical_actions: ["피하기", "움막보강", "불지키기", "견디기"]

  - id: injury
    ko: 부상
    desc: "누군가 다쳤다"
    typical_actions: ["돌보기", "풀바르기", "놔두기", "걱정하기"]

  - id: kin_death
    ko: 친족사망
    desc: "가까운 이가 숨을 거두었다"
    typical_actions: ["슬퍼하기", "묻어주기", "분노하기", "받아들이기"]

  - id: theft
    ko: 도둑적발
    desc: "누군가 먹거리를 훔친 것이 드러났다"
    typical_actions: ["따지기", "용서하기", "쫓아내기", "되찾기"]

  - id: migration
    ko: 무리이동
    desc: "무리가 새로운 곳으로 옮겨가야 한다"
    typical_actions: ["이끌기", "따르기", "거부하기", "정찰하기"]

  - id: tool_craft
    ko: 도구제작
    desc: "새로운 도구를 만들고 있다"
    typical_actions: ["깎기", "다듬기", "시험하기", "가르치기"]

  - id: fire
    ko: 불발견
    desc: "불을 발견하거나 다루고 있다"
    typical_actions: ["지키기", "옮기기", "두려워하기", "놀라기"]

  - id: strangers
    ko: 낯선무리
    desc: "낯선 무리가 다가오고 있다"
    typical_actions: ["경계하기", "맞이하기", "숨기", "위협하기"]
```

### config/personalities.yaml
```yaml
# 10개 성격군 — Gemini 아키타입 기반, HEXACO 매핑
personalities:
  - id: cautious_elder
    ko: 신중한원로
    keywords: ["겁많음", "꼼꼼함", "조용함", "규율중시"]
    hexaco: {H: 0.8, E: 0.8, X: 0.3, A: 0.6, C: 0.8, O: 0.3}
    default_register: hao  # 하오체
    desc: "위험을 경계하고 규율을 중시하는 어른"

  - id: reckless_hunter
    ko: 무모한사냥꾼
    keywords: ["대담함", "충동적", "거침없음", "위계무시"]
    hexaco: {H: 0.3, E: 0.1, X: 0.9, A: 0.3, C: 0.2, O: 0.7}
    default_register: hae  # 해체
    desc: "두려움 없이 위험에 뛰어드는 사냥꾼"

  - id: visionary_shaman
    ko: 선지적주술사
    keywords: ["신비로움", "탐구적", "은유적", "직관적"]
    hexaco: {H: 0.6, E: 0.7, X: 0.5, A: 0.5, C: 0.4, O: 0.9}
    default_register: hao
    desc: "자연의 징조를 읽고 해석하는 주술사"

  - id: vengeful_warrior
    ko: 복수심전사
    keywords: ["냉혹함", "복수심", "공격적", "비타협적"]
    hexaco: {H: 0.2, E: 0.4, X: 0.8, A: 0.1, C: 0.5, O: 0.3}
    default_register: haera  # 해라체
    desc: "원한을 잊지 않는 전사"

  - id: empathetic_healer
    ko: 공감하는치유자
    keywords: ["따뜻함", "다정함", "걱정많음", "이타적"]
    hexaco: {H: 0.8, E: 0.8, X: 0.6, A: 0.9, C: 0.7, O: 0.6}
    default_register: haera
    desc: "다친 이를 돌보는 따뜻한 치유자"

  - id: greedy_gatherer
    ko: 탐욕스러운채집가
    keywords: ["교활함", "탐욕스러움", "계산적", "자기중심"]
    hexaco: {H: 0.1, E: 0.5, X: 0.6, A: 0.4, C: 0.8, O: 0.2}
    default_register: hao
    desc: "자기 이득을 최우선으로 챙기는 채집가"

  - id: diligent_craftsman
    ko: 집요한제작자
    keywords: ["꼼꼼함", "집요함", "과묵함", "완벽주의"]
    hexaco: {H: 0.7, E: 0.4, X: 0.2, A: 0.5, C: 0.9, O: 0.8}
    default_register: haera
    desc: "도구 만들기에 몰두하는 장인"

  - id: charismatic_chief
    ko: 카리스마족장
    keywords: ["당당함", "설득력", "통솔력", "결단력"]
    hexaco: {H: 0.5, E: 0.3, X: 0.9, A: 0.7, C: 0.7, O: 0.5}
    default_register: hao
    desc: "무리를 이끄는 카리스마 있는 우두머리"

  - id: paranoid_scout
    ko: 편집증정찰병
    keywords: ["불안함", "경계심", "예민함", "의심많음"]
    hexaco: {H: 0.5, E: 0.9, X: 0.3, A: 0.4, C: 0.8, O: 0.5}
    default_register: hae
    desc: "모든 것을 위협으로 보는 정찰병"

  - id: stoic_observer
    ko: 냉소적방관자
    keywords: ["무감정", "초연함", "관찰자", "독립적"]
    hexaco: {H: 0.7, E: 0.1, X: 0.1, A: 0.5, C: 0.6, O: 0.6}
    default_register: haera
    desc: "감정 없이 상황을 지켜보는 방관자"
```

### config/emotions.yaml
```yaml
# Plutchik 8 기본 감정
emotions:
  - id: joy
    ko: 기쁨
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["방긋", "덩실덩실", "싱글벙글"]

  - id: sadness
    ko: 슬픔
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["그렁그렁", "글썽글썽", "훌쩍"]

  - id: fear
    ko: 두려움
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["오들오들", "덜컥덜컥", "벌벌", "후들후들"]

  - id: anger
    ko: 분노
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["부들부들", "벌컥", "으드득"]

  - id: trust
    ko: 믿음
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["든든히", "묵묵히"]

  - id: disgust
    ko: 역겨움
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["으윽", "칙칙"]

  - id: surprise
    ko: 놀람
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["깜짝", "펄쩍", "화들짝"]

  - id: anticipation
    ko: 기대
    intensities: [0.3, 0.6, 0.9]
    mimetics: ["두근두근", "설레설레"]
```

---

## 3. 데이터 생성 스크립트

### scripts/generate_data.py

```python
#!/usr/bin/env python3
"""
WorldSim Training Data Generator — Phase A
OpenRouter API로 특화형 파인튜닝 데이터 5,500개 생성
"""

import os
import json
import yaml
import time
import random
import hashlib
import jsonlines
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-20250514")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS_PER_COMBO = 3  # 조합당 변형 수
SEED = 42
random.seed(SEED)

# ── Load configs ──
def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)

situations = load_yaml("config/situations.yaml")["situations"]
personalities = load_yaml("config/personalities.yaml")["personalities"]
emotions = load_yaml("config/emotions.yaml")["emotions"]

# ── OpenRouter client ──
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)

# ── Register instructions ──
REGISTER_INSTRUCTIONS = {
    "haera": "해라체로 써라. 문장을 -다, -는다, -았다, -었다 로 끝내라.",
    "hao": "하오체로 써라. 문장을 -오, -소, -시오 로 끝내라.",
    "hae": "해체로 써라. 문장을 -해, -야, -지, -어 로 끝내라.",
}

# ── System prompt (Teacher에게 주는 지시) ──
SYSTEM_PROMPT = """너는 석기시대 생존 시뮬레이션 게임의 학습 데이터를 만드는 전문가다.

[엄격한 규칙]
1. 반드시 순우리말만 사용하라. 한자어(漢字語)와 현대 외래어는 절대 금지.
   금지 단어: 식량, 마을, 전투, 자연, 건축, 국가, 사회, 족장, 구출, 기상, 맹수, 동맹, 생존, 결정, 공격, 방어, 이동, 제작, 수집, 교환, 상황, 감정, 성격, 인물
   허용 단어: 먹거리, 무리, 피 흘리는 싸움, 온 누리, 집짓기, 우두머리, 빼내오기, 날씨, 날랜 짐승, 손잡기, 살아남기, 마음먹기, 덤비기, 막아서기, 옮겨가기, 만들기, 모으기, 바꾸기
2. 의성어/의태어를 적극 사용하라 (오들오들, 덜컥, 방긋, 부들부들 등).
3. 지정된 길이를 정확히 지켜라.
4. JSON, 마크다운, 영어를 출력하지 마라. 순수 한국어 텍스트만.
5. 매번 다른 표현을 써라. 같은 문장 구조를 반복하지 마라."""


def generate_task_a(personality, variant_idx):
    """Task A: 성격 한줄 묘사 (1문장, 20-40자)"""
    prompt = f"""[과제] 아래 성격의 석기시대 인물을 순우리말 1문장(20~40글자)으로 묘사하라.

[성격]
이름: {personality['ko']}
특징: {', '.join(personality['keywords'])}
설명: {personality['desc']}

[어투] {REGISTER_INSTRUCTIONS['haera']}
[변형번호] {variant_idx} (이전과 다른 표현을 써라)

1문장만 출력하라. 다른 것은 쓰지 마라."""

    return {
        "task": "A",
        "personality_id": personality["id"],
        "variant": variant_idx,
        "prompt": prompt,
        "expected_len": "20-40자",
        "expected_sentences": 1,
        "register": "haera",
    }


def generate_task_b(personality, emotion, situation, variant_idx):
    """Task B: 상황 반응 묘사 (2문장, 30-60자)"""
    intensity = random.choice(emotion["intensities"])
    mimetic = random.choice(emotion["mimetics"]) if emotion["mimetics"] else ""

    prompt = f"""[과제] 아래 인물이 아래 상황에서 보이는 반응을 순우리말 2문장(30~60글자)으로 묘사하라.

[인물]
성격: {', '.join(personality['keywords'])}

[느낌] {emotion['ko']} (세기: {'강함' if intensity > 0.7 else '보통' if intensity > 0.4 else '약함'})
참고 의성어: {mimetic}

[벌어진 일] {situation['desc']}

[어투] {REGISTER_INSTRUCTIONS['haera']}
[변형번호] {variant_idx}

정확히 2문장만 출력하라."""

    return {
        "task": "B",
        "personality_id": personality["id"],
        "emotion_id": emotion["id"],
        "emotion_intensity": intensity,
        "situation_id": situation["id"],
        "variant": variant_idx,
        "prompt": prompt,
        "expected_len": "30-60자",
        "expected_sentences": 2,
        "register": "haera",
    }


def generate_task_c(personality, emotion, situation, variant_idx):
    """Task C: 짧은 대사 (1문장 직접화법, 15-30자)"""
    register = personality["default_register"]

    prompt = f"""[과제] 아래 인물이 아래 상황에서 하는 말을 순우리말 1문장 대사(15~30글자)로 써라.

[인물]
성격: {', '.join(personality['keywords'])}
역할이 쓰는 말투: {REGISTER_INSTRUCTIONS[register]}

[느낌] {emotion['ko']}
[벌어진 일] {situation['desc']}

[변형번호] {variant_idx}

큰따옴표로 감싼 대사 1문장만 출력하라. 예: "어서 피하시오!"
다른 설명은 쓰지 마라."""

    return {
        "task": "C",
        "personality_id": personality["id"],
        "emotion_id": emotion["id"],
        "situation_id": situation["id"],
        "register": register,
        "variant": variant_idx,
        "prompt": prompt,
        "expected_len": "15-30자",
        "expected_sentences": 1,
    }


def generate_task_d(situation, variant_idx):
    """Task D: 알림 카피 (1문장, 10-25자)"""
    names = ["돌이", "바위", "새벽", "이슬", "구름", "달빛", "불꽃", "물결", "바람", "씨앗"]
    name = names[variant_idx % len(names)]

    prompt = f"""[과제] 아래 상황이 일어났다는 알림을 순우리말 1문장(10~25글자)으로 써라.

[누가] {name}
[벌어진 일] {situation['desc']}

[어투] 해라체 (-다, -았다, -었다 로 끝)
[변형번호] {variant_idx}

1문장만 출력하라. 이름을 반드시 포함하라."""

    return {
        "task": "D",
        "name": name,
        "situation_id": situation["id"],
        "variant": variant_idx,
        "prompt": prompt,
        "expected_len": "10-25자",
        "expected_sentences": 1,
        "register": "haera",
    }


def call_api(system_prompt, user_prompt, max_retries=3):
    """OpenRouter API 호출 (재시도 포함)"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=128,
                temperature=0.8,
                extra_headers={
                    "HTTP-Referer": "https://github.com/hyunlord/worldsim-training",
                    "X-Title": "WorldSim Training Data Generation",
                },
            )
            text = response.choices[0].message.content.strip()
            usage = response.usage
            return text, usage.total_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after {max_retries} attempts: {e}")
                return None, 0
    return None, 0


def generate_all():
    """전체 데이터 생성"""
    jobs = []

    # Task A: 128 아키타입 × 5 변형 = 640
    # (10개 성격군 × 5 변형 = 50으로 시작, 나중에 128로 확장)
    print("Building Task A jobs...")
    for pers in personalities:
        for v in range(5):
            jobs.append(generate_task_a(pers, v))

    # Task B: 10상황 × 8감정 × 10성격 = 800, × 3변형 = 2400
    print("Building Task B jobs...")
    for sit in situations:
        for emo in emotions:
            for pers in personalities:
                for v in range(VARIANTS_PER_COMBO):
                    jobs.append(generate_task_b(pers, emo, sit, v))

    # Task C: 10상황 × 10성격 × 3변형 = 300 (감정은 랜덤)
    print("Building Task C jobs...")
    for sit in situations:
        for pers in personalities:
            for v in range(VARIANTS_PER_COMBO):
                emo = random.choice(emotions)
                jobs.append(generate_task_c(pers, emo, sit, v))

    # Task D: 20상황(10×2이름) × 5변형 = 100
    print("Building Task D jobs...")
    for sit in situations:
        for v in range(5):
            jobs.append(generate_task_d(sit, v))

    random.shuffle(jobs)

    print(f"\nTotal jobs: {len(jobs)}")
    print(f"  Task A: {sum(1 for j in jobs if j['task'] == 'A')}")
    print(f"  Task B: {sum(1 for j in jobs if j['task'] == 'B')}")
    print(f"  Task C: {sum(1 for j in jobs if j['task'] == 'C')}")
    print(f"  Task D: {sum(1 for j in jobs if j['task'] == 'D')}")

    # 생성 실행
    total_tokens = 0
    results = []
    output_file = OUTPUT_DIR / f"generated_{int(time.time())}.jsonl"

    with jsonlines.open(output_file, mode='w') as writer:
        for job in tqdm(jobs, desc="Generating"):
            text, tokens = call_api(SYSTEM_PROMPT, job["prompt"])
            total_tokens += tokens

            if text:
                result = {
                    **job,
                    "output": text,
                    "tokens_used": tokens,
                    "model": OPENROUTER_MODEL,
                    "timestamp": time.time(),
                }
                # prompt 필드는 학습에 불필요하므로 제거 가능
                del result["prompt"]
                writer.write(result)
                results.append(result)

            # Rate limit 방지
            time.sleep(0.1)

    # 요약
    cost_estimate = total_tokens * 0.000015  # ~$15/M tokens (Sonnet 평균)
    print(f"\n=== Generation Complete ===")
    print(f"Output: {output_file}")
    print(f"Total examples: {len(results)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Estimated cost: ${cost_estimate:.2f}")
    print(f"\nTask distribution:")
    for task in ["A", "B", "C", "D"]:
        count = sum(1 for r in results if r["task"] == task)
        print(f"  Task {task}: {count}")


if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY in .env file")
        exit(1)
    generate_all()
```

---

## 4. QA 검증 스크립트

### scripts/validate_data.py

```python
#!/usr/bin/env python3
"""
WorldSim Training Data Validator — 자동 QA
"""

import json
import re
import jsonlines
from pathlib import Path
from collections import Counter

# 금지 한자어 목록
FORBIDDEN_WORDS = [
    "식량", "마을", "전투", "자연", "건축", "국가", "사회", "족장",
    "구출", "기상", "맹수", "동맹", "생존", "결정", "공격", "방어",
    "이동", "제작", "수집", "교환", "상황", "감정", "성격", "인물",
    "환경", "위험", "동물", "식물", "무기", "도구", "기술", "문화",
]

# 어투별 종결어미 패턴
REGISTER_ENDINGS = {
    "haera": [r"다[.\s]?$", r"는다[.\s]?$", r"았다[.\s]?$", r"었다[.\s]?$", r"였다[.\s]?$", r"한다[.\s]?$"],
    "hao": [r"오[.\s!]?$", r"소[.\s!]?$", r"시오[.\s!]?$", r"구려[.\s!]?$"],
    "hae": [r"해[.\s!]?$", r"야[.\s!]?$", r"지[.\s!]?$", r"어[.\s!]?$", r"았어[.\s!]?$"],
}

# 메타 발화 패턴
META_PATTERNS = [
    r"AI", r"모델", r"프롬프트", r"학습", r"데이터", r"시뮬레이션",
    r"석기시대 생존 시뮬레이션", r"월드심", r"WorldSim",
    r"인공지능", r"언어 모델",
]

# Task별 길이 제한
TASK_LIMITS = {
    "A": {"min_chars": 15, "max_chars": 50, "sentences": 1},
    "B": {"min_chars": 25, "max_chars": 80, "sentences": 2},
    "C": {"min_chars": 10, "max_chars": 40, "sentences": 1},
    "D": {"min_chars": 8, "max_chars": 30, "sentences": 1},
}


def count_sentences(text):
    """대략적인 문장 수 (마침표/느낌표/물음표 기준)"""
    return len(re.split(r'[.!?]+', text.strip())) - (0 if text.strip()[-1] in '.!?' else 0)


def check_forbidden(text):
    """금지 한자어 검사"""
    found = []
    for word in FORBIDDEN_WORDS:
        if word in text:
            found.append(word)
    return found


def check_register(text, expected_register):
    """어투 종결어미 검사"""
    if expected_register not in REGISTER_ENDINGS:
        return True  # 알 수 없는 어투는 통과
    patterns = REGISTER_ENDINGS[expected_register]
    # 마지막 문장의 마지막 부분 체크
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences:
        return False
    last = sentences[-1]
    return any(re.search(p, last) for p in patterns)


def check_meta(text):
    """메타 발화 검사"""
    found = []
    for pattern in META_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(pattern)
    return found


def check_repetition(text, threshold=0.5):
    """반복 패턴 검사"""
    words = text.split()
    if len(words) < 4:
        return False
    # 연속 반복 체크
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    # bigram 반복 비율
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    if bigrams:
        counter = Counter(bigrams)
        most_common_ratio = counter.most_common(1)[0][1] / len(bigrams)
        if most_common_ratio > threshold:
            return True
    return False


def validate_file(input_path):
    """전체 파일 검증"""
    results = {"pass": [], "fail": []}
    stats = Counter()

    with jsonlines.open(input_path) as reader:
        for item in reader:
            text = item.get("output", "")
            task = item.get("task", "?")
            register = item.get("register", "haera")
            violations = []

            # 1. 길이 체크
            limits = TASK_LIMITS.get(task, {"min_chars": 5, "max_chars": 100})
            char_count = len(text)
            if char_count < limits["min_chars"]:
                violations.append(f"too_short({char_count}<{limits['min_chars']})")
            if char_count > limits["max_chars"] * 1.5:  # 50% 여유
                violations.append(f"too_long({char_count}>{limits['max_chars']})")

            # 2. 금지어 체크
            forbidden = check_forbidden(text)
            if forbidden:
                violations.append(f"forbidden({','.join(forbidden)})")

            # 3. 어투 체크
            if not check_register(text, register):
                violations.append(f"register_mismatch(expected={register})")

            # 4. 메타 발화 체크
            meta = check_meta(text)
            if meta:
                violations.append(f"meta({','.join(meta)})")

            # 5. 반복 체크
            if check_repetition(text):
                violations.append("repetition")

            # 6. 빈 출력 체크
            if not text or len(text.strip()) < 3:
                violations.append("empty")

            # 7. JSON/영어 혼입 체크
            if "{" in text or "}" in text or "```" in text:
                violations.append("json_leak")
            if re.search(r'[a-zA-Z]{3,}', text):
                violations.append("english_leak")

            # 판정
            item["violations"] = violations
            if violations:
                results["fail"].append(item)
                for v in violations:
                    stats[v.split("(")[0]] += 1
            else:
                results["pass"].append(item)

    # 결과 저장
    output_dir = Path("data/validated")
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_dir / "passed.jsonl", mode='w') as w:
        for item in results["pass"]:
            w.write(item)

    with jsonlines.open(output_dir / "failed.jsonl", mode='w') as w:
        for item in results["fail"]:
            w.write(item)

    # 리포트
    total = len(results["pass"]) + len(results["fail"])
    pass_rate = len(results["pass"]) / total * 100 if total > 0 else 0

    print(f"\n=== Validation Report ===")
    print(f"Total: {total}")
    print(f"Passed: {len(results['pass'])} ({pass_rate:.1f}%)")
    print(f"Failed: {len(results['fail'])} ({100-pass_rate:.1f}%)")
    print(f"\nViolation breakdown:")
    for violation, count in stats.most_common():
        print(f"  {violation}: {count}")
    print(f"\nOutput: data/validated/passed.jsonl")

    return pass_rate


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # 가장 최근 raw 파일 자동 선택
        raw_files = sorted(Path("data/raw").glob("*.jsonl"))
        if not raw_files:
            print("No raw data files found. Run generate_data.py first.")
            exit(1)
        input_path = raw_files[-1]
    else:
        input_path = sys.argv[1]

    print(f"Validating: {input_path}")
    validate_file(input_path)
```

---

## 5. 실행 순서

```bash
# 1. 레포 세팅
cd worldsim-training
pip install -r requirements.txt
cp .env.example .env
# .env에 OPENROUTER_API_KEY 입력

# 2. 데이터 생성 (~수 시간, ~$22)
python scripts/generate_data.py

# 3. QA 검증
python scripts/validate_data.py

# 4. 통과율 확인
# 목표: 80% 이상 pass
# fail 원인 확인 후 config/프롬프트 조정 → 재생성
```

---

*WorldSim Training Data Generation — Phase A — 2026-03-08*
