"""VLM-based sprite evaluator using Qwen2.5-VL-3B."""
import json, re, time
from pathlib import Path
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
SPRITE_DIR = Path("output/wildlife_v1_32")
OUTPUT_JSON = Path("output/vlm_evaluation.json")
UPSCALE = 4

QUESTIONS = {
    "count": (
        "How many distinct animals can you see in this pixel art image? "
        "Look carefully — sometimes there are 2 animals close together. "
        "Answer ONLY with a single digit (1, 2, 3, or 4+)."
    ),
    "view": (
        "Look at this pixel art animal carefully. "
        "Is the animal shown in: A) side view (profile, body horizontal), "
        "B) front view (facing the viewer directly), "
        "C) back/rear view (showing the back of the animal), "
        "D) angled view, or E) cannot determine. "
        "Answer ONLY with the single letter A, B, C, D, or E."
    ),
    "species": (
        "What animal is shown in this pixel art? "
        "A) wolf (canine, gray fur, lean), "
        "B) bear (large heavy mammal, brown fur, four legs), "
        "C) boar or wild pig (sturdy body, snout, sometimes tusks), "
        "D) different animal, "
        "E) unclear. "
        "Answer ONLY with the single letter A, B, C, D, or E."
    ),
    "quality": (
        "Rate this pixel art sprite quality from 1 to 10:\n"
        "10 = excellent, very recognizable subject, clean silhouette\n"
        "5 = okay, recognizable but has issues\n"
        "1 = unrecognizable mess\n"
        "Answer ONLY with a single digit 1-10."
    ),
}

print(f"Loading {MODEL_NAME}...")
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, device_map="cpu",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"Loaded in {time.time()-t0:.1f}s")


def ask_vlm(img, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=1.0)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def parse_count(r):
    m = re.search(r"\d+", r)
    return min(int(m.group()), 4) if m else -1

def parse_letter(r, valid):
    m = re.search(rf"[{valid}]", r.upper())
    return m.group() if m else "?"

def parse_quality(r):
    m = re.search(r"\b([1-9]|10)\b", r)
    return int(m.group()) if m else -1


def evaluate_sprite(path, expected_species):
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    img = bg.resize((img.width * UPSCALE, img.height * UPSCALE), Image.NEAREST)

    res = {"file": path.name, "expected_species": expected_species}
    for key, q in QUESTIONS.items():
        try:
            t0 = time.time()
            raw = ask_vlm(img, q)
            res[f"{key}_raw"] = raw
            res[f"{key}_time"] = round(time.time() - t0, 2)
            if key == "count":   res["count"]   = parse_count(raw)
            elif key == "view":  res["view"]    = parse_letter(raw, "ABCDE")
            elif key == "species": res["species"] = parse_letter(raw, "ABCDE")
            elif key == "quality": res["quality"] = parse_quality(raw)
        except Exception as e:
            res[f"{key}_error"] = str(e)
    return res


def score_result(r, expected_species):
    score, issues = 100, []
    count = r.get("count", -1)
    if count > 1:   score -= 50; issues.append(f"다수 동물 ({count}개)")
    elif count < 0: score -= 10; issues.append("count 파싱 실패")

    view = r.get("view", "?")
    if   view == "B": score -= 30; issues.append("정면 뷰")
    elif view == "C": score -= 30; issues.append("후면 뷰")
    elif view == "D": score -= 15; issues.append("각도 뷰")
    elif view == "E": score -= 20; issues.append("뷰 불분명")
    elif view == "?": score -= 10; issues.append("view 파싱 실패")

    sp = r.get("species", "?")
    exp_letter = {"wolf": "A", "bear": "B", "boar": "C"}.get(expected_species, "?")
    if sp == "E":             score -= 25; issues.append(f"{expected_species} 인식 불가")
    elif sp == "D":           score -= 35; issues.append("다른 동물로 인식")
    elif sp in "ABC" and sp != exp_letter:
        wrong = {"A":"wolf","B":"bear","C":"boar"}[sp]
        score -= 40; issues.append(f"{wrong}로 잘못 인식")
    elif sp == "?":           score -= 15; issues.append("species 파싱 실패")

    q = r.get("quality", -1)
    if 5 <= q < 7:   score -= 5;  issues.append(f"품질 보통 ({q}/10)")
    elif 3 <= q < 5: score -= 15; issues.append(f"품질 낮음 ({q}/10)")
    elif 0 < q < 3:  score -= 30; issues.append(f"품질 매우 낮음 ({q}/10)")

    return max(0, score), issues


# === 실행 ===
all_files = [(sp, f)
             for sp in ["wolf","bear","boar"]
             for f in sorted((SPRITE_DIR / sp).glob(f"{sp}_*.png"))]

print(f"\n=== 평가 시작 ({len(all_files)} sprites) ===\n")
t_start = time.time()
all_results = {}

for i, (species, f) in enumerate(all_files):
    t0 = time.time()
    ev = evaluate_sprite(f, species)
    score, issues = score_result(ev, species)
    ev.update({"score": score, "issues": issues})
    all_results[f"{species}/{f.name}"] = ev

    elapsed = time.time() - t0
    eta = elapsed * (len(all_files) - i - 1)
    print(f"[{i+1:2d}/{len(all_files)}] {f.name}: {score:3d}pt "
          f"count={ev.get('count')} view={ev.get('view')} "
          f"sp={ev.get('species')} q={ev.get('quality')} "
          f"| {elapsed:.1f}s ETA {eta/60:.1f}min")
    if issues:
        print(f"           ⚠ {'; '.join(issues)}")

total = time.time() - t_start
print(f"\n완료: {total/60:.1f}분")

OUTPUT_JSON.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
print(f"저장: {OUTPUT_JSON}")

for sp in ["wolf","bear","boar"]:
    rows = sorted([(k.split("/")[1], v) for k,v in all_results.items()
                   if k.startswith(f"{sp}/")], key=lambda x:-x[1]["score"])
    print(f"\n{sp.upper()} Top5:")
    for fn, r in rows[:5]:
        issues = "; ".join(r.get("issues",[])) or "OK"
        print(f"  {fn}: {r['score']}pt — {issues}")
