"""
Speakers and paragraphs of a lesson against its edited transcript (TASK-speakers-paragraphs.md, step 1.4).

The edited transcript (kabbalahmedia .doc -> text, a paragraph per line) marks who speaks by the line:
"תלמיד:" / "שאלה:" -> תלמיד; "קריין:" -> קריין, and so are the quoted source lines read after it;
any other line -> רב. Its words are aligned to the model's (difflib on normalized words); on the aligned words:
  - speakers: the share of words of each human label that got the same label from the model, the confusion,
    speaker changes missed / extra;
  - paragraphs (inside the Rav's speech on both sides, a speaker change not counted): exact P/R/F1 of paragraph
    starts on the same word, tol1 -- within one sentence (as mdb-lora's eval.py).
Run: python -m ai_trns.paragraphs.evaluate <lesson.doc.txt> <job result ?format=paragraphs> [...pairs]
"""
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher

from .build import norm

RAV, READER, STUDENT = "רב", "קריין", "תלמיד"
PREFIXES = {"תלמיד": STUDENT, "שאלה": STUDENT, "קריין": READER}
_PREFIX_RE = re.compile(r"^\s*(" + "|".join(PREFIXES) + r")\s*:\s*")


def human_words(text: str) -> list[dict]:
    """[{"key", "label", "para"}]: the transcript's words with the speaker of their line and the line number"""
    out, reading, para = [], False, 0
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        m = _PREFIX_RE.match(line)
        if m:
            label = PREFIXES[m.group(1)]
            reading = label == READER
            line = line[m.end():]
        elif reading and line.lstrip().startswith(('"', "״", "“")):
            label = READER
        else:
            label, reading = RAV, False
        keys = [k for k in (norm(w) for w in line.split()) if k]
        if keys:
            out += [{"key": k, "label": label, "para": para} for k in keys]
            para += 1
    return out


def model_words(result: dict) -> list[dict]:
    """[{"key", "label", "para", "sentence"}] from a ?format=paragraphs result"""
    out, sentence = [], 0
    for n, p in enumerate(result["paragraphs"]):
        for w in p["words"]:
            key = norm(w["word"])
            if key:
                out.append({"key": key, "label": p["label"] or STUDENT, "para": n, "sentence": sentence})
            if re.search(r"[.!?…:]['\"”’»)]*$", w["word"].strip()):
                sentence += 1
    return out


def align(human: list[dict], model: list[dict]) -> list[tuple[int, int]]:
    matcher = SequenceMatcher(None, [w["key"] for w in human], [w["key"] for w in model], autojunk=False)
    return [(a + k, b + k) for a, b, size in matcher.get_matching_blocks() for k in range(size)]


def _prf(tp: int, pred: int, gold: int) -> dict:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "pred": pred, "gold": gold}


def evaluate(text: str, result: dict) -> dict:
    human, model = human_words(text), model_words(result)
    pairs = align(human, model)
    confusion = Counter((human[a]["label"], model[b]["label"]) for a, b in pairs)
    per_label = {}
    for label in (RAV, READER, STUDENT):
        total = sum(v for (h, _), v in confusion.items() if h == label)
        per_label[label] = {"words": total, "right": round(confusion[(label, label)] / total, 3) if total else None}
    changes = {"missed": 0, "extra": 0, "both": 0}
    gold, pred, tp_exact, tp_tol = 0, 0, 0, 0
    gold_sentences, pred_sentences = [], []
    for (a0, b0), (a1, b1) in zip(pairs, pairs[1:]):
        h_change = human[a1]["label"] != human[a0]["label"]
        m_change = model[b1]["label"] != model[b0]["label"]
        if h_change and m_change:
            changes["both"] += 1
        elif h_change:
            changes["missed"] += 1
        elif m_change:
            changes["extra"] += 1
        # paragraphs inside the Rav's speech on both sides
        if h_change or m_change or not (human[a1]["label"] == model[b1]["label"] == RAV):
            continue
        h_start = human[a1]["para"] != human[a0]["para"]
        m_start = model[b1]["para"] != model[b0]["para"]
        gold += h_start
        pred += m_start
        tp_exact += h_start and m_start
        if h_start:
            gold_sentences.append(model[b1]["sentence"])
        if m_start:
            pred_sentences.append(model[b1]["sentence"])
    used = set()
    for s in pred_sentences:
        hit = next((g for g in gold_sentences if abs(g - s) <= 1 and g not in used), None)
        if hit is not None:
            used.add(hit)
            tp_tol += 1
    return {
        "human_words": len(human), "model_words": len(model), "aligned": len(pairs),
        "speakers": per_label,
        "confusion": {f"{h}->{m}": v for (h, m), v in sorted(confusion.items())},
        "speaker_changes": changes,
        "paragraphs_exact": _prf(tp_exact, pred, gold),
        "paragraphs_tol1": _prf(tp_tol, len(pred_sentences), len(gold_sentences)),
    }


def main(argv: list[str]) -> None:
    if len(argv) < 2 or len(argv) % 2:
        raise SystemExit(__doc__)
    for doc_path, result_path in zip(argv[::2], argv[1::2]):
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
        with open(result_path, encoding="utf-8") as f:
            result = json.load(f)
        print(doc_path, json.dumps(evaluate(text, result), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
