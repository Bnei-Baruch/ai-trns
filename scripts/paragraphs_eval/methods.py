"""
Paragraphs of the Rav on ASR text: methods on one fixed lesson set (TASK-paragraphs.md). CPU, stdlib only.

Per lesson <id> (test.txt / tune.txt), in --dir:
  out/<id>.json   the diarize document (ai-trns transcript -> diarize, prod voice samples), no paragraphs
  docs/<id>.txt   the edited transcript (the reference: paragraphs and who speaks)
  preds/<adapter>.m<merge>.jsonl   adapter answers for the Rav text: {"id", "sentences", "breaks"} (predict_set.py)
  emb/<id>.m<merge>.json          bge-m3 embeddings of the Rav text's sentences (embed_set.py)
  sources/<id>.txt  (--sources) the lesson's source text (fetch_sources.py): the editor's "Rav" passages that read
                  it are the reader's (evaluate.source_passages) -- the reference no longer counts the reading as Rav
  exclude.json    (--exclude) {"<id>": "all" | [[start_s, end_s], ...]}: lessons / spans left out of every score
                  (the Hebrew track is a simultaneous translation there: TASK-speakers-paragraphs.md)
Merge (--merges): N -- diarize speakers, others' turns of <= N words between two Rav turns -> Rav; "o" -- the editor's
speakers (diagnosis); "s" -- the editor's speakers with the source reading as the reader's (needs --sources).
Every method gives paragraph starts (word indices) inside the Rav's turns; the score is
ai_trns.paragraphs.evaluate (exact: the same word as the editor's paragraph start, tol1: within one sentence;
only where both the editor and the model have the Rav, speaker changes not counted), summed over lessons.

  python3 scripts/paragraphs_eval/methods.py --dir D --set tune grid      # tune thresholds (tune set only)
  python3 scripts/paragraphs_eval/methods.py --dir D --set test table --params best.json
  python3 scripts/paragraphs_eval/methods.py --dir D --set test --sources --exclude D/exclude.json speakers
"""
import argparse
import bisect
import copy
import itertools
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from ai_trns.paragraphs import build, evaluate  # noqa: E402

# mdb-lora adapters/paragraphing/segment.py SENTENCE_END_RE: the sentences the adapter numbers
SENTENCE_END_RE = re.compile(r"(?:[.!?…:]+[\"'”’»)\]*]*|\*|\d\))(?= )")
RAV = build.MODEL_SPEAKER


def read_set(path):
    ids = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


def merge_short(doc, max_words):
    """Turns of others of <= max_words words between two turns of the Rav -> the Rav's (diarize noise, back-channel)"""
    if not max_words:
        return doc
    doc = copy.deepcopy(doc)
    words = doc["words"]
    turns = build.speaker_turns(words)
    who = [build.turn_speaker(words, t) for t in turns]
    for k in range(1, len(turns) - 1):
        if who[k] != RAV and who[k - 1] == RAV and who[k + 1] == RAV and len(turns[k]) <= max_words:
            for i in turns[k]:
                words[i]["speaker"] = RAV
    return doc


def oracle_speakers(doc, text, source=None):
    """Diagnosis only: every word gets the editor's speaker (aligned words; the others -- the one before them),
    so the paragraphs are measured without diarize errors; with the source, its reading is the reader's"""
    doc = copy.deepcopy(doc)
    words = doc["words"]
    human = evaluate.human_words(text, source)
    keyed = [i for i, w in enumerate(words) if build.norm(w["word"])]
    names = {evaluate.RAV: RAV, evaluate.READER: build.READER, evaluate.STUDENT: "SPEAKER_99"}
    label = {}
    for a, b in evaluate.align(human, [{"key": build.norm(words[i]["word"])} for i in keyed]):
        label[keyed[b]] = names[human[a]["label"]]
    prev = RAV
    for i, w in enumerate(words):
        prev = label.get(i, prev)
        w["speaker"] = prev
    return doc


def read_source(d, lid):
    path = os.path.join(d, "sources", lid + ".txt")
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


class Lesson:
    def __init__(self, d, lid, merge, sources=False, exclude=()):
        self.id, self.merge = lid, merge
        self.text = open(os.path.join(d, "docs", lid + ".txt"), encoding="utf-8").read()
        self.source = read_source(d, lid) if sources else None
        doc = json.load(open(os.path.join(d, "out", lid + ".json"), encoding="utf-8"))
        if merge in ("o", "s"):
            self.doc = oracle_speakers(doc, self.text, read_source(d, lid) if merge == "s" else None)
        else:
            self.doc = merge_short(doc, int(merge))
        self.words = self.doc["words"]
        self.turns = build.speaker_turns(self.words)
        self.rav_turns = [t for t in self.turns if build.turn_speaker(self.words, t) == RAV]
        self.flat, self.index, self.offsets = build.model_text(self.words, self.turns)
        self.sentences = []
        start = 0
        for m in SENTENCE_END_RE.finditer(self.flat):
            self.sentences.append([start, m.end()])
            start = m.end() + 1
        if start < len(self.flat):
            self.sentences.append([start, len(self.flat)])
        # word index where each sentence starts
        self.sent_word = [self.index[min(bisect.bisect_left(self.offsets, s), len(self.index) - 1)]
                          for s, _ in self.sentences] if self.index else []
        self.sent_starts = set(self.sent_word)
        self.turn_of = {}
        for n, t in enumerate(self.rav_turns):
            for i in t:
                self.turn_of[i] = n
        base = build.paragraphs(self.doc)
        self.human = evaluate.human_words(self.text, self.source)
        self.pairs = evaluate.align(self.human, evaluate.model_words(base))
        if exclude:  # the words heard inside an excluded span are left out of every score
            times = [w["start"] for p in base["paragraphs"] for w in p["words"] if build.norm(w["word"])]
            self.pairs = [(a, b) for a, b in self.pairs if not any(s <= times[b] < e for s, e in exclude)]

    def score(self, starts):
        doc = dict(self.doc, paragraph_starts=sorted(starts))
        return evaluate.evaluate(self.text, build.paragraphs(doc), self.pairs, self.source)


# ---- methods: each returns a set of word indices -------------------------------------------------------------

def none(lesson, **_):
    return set()


def pauses(lesson, n, m, sent_only=False, **_):
    """a new paragraph at a pause > n s, at least m words after the last one (sent_only: at sentence starts)"""
    w, out = lesson.words, set()
    for turn in lesson.rav_turns:
        last = turn.start
        for i in turn[1:]:
            gap = w[i]["start"] - w[i - 1]["end"]
            if gap > n and i - last >= m and (not sent_only or i in lesson.sent_starts):
                out.add(i)
                last = i
    return out


def thin(lesson, starts, k):
    """at most one start per k words: a start closer than k words to the previous one (or to its turn's start) is dropped"""
    if not k:
        return set(starts)
    out, last = set(), {}
    for i in sorted(starts):
        t = lesson.turn_of.get(i)
        if t is None:
            continue
        prev = last.get(t, lesson.rav_turns[t].start)
        if i - prev >= k:
            out.add(i)
            last[t] = i
    return out


def adapter(lesson, preds, k=0, pause=False, **_):
    p = preds.get(lesson.id)
    if p is None:
        raise KeyError(f"no adapter answer for {lesson.id}")
    if p["sentences"] != lesson.sentences:
        raise ValueError(f"{lesson.id}: sentences differ from the adapter's")
    starts = build.starts_from_sentences(lesson.index, lesson.offsets, p["sentences"], p["breaks"])
    # as the worker does (AI_TRNS_PARAGRAPHS_MIN_WORDS, AI_TRNS_PARAGRAPHS_BY_PAUSE)
    return set(build.thin_starts(lesson.words, starts, k, pause))


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1)


def _mean(vs):
    return [sum(c) / len(vs) for c in zip(*vs)]


def texttiling(lesson, emb, w, c, k, **_):
    """TextTiling on sentence embeddings: similarity of the w sentences before and after each sentence start,
    depth of its valley; a start where depth > mean + c * std, at least k words after the previous start"""
    e = emb[lesson.id]
    if len(e) != len(lesson.sentences):
        raise ValueError(f"{lesson.id}: {len(e)} embeddings for {len(lesson.sentences)} sentences")
    n = len(e)
    if n < 3:
        return set()
    key = (lesson.id, lesson.merge, w)
    if key not in _sims:
        _sims[key] = [None] + [_cos(_mean(e[max(0, b - w):b]), _mean(e[b:b + w])) for b in range(1, n)]
    sims = _sims[key]
    depth = [0.0] * n
    for b in range(1, n):
        left = sims[b]
        j = b - 1
        while j >= 1 and sims[j] >= left:
            left = sims[j]
            j -= 1
        right = sims[b]
        j = b + 1
        while j < n and sims[j] >= right:
            right = sims[j]
            j += 1
        depth[b] = left - sims[b] + right - sims[b]
    ds = depth[1:]
    mu = sum(ds) / len(ds)
    sd = math.sqrt(sum((x - mu) ** 2 for x in ds) / len(ds))
    cands = sorted((b for b in range(1, n) if depth[b] > mu + c * sd), key=lambda b: -depth[b])
    chosen = []
    for b in cands:  # deepest first, keeping k words apart
        i = lesson.sent_word[b]
        if all(abs(i - j) >= k or lesson.turn_of.get(i) != lesson.turn_of.get(j) for j in chosen):
            chosen.append(i)
    return set(chosen)


_sims = {}


def oracle(lesson, **_):
    """the ceiling of 'paragraphs start at sentence starts': the editor's starts moved to the nearest sentence start
    (<= 3 words, the same turn) -- uses the reference, not a method"""
    keyed = [i for i, w in enumerate(lesson.words) if build.norm(w["word"])]
    model = evaluate.model_words(build.paragraphs(lesson.doc))
    human = lesson.human
    out = set()
    for (a0, b0), (a1, b1) in zip(lesson.pairs, lesson.pairs[1:]):
        if human[a1]["para"] == human[a0]["para"] or human[a1]["label"] != evaluate.RAV:
            continue
        if model[b1]["label"] != evaluate.RAV or model[b0]["label"] != evaluate.RAV:
            continue
        i = keyed[b1]
        t = lesson.turn_of.get(i)
        if t is None:
            continue
        k = bisect.bisect_left(lesson.sent_word, i)
        near = [lesson.sent_word[c] for c in (k - 1, k) if 0 <= c < len(lesson.sent_word)]
        near = [j for j in near if abs(j - i) <= 3 and lesson.turn_of.get(j) == t]
        if near:
            out.add(min(near, key=lambda j: abs(j - i)))
    return out


METHODS = {"none": none, "pauses": pauses, "adapter": adapter, "texttiling": texttiling, "oracle": oracle}


def run(lessons, method, **params):
    tot = {"exact": [0, 0, 0], "tol1": [0, 0, 0]}
    per = {}
    for les in lessons:
        ev = les.score(METHODS[method](les, **params))
        per[les.id] = ev
        for name, key in (("exact", "paragraphs_exact"), ("tol1", "paragraphs_tol1")):
            for n, f in enumerate(("tp", "pred", "gold")):
                tot[name][n] += ev[key][f]
    out = {}
    for name, (tp, pred, gold) in tot.items():
        p = tp / pred if pred else 0.0
        r = tp / gold if gold else 0.0
        out[name] = {"p": round(p, 3), "r": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
                     "tp": tp, "pred": pred, "gold": gold}
    return out, per


def load_preds(d, name, merge):
    path = os.path.join(d, "preds", f"{name}.m{merge}.jsonl")
    if not os.path.exists(path):
        return None
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def load_emb(d, ids, merge):
    out = {}
    for lid in ids:
        path = os.path.join(d, "emb", f"{lid}.m{merge}.json")
        if not os.path.exists(path):
            return None
        out[lid] = json.load(open(path))
    return out


def grid_specs(d, ids, merges, adapters):
    for merge in merges:
        yield merge, "none", {}
        yield merge, "oracle", {}
        for n, m, s in itertools.product((0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), (0, 20, 40, 60, 80), (False, True)):
            yield merge, "pauses", {"n": n, "m": m, "sent_only": s}
        if load_emb(d, ids, merge) is not None:
            for w, c, k in itertools.product((2, 3, 4, 6), (0.0, 0.5, 1.0, 1.5), (0, 20, 40, 60)):
                yield merge, "texttiling", {"w": w, "c": c, "k": k}
        for a in adapters:
            if load_preds(d, a, merge) is not None:
                for k in (0, 10, 20, 30, 40, 50, 60, 80):
                    yield merge, "adapter", {"name": a, "k": k}
                    if k:
                        yield merge, "adapter", {"name": a, "k": k, "pause": True}


def label(method, params):
    if method == "adapter":
        return f"adapter {params['name']} k={params['k']}" + (" by pause" if params.get("pause") else "")
    return method + (" " + " ".join(f"{k}={v}" for k, v in params.items()) if params else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", required=True, help="tune / test: <this dir>/<set>.txt")
    ap.add_argument("--merges", default="0,3,6")
    ap.add_argument("--adapters", default="paragraphing,paragraphing_asr")
    ap.add_argument("--sources", action="store_true", help="the reference: the reading of <dir>/sources/<id>.txt is the reader's")
    ap.add_argument("--exclude", help="exclude.json: lessons / time spans left out")
    ap.add_argument("cmd", choices=["grid", "table", "texts", "speakers"], help="texts: write <dir>/texts.<set>.m<merge>.jsonl "
                    "(the Rav text of each lesson: input of predict_set.py / embed_set.py); speakers: the share of the "
                    "editor's Rav words the diarize document has as Rav and of the students' it doesn't, without and "
                    "with the source reading (and the exclusions)")
    ap.add_argument("--params", help="table: JSON list of [label, merge, method, params]")
    ap.add_argument("--out", help="write the results (JSON)")
    args = ap.parse_args()
    ids = read_set(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.set + ".txt"))
    ids = [i for i in ids if os.path.exists(os.path.join(args.dir, "out", i + ".json"))]
    merges = args.merges.split(",")  # "o", "s": the editor's speakers (diagnosis)
    exclude = json.load(open(args.exclude)) if args.exclude else {}
    if args.cmd == "speakers":
        return speakers_table(args.dir, ids, exclude, args.out)
    ids = [i for i in ids if exclude.get(i) != "all"]
    cache = {}

    def lessons(merge):
        if merge not in cache:
            cache[merge] = [Lesson(args.dir, i, merge, args.sources, exclude.get(i) or ()) for i in ids]
        return cache[merge]

    if args.cmd == "texts":
        for merge in merges:
            with open(os.path.join(args.dir, f"texts.{args.set}.m{merge}.jsonl"), "w", encoding="utf-8") as f:
                for les in lessons(merge):
                    f.write(json.dumps({"id": les.id, "text": les.flat,
                                        "sentences": [les.flat[a:b] for a, b in les.sentences]}, ensure_ascii=False) + "\n")
        return
    results = []
    if args.cmd == "grid":
        specs = list(grid_specs(args.dir, ids, merges, args.adapters.split(",")))
    else:
        specs = [(str(m), meth, p) for _, m, meth, p in json.load(open(args.params))]
    for merge, method, params in specs:
        extra = {}
        if method == "adapter":
            extra["preds"] = load_preds(args.dir, params["name"], merge)
        if method == "texttiling":
            extra["emb"] = load_emb(args.dir, ids, merge)
        tot, per = run(lessons(merge), method, **params, **extra)
        results.append({"merge": merge, "method": method, "params": params, "total": tot,
                        "per_lesson": {k: {"exact": v["paragraphs_exact"], "tol1": v["paragraphs_tol1"],
                                           "speakers": v["speakers"]} for k, v in per.items()}})
        e, t = tot["exact"], tot["tol1"]
        print(f"m{merge:<2} {label(method, params):45s} exact P {e['p']:.3f} R {e['r']:.3f} F1 {e['f1']:.3f} "
              f"| tol1 F1 {t['f1']:.3f} | pred {e['pred']} gold {e['gold']}", flush=True)
    print(f"{len(ids)} lessons: {' '.join(ids)}")
    if args.out:
        json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)


def speakers_table(d, ids, exclude, out=None):
    """Per lesson: Rav = editor's Rav words named Rav by diarize, students = editor's student words not named Rav;
    before (the transcript as is) and after (the source reading -> reader, exclusions left out); reading = the
    share of the source passages diarize gave to someone else (the reader, if he read them)"""
    rows = []
    for lid in ids:
        row = {"id": lid, "source": read_source(d, lid) is not None, "excluded": exclude.get(lid)}
        for key, kw in (("old", {}), ("new", {"sources": True, "exclude": exclude.get(lid) or ()})):
            if key == "new" and exclude.get(lid) == "all":
                continue
            les = Lesson(d, lid, "0", **kw)
            ev = les.score(set())
            conf = {tuple(k.split("->")): v for k, v in ev["confusion"].items()}
            tot = lambda h: sum(v for (a, _), v in conf.items() if a == h)
            rav = lambda h: conf.get((h, evaluate.RAV), 0)
            row[key] = {"rav": rav(evaluate.RAV) / tot(evaluate.RAV) if tot(evaluate.RAV) else None,
                        "rav_words": tot(evaluate.RAV),
                        "students": 1 - rav(evaluate.STUDENT) / tot(evaluate.STUDENT) if tot(evaluate.STUDENT) else None,
                        "student_words": tot(evaluate.STUDENT)}
            if key == "new" and les.source:
                old = evaluate.human_words(les.text)
                marked = {a for a, w in enumerate(les.human) if w["label"] != old[a]["label"]}
                pairs = [(a, b) for a, b in les.pairs if a in marked]
                model = evaluate.model_words(build.paragraphs(les.doc))
                row[key]["reading_words"] = len(marked)
                row[key]["reading_not_rav"] = (sum(model[b]["label"] != evaluate.RAV for _, b in pairs) / len(pairs)
                                               if pairs else None)
        rows.append(row)
        f = lambda x: "  -  " if x is None else f"{x:.3f}"
        o, n = row["old"], row.get("new") or {}
        print(f"{lid} src={'y' if row['source'] else 'n'} Rav {f(o['rav'])} -> {f(n.get('rav'))} ({o['rav_words']} -> "
              f"{n.get('rav_words', '-')} w) | students {f(o['students'])} -> {f(n.get('students'))} | reading "
              f"{n.get('reading_words', 0)} w, not Rav {f(n.get('reading_not_rav'))}", flush=True)
    if out:
        json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
