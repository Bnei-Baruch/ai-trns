"""
Lesson markup from a diarize job's paragraphs and, if the lesson has one, its human transcript. Stdlib only.

    {
      "version": 1, "uid", "language",
      "media":      {"file_uid", "duration_s"},
      "transcript": {"file_uid"} | null,              # the kmedia transcript (archive-unzip doc2html) it is built on
      "run":        {"id", "created", "model", "source": "transcript" | "audio"},
      "paragraphs": [{"i", "start", "end", "speaker"}],
      "chapters":   [{"type", "title", "start", "end", "p_start", "p_end"}],
      "stats":      {...}
    }

paragraphs[].i, chapters[].p_start / p_end: index of the <p> element in the transcript's doc2html HTML -- the
paragraphs kmedia shows -- or, with no transcript (source "audio"), of the diarize paragraph (its text is given).
speaker: הרב / קריין / תלמיד (ai_trns.diarize.labels.display_name) or null. Times in seconds of the media.

Chapter types (a client shows an unknown one as "other"):
  reading     the reader reads the source; title: the reader's announcement as written ("קטע מס' 46 ...",
              "אות ט\"ו") -- not parsed: transcripts number items in different ways and many lessons have none
  commentary  the Rav after a reading; title: the reading's
  qa          a student's question, the follow-ups and the Rav's answers; title: the question
  other       opening, announcements
Not marked (hard to get reliably, left out on purpose): the source item as a number, its place in the source
document, seminars.

With a transcript the chapters come from its marks, as the editors write them: "קריין: ..." starts a reading (but
the lesson's own title, "קריין: שיעור ..."), "שאלה:" or a "תלמיד:" after the Rav spoke starts a question (one
right after a student's turn is a follow-up in it), unmarked paragraphs are the Rav's. The paragraphs get
their times by aligning their words to the heard ones (ai_trns.textdiff). Without one the chapters come from the
audio: the diarize paragraphs' `reading` flag and speakers.
"""
import html as htmllib
import re
from typing import Optional

from .. import textdiff

RAV, READER, STUDENT = "הרב", "קריין", "תלמיד"

_P_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_READER_RE = re.compile(r"^\s*קריין\s*:")
_QUESTION_RE = re.compile(r"^\s*שאלה\s*:")
_STUDENT_RE = re.compile(r"^\s*(?:תלמיד|תלמידה)\s*:")
_MARK_RE = re.compile(r"^\s*(?:קריין|שאלה|תלמיד|תלמידה)\s*:\s*")
# the reader announcing the lesson itself, not a reading
_LESSON_TITLE_RE = re.compile(r"^\s*קריין\s*:\s*(?:שיעור|שיחה|כנס)\b")
# a paragraph of the reading itself: the quoted text and its reference "(הרב"ש. מאמר 3 ...)"
_QUOTED = ('"', "“", "״", "„", "(")
TITLE_CHARS = 120


def transcript_paragraphs(doc_html: str) -> list[str]:
    """The text of every <p> of a doc2html document, in order (empty ones kept: the index is the <p>'s)"""
    return [" ".join(htmllib.unescape(_TAG_RE.sub("", m)).split()) for m in _P_RE.findall(doc_html)]


def _title(text: str) -> str:
    text = _MARK_RE.sub("", text).strip()
    return text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS].rsplit(" ", 1)[0] + "…"


def marked_chapters(paras: list[str]) -> Optional[list[dict]]:
    """Chapters by the transcript's marks: [{type, title, p_start, p_end}]; None if it has no marks"""
    if not any(_READER_RE.match(p) or _QUESTION_RE.match(p) or _STUDENT_RE.match(p) for p in paras):
        return None
    out, cur, reading_title = [], None, ""
    # with "שאלה:" marks a "תלמיד:" is a follow-up in the question; without them it asks a new one
    uses_question = any(_QUESTION_RE.match(p) for p in paras)

    def start(kind, i, title=""):
        nonlocal cur
        cur = {"type": kind, "title": title, "p_start": i, "p_end": i}
        out.append(cur)

    prev = None  # the previous paragraph's mark: "reader" | "student" | None (the Rav)
    for i, p in enumerate(paras):
        if not p:
            continue
        mark = "reader" if _READER_RE.match(p) else "student" if (_QUESTION_RE.match(p) or _STUDENT_RE.match(p)) \
            else None
        if mark == "reader":
            if _LESSON_TITLE_RE.match(p):
                if cur is None or cur["type"] != "other":
                    start("other", i, _title(p))
            elif cur is not None and cur["type"] == "reading" and prev == "reader":
                # the book, then the item: one announcement
                cur["title"] = reading_title = _title(f"{cur['title']} · {_title(p)}")
            else:
                reading_title = _title(p)
                start("reading", i, reading_title)
        elif mark == "student":
            follow_up = cur is not None and cur["type"] == "qa" and not _QUESTION_RE.match(p) \
                and (uses_question or prev == "student")
            if not follow_up:
                start("qa", i, _title(p))
        elif cur is None:
            start("other", i, _title(p))
        elif cur["type"] == "reading" and i > cur["p_start"] + 1 and not p.startswith(_QUOTED):
            start("commentary", i, reading_title)
        cur["p_end"] = i
        prev = mark
    merged = []  # one opening, not one per paragraph mark before the first reading
    for c in out:
        if merged and c["type"] == "other" and merged[-1]["type"] == "other":
            merged[-1]["p_end"] = c["p_end"]
        else:
            merged.append(c)
    return merged


def marked_speaker(paras: list[str], chapters: list[dict]) -> dict[int, str]:
    """{paragraph index: speaker} the marks tell: the reader's announcements and reading, the students' turns"""
    out = {}
    for c in chapters:
        for i in range(c["p_start"], c["p_end"] + 1):
            p = paras[i]
            if not p:
                continue
            if _READER_RE.match(p) or (c["type"] == "reading" and i > c["p_start"] and p.startswith(_QUOTED)):
                out[i] = READER
            elif _QUESTION_RE.match(p) or _STUDENT_RE.match(p):
                out[i] = STUDENT
    return out


def align(paras: list[str], words: list[dict]) -> list[Optional[tuple]]:
    """(start, end, [heard word indices]) of every paragraph by its words' match to the heard ones; None where
    nothing matched or the match runs back in time (a repeated phrase matched elsewhere)"""
    ref, ref_para = [], []
    for i, p in enumerate(paras):
        for tok in _MARK_RE.sub("", p).split():
            norm = textdiff._norm(tok)
            if norm:
                ref.append(norm)
                ref_para.append(i)
    heard = [j for j, w in enumerate(words) if textdiff._norm(w["word"])]
    hyp = [textdiff._norm(words[j]["word"]) for j in heard]
    matched: dict[int, list[int]] = {}
    for op in textdiff._word_ops(ref, hyp, textdiff.FUZZY_THRESHOLD):
        if op[0] == "eq":
            matched.setdefault(ref_para[op[1]], []).append(heard[op[2]])
    out: list[Optional[tuple]] = [None] * len(paras)
    last_start = -1.0
    for i in range(len(paras)):
        js = matched.get(i)
        # a paragraph needs a few matched words: one stray common word is not an anchor
        if not js or len(js) < min(3, len([t for t in paras[i].split() if textdiff._norm(t)])):
            continue
        start, end = words[js[0]]["start"], words[js[-1]]["end"]
        if start < last_start:
            continue
        out[i] = (start, end, js)
        last_start = start
    return out


def _fill(spans: list[Optional[tuple]], paras: list[str], duration: float) -> list[Optional[tuple]]:
    """Non-empty paragraphs without a time get the gap between their timed neighbours"""
    out = list(spans)
    for i, p in enumerate(paras):
        if not p or out[i] is not None:
            continue
        prev = next((out[k][1] for k in range(i - 1, -1, -1) if out[k]), 0.0)
        nxt = next((out[k][0] for k in range(i + 1, len(out)) if out[k]), duration)
        out[i] = (prev, max(prev, nxt), [])
    return out


def _majority(labels: list[Optional[str]]) -> Optional[str]:
    counts: dict[str, int] = {}
    for x in labels:
        if x:
            counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get) if counts else None


def heard_words(diarize_paragraphs: list[dict]) -> list[dict]:
    """The words of a diarize job's paragraphs, each with its paragraph's label and reading flag"""
    return [{"word": w["word"], "start": w["start"], "end": w["end"], "label": p.get("label"),
             "reading": bool(p.get("reading"))}
            for p in diarize_paragraphs for w in p.get("words", [])]


def _chapter_times(chapters: list[dict], paragraphs: list[dict], duration: float) -> list[dict]:
    at = {p["i"]: p for p in paragraphs}
    out = []
    for c in chapters:
        ps = [at[i] for i in range(c["p_start"], c["p_end"] + 1) if i in at]
        if not ps:
            continue
        out.append({**c, "start": ps[0]["start"], "end": ps[-1]["end"]})
    for a, b in zip(out, out[1:]):  # a chapter lasts until the next one starts
        a["end"] = max(a["start"], b["start"])
    if out:
        out[0]["start"] = min(out[0]["start"], 0.0) if out[0]["type"] == "other" else out[0]["start"]
        out[-1]["end"] = max(out[-1]["end"], duration)
    return out


def from_transcript(paras: list[str], words: list[dict], duration: float) -> Optional[tuple[list, list, dict]]:
    """(paragraphs, chapters, stats) of a marked transcript on the heard words; None if it has no marks"""
    chapters = marked_chapters(paras)
    if chapters is None:
        return None
    spans = align(paras, words)
    aligned = sum(1 for p, s in zip(paras, spans) if p and s)
    spans = _fill(spans, paras, duration)
    marked = marked_speaker(paras, chapters)
    paragraphs = []
    for i, (p, s) in enumerate(zip(paras, spans)):
        if not p or s is None:
            continue
        # not heard (a header of the document): no speaker
        speaker = marked.get(i) or _majority([words[j]["label"] for j in s[2]]) or (RAV if s[2] else None)
        if speaker == READER and i not in marked:  # the audio's reader where the text has no mark: the Rav
            speaker = RAV
        paragraphs.append({"i": i, "start": round(s[0], 2), "end": round(s[1], 2), "speaker": speaker})
    stats = {"transcript_paragraphs": sum(1 for p in paras if p), "aligned_paragraphs": aligned}
    return paragraphs, _chapter_times(chapters, paragraphs, duration), stats


def from_audio(diarize_paragraphs: list[dict], duration: float) -> tuple[list, list, dict]:
    """(paragraphs, chapters, stats) from the diarize paragraphs alone: reading by the `reading` flag, the Rav
    after it commentary, a student's turn opens a question"""
    paragraphs, chapters, cur = [], [], None
    for i, p in enumerate(diarize_paragraphs):
        label = p.get("label")
        paragraphs.append({"i": i, "start": round(p["start"], 2), "end": round(p["end"], 2), "speaker": label,
                           "text": p.get("text", "")})
        kind = None  # None: the paragraph goes on the current chapter
        if p.get("reading"):
            kind = None if cur is not None and cur["type"] == "reading" else "reading"
        elif label == STUDENT:
            # a new question unless it is a follow-up in a question the Rav has not answered yet
            if cur is None or cur["type"] != "qa" or _rav_answered(diarize_paragraphs, cur["p_start"], i):
                kind = "qa"
        elif cur is not None and cur["type"] == "reading":
            kind = "commentary"
        elif cur is None:
            kind = "other"
        if kind:
            cur = {"type": kind, "title": _title(p.get("text", "")) if kind in ("qa", "reading") else "",
                   "p_start": i, "p_end": i}
            chapters.append(cur)
        cur["p_end"] = i
    return paragraphs, _chapter_times(chapters, paragraphs, duration), {"diarize_paragraphs": len(paragraphs)}


def _rav_answered(ps: list[dict], a: int, b: int) -> bool:
    return any(ps[k].get("label") == RAV for k in range(a, b))


def markup(uid: str, language: str, media: dict, diarize_paragraphs: list[dict], transcript_html: Optional[str],
           transcript: Optional[dict], run: dict) -> dict:
    """The markup document (see the module doc)"""
    duration = float(media.get("duration_s") or (diarize_paragraphs[-1]["end"] if diarize_paragraphs else 0.0))
    built, source = None, "audio"
    if transcript_html:
        built = from_transcript(transcript_paragraphs(transcript_html), heard_words(diarize_paragraphs), duration)
        source = "transcript" if built else "audio"
    if built is None:
        built = from_audio(diarize_paragraphs, duration)
    paragraphs, chapters, stats = built
    from . import VERSION
    return {"version": VERSION, "uid": uid, "language": language, "media": media,
            "transcript": transcript if source == "transcript" else None,
            "run": {**run, "source": source}, "paragraphs": paragraphs, "chapters": chapters,
            "stats": {**stats, "chapters": len(chapters)}}
