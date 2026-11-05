"""
The source text of a lesson (what the reader reads aloud), found by the lesson's kmedia unit id.

kmedia backend (public): the unit's `sources` (SOURCE units: the Hebrew .docx, else the .pdf), its derived LIKUTIM
units (Hebrew .docx, else .pdf) and its own files with insert_type "likutim" (none seen on lessons so far, accepted);
files from the CDN by file id; .docx -> the paragraphs' text (python-docx, as the API's uploaded source).
PDF (old TES parts have no .docx; needs pypdf, skipped without it): their glyphs are ISO-8859-8 codes shown as /Cnnn
or StandardEncoding names and the text has no spaces -> decoded and split into words by a vocabulary (the lesson's
transcript, the other sources): enough for the n-gram match of ai_trns.paragraphs.build.
A lesson's LIKUTIM is sometimes another part's: `select` keeps only the texts the transcript reads.
Fetching never raises (fetch): no source = the job goes on without one.
Measured on 37 lessons (TASK-paragraphs.md, 2026-11-05): 35 had a source; 3 sources matched nothing.
"""
import io
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Iterable, Optional

from .paragraphs import build

log = logging.getLogger("ai-trns")

BACKEND = "https://kabbalahmedia.info/backend"
CDN = "https://cdn.kabbalahmedia.info"
TIMEOUT = 15            # seconds per request
DEADLINE = 60           # seconds for all of a lesson's requests
MAX_FILE_MB = 20
UID_RE = re.compile(r"^[A-Za-z0-9]{8}$")
# A text the transcript reads at least this many words of (n-gram runs, build.source_covered) is the lesson's;
# fewer -- another part's LIKUTIM, a common phrase (as evaluate.SOURCE_MIN_WORDS: a reading, not a quote)
MIN_MATCHED_WORDS = 12
LONGEST_WORD = 20       # PDF segmentation: the longest vocabulary word tried
STD = {"Æ": 0xE1, "ª": 0xE3, "Ł": 0xE8, "Ø": 0xE9, "Œ": 0xEA, "º": 0xEB, "æ": 0xF1, "ı": 0xF5, "ł": 0xF8, "ø": 0xF9,
       "œ": 0xFA, "ß": 0xFB}


def _get(url: str, timeout: float) -> bytes:
    """GET -> body (at most MAX_FILE_MB); raises OSError / ValueError"""
    req = urllib.request.Request(url, headers={"User-Agent": "ai-trns"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read(MAX_FILE_MB * 1024 * 1024 + 1)
    if len(body) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"larger than {MAX_FILE_MB} MB")
    return body


def _json(url: str, timeout: float):
    return json.loads(_get(url, timeout))


def unit_files(unit: str, timeout: float = TIMEOUT) -> list[dict]:
    units = _json(f"{BACKEND}/content_units?id={urllib.parse.quote(unit)}&with_files=true", timeout)["content_units"]
    return units[0].get("files", []) if units else []


def text_files(uid: str, timeout: float = TIMEOUT) -> list[dict]:
    """[{"id", "name", "unit", "kind"}]: the Hebrew text files of a lesson's sources ("source"), LIKUTIM ("likutim")
    and own likutim files ("likutim"), the .docx preferred over a .pdf per unit"""
    lesson = _json(f"{BACKEND}/content_units/{urllib.parse.quote(uid)}", timeout)
    groups = [(s, "source", unit_files(s, timeout)) for s in lesson.get("sources") or []]
    groups += [(d.get("id") or k, "likutim", d.get("files", [])) for k, d in (lesson.get("derived_units") or {}).items()
               if d.get("content_type") == "LIKUTIM"]
    groups.append((uid, "likutim", [f for f in lesson.get("files", []) if f.get("insert_type") == "likutim"]))
    out = []
    for unit, kind, files in groups:
        he = [f for f in files if f.get("language") == "he" and f.get("type") == "text"
              and f.get("insert_type") != "source-scan"]
        docx = [f for f in he if f["name"].lower().endswith(".docx")]
        pdf = [f for f in he if f["name"].lower().endswith(".pdf")]
        out += [{"id": f["id"], "name": f["name"], "unit": unit, "kind": kind} for f in (docx or pdf[:1])]
    return out


def docx_text(content: bytes) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(io.BytesIO(content)).paragraphs)


def pdf_letters(content: bytes) -> str:
    """An old TES PDF -> its Hebrew letters, lines kept, no spaces (raises ImportError without pypdf)"""
    from pypdf import PdfReader
    text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(content)).pages)
    heb = lambda c: bytes([c]).decode("iso-8859-8") if 0xE0 <= c <= 0xFA else ""
    text = re.sub(r"/C(\d{3})", lambda m: heb(int(m.group(1))), text)
    return "".join(heb(STD[c]) if c in STD else c for c in text)


def segment(line: str, vocab: set, longest: int) -> str:
    """Letters without spaces -> words of vocab (an unknown letter costs 3, a word 1); keeps the letters' order"""
    s = build.norm(line)
    n = len(s)
    best = [0.0] + [float("inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        best[i], back[i] = best[i - 1] + 3, i - 1
        for j in range(max(0, i - longest), i):
            if s[j:i] in vocab and best[j] + 1 < best[i]:
                best[i], back[i] = best[j] + 1, j
    words, i = [], n
    while i > 0:
        words.append(s[back[i]:i])
        i = back[i]
    return " ".join(reversed(words))


def vocabulary(texts: Iterable[str]) -> set:
    vocab = {build.norm(w) for t in texts for w in t.split()}
    vocab.discard("")
    return vocab


def segment_text(letters: str, vocab: set) -> str:
    if not vocab:
        return ""
    longest = min(LONGEST_WORD, max(map(len, vocab)))
    return "\n".join(segment(line, vocab, longest) for line in letters.split("\n"))


def is_pdf(name: str) -> bool:
    return name.lower().endswith(".pdf")


class Cache:
    """Extracted texts by CDN file id (a file id's content never changes): <dir>/<id>.txt (a PDF's: its letters,
    not yet split into words). Kept at most max_days since last use and max_mb in all, the least recently used
    dropped first"""

    def __init__(self, path: str, max_mb: Optional[float] = 50, max_days: Optional[float] = 30):
        self.path, self.max_mb, self.max_days = path, max_mb, max_days

    def _file(self, fid: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9]+", fid):
            raise ValueError(f"bad file id {fid!r}")
        return os.path.join(self.path, fid + ".txt")

    def get(self, fid: str) -> Optional[str]:
        path = self._file(fid)
        try:
            if self.max_days is not None and time.time() - os.path.getmtime(path) > self.max_days * 86400:
                return None
            with open(path, encoding="utf-8") as f:
                text = f.read()
            os.utime(path)
            return text
        except OSError:
            return None

    def put(self, fid: str, text: str):
        os.makedirs(self.path, exist_ok=True)
        path = self._file(fid)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(path + ".tmp", path)
        self.prune()

    def prune(self):
        try:
            entries = [(e.stat().st_mtime, e.stat().st_size, e.path) for e in os.scandir(self.path)
                       if e.name.endswith(".txt")]
        except OSError:
            return
        now, total, keep = time.time(), 0, []
        for mtime, size, path in sorted(entries, reverse=True):     # newest first
            old = self.max_days is not None and now - mtime > self.max_days * 86400
            full = self.max_mb is not None and total + size > self.max_mb * 1024 * 1024
            if old or full:
                try:
                    os.remove(path)
                except OSError:
                    pass
            else:
                total += size
                keep.append(path)


def file_text(f: dict, cache: Optional[Cache] = None, timeout: float = TIMEOUT) -> str:
    """A text file of text_files -> its text (a PDF: its letters, see pdf_letters); raises on failure"""
    text = cache.get(f["id"]) if cache else None
    if text is None:
        content = _get(f"{CDN}/{urllib.parse.quote(f['id'])}", timeout)
        text = pdf_letters(content) if is_pdf(f["name"]) else docx_text(content)
        if cache:
            try:
                cache.put(f["id"], text)
            except OSError as e:
                log.warning("sources: cache %s: %s", f["id"], e)
    return text


def fetch(uid: str, cache: Optional[Cache] = None, vocab_text: str = "", timeout: float = TIMEOUT,
          deadline: float = DEADLINE) -> dict:
    """A lesson's source texts; never raises.
    {"uid", "files": [{"id", "name", "unit", "kind", "text"}], "skipped": [{"id", "name", "reason"}], "error"?}
    vocab_text: words to split a PDF's letters by (the lesson's transcript), on top of the other files' words"""
    out = {"uid": uid, "files": [], "skipped": []}
    if not UID_RE.match(uid or ""):
        out["error"] = "bad uid"
        return out
    end = time.monotonic() + deadline
    left = lambda: max(1.0, min(timeout, end - time.monotonic()))
    try:
        files = text_files(uid, left())
    except Exception as e:
        log.warning("sources %s: kmedia: %s", uid, e)
        out["error"] = f"kmedia: {e}"
        return out
    pdfs = []
    for f in files:
        if time.monotonic() > end:
            out["skipped"].append({"id": f["id"], "name": f["name"], "reason": "deadline"})
            continue
        try:
            text = file_text(f, cache, left())
        except ImportError:
            out["skipped"].append({"id": f["id"], "name": f["name"], "reason": "pypdf not installed"})
            continue
        except Exception as e:
            log.warning("sources %s: %s (%s): %s", uid, f["name"], f["id"], e)
            out["skipped"].append({"id": f["id"], "name": f["name"], "reason": str(e)[:200]})
            continue
        (pdfs if is_pdf(f["name"]) else out["files"]).append({**f, "text": text})
    if pdfs:
        vocab = vocabulary([vocab_text, *(f["text"] for f in out["files"])])
        for f in pdfs:
            text = segment_text(f["text"], vocab)
            if text.strip():
                out["files"].append({**f, "text": text})
            else:
                out["skipped"].append({"id": f["id"], "name": f["name"], "reason": "pdf: no words"})
    for s in out["skipped"]:
        log.info("sources %s: skipped %s (%s): %s", uid, s["name"], s["id"], s["reason"])
    log.info("sources %s: %s", uid, [f["name"] for f in out["files"]] or "none")
    return out


def transcript_keys(words: list[dict]) -> list[str]:
    """Diarization words -> their normalized forms, as matched_words takes them"""
    return [k for k in (build.norm(w["word"]) for w in words) if k]


def matched_words(text: str, keys: list[str]) -> int:
    """How many of the transcript's words (normalized) are in runs of build.READER_NGRAM words that the text has"""
    return sum(build.source_covered(keys, build.source_grams(text)))


def select(texts: list[dict], words: list[dict], min_matched: int = MIN_MATCHED_WORDS) -> tuple[Optional[str], list]:
    """The texts a lesson's transcript (diarization words) reads, joined -> (source text or None, per text
    {"id", "name", "unit", "kind", "matched_words", "used"}); a text read less than min_matched words is dropped"""
    keys = transcript_keys(words)
    used, info = [], []
    for t in texts:
        matched = matched_words(t["text"], keys)
        ok = matched >= min_matched
        info.append({**{k: t[k] for k in ("id", "name", "unit", "kind") if k in t}, "matched_words": matched,
                     "used": ok})
        if ok:
            used.append(t["text"])
        else:
            log.info("source %s (%s): %d words of the transcript, not used", t.get("name"), t.get("id"), matched)
    return ("\n\n".join(used) or None), info
