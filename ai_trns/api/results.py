"""
What a job makes of its worker's document, by kind (ai_trns.api.jobs.kind_of). CPU only.

    kind        engine       inputs in the job dir    formats (the first is the default)
    transcribe  transcribe   -                        txt, srt, json
    tadiff      transcribe   text.txt                 diff, txt, srt, json
    docdiff     transcribe   doc.docx                 docx, diff, txt, srt, json
    diarize     diarize      cues.json, sample<i>.*   srt, json
    live        live         -                        srt, txt, json

json is always the worker's document as is: a words document (ai_trns.words) from the
transcribe engine, a diarization document (ai_trns.diarize) from the diarize one.
diff is the text <-> heard words diff (ai_trns.textdiff), docx the document with the passages
skipped in the audio highlighted.
"""
import io
import json
import os
from typing import Optional
from urllib.parse import quote

from fastapi import HTTPException

from .. import textdiff
from ..diarize import labels
from ..subtitles import fix_rtl_srt, render as render_words
from .jobs import kind_of

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MEDIA_TYPES = {"txt": "text/plain; charset=utf-8", "srt": "application/x-subrip; charset=utf-8",
               "json": "application/json", "diff": "application/json", "docx": DOCX}
FORMATS = {
    "transcribe": ("txt", "srt", "json"),
    "tadiff": ("diff", "txt", "srt", "json"),
    "docdiff": ("docx", "diff", "txt", "srt", "json"),
    "diarize": ("srt", "json"),
    "live": ("srt", "txt", "json"),
}
# Counters a finished job keeps ("stats" in the job view, X-<Kind>-* headers of its result)
STATS = {
    "tadiff": ("text_words", "audio_words", "removed", "added", "changed"),
    "docdiff": ("text_words", "audio_words", "removed_total", "removed_marked", "removed", "added", "changed"),
    "diarize": ("speakers", "named", "split_cues", "mixed_cues", "vad_fallback_cues"),
}
# Inputs a job kind keeps in its dir, next to the uploaded audio
TEXT, DOC, CUES = "text.txt", "doc.docx", "cues.json"
# docdiff: shortest skipped passage that is highlighted; shorter gaps are mostly ASR noise
MIN_REMOVED_WORDS = 5


def formats(params: dict) -> tuple:
    return FORMATS[kind_of(params)]


def parse_formats(value: str, kind: str) -> list:
    """Comma-separated formats asked for a new job, or 422"""
    formats = sorted({f.strip() for f in value.split(",") if f.strip()})
    if not formats or any(f not in FORMATS[kind] for f in formats):
        raise HTTPException(422, f"format must be a comma-separated subset of {', '.join(FORMATS[kind])}")
    return formats


def _header(kind: str, key: str) -> str:
    return f"X-{kind.capitalize()}-{key.replace('_', '-').title()}"


STATS_HEADERS = [_header(kind, key) for kind, keys in STATS.items() for key in keys]


def stats_headers(job: dict) -> dict:
    kind = kind_of(job["params"])
    return {_header(kind, k): str(v) for k, v in (job.get("stats") or {}).items() if k in STATS.get(kind, ())}


def content_disposition(name: str) -> str:
    """attachment with a UTF-8 file name, and an ASCII fallback for old clients"""
    fallback = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(name)}"


def file_name(fmt: str) -> str:
    """Where a rendered format is kept in the job dir"""
    return "result.diff.json" if fmt == "diff" else f"result.{fmt}"


def download_name(job: dict, fmt: str) -> str:
    params = job["params"]
    if fmt == "docx" and params.get("doc_name"):
        return f"{os.path.splitext(params['doc_name'])[0]}_marked.docx"
    return f"{job['id']}.diff.json" if fmt == "diff" else f"{job['id']}.{fmt}"


def _read(job_dir: str, name: str, mode: str = "r"):
    with open(os.path.join(job_dir, name), mode, **({} if "b" in mode else {"encoding": "utf-8"})) as f:
        return f.read()


def _docdiff(job_dir: str, params: dict, doc: dict) -> tuple[bytes, dict]:
    import docx
    from .docx_marks import extract_text, mark_red
    document = docx.Document(io.BytesIO(_read(job_dir, DOC, "rb")))
    diff = textdiff.compare(extract_text(document), textdiff.words_of(doc))
    min_words = params.get("min_removed_words") or MIN_REMOVED_WORDS
    marked = [tuple(e["text_idx"]) for e in diff["removed"] if e["text_idx"][1] - e["text_idx"][0] + 1 >= min_words]
    mark_red(document, marked)
    out = io.BytesIO()
    document.save(out)
    return out.getvalue(), {**diff["stats"], "removed_marked": len(marked), "removed_total": len(diff["removed"])}


def _diff(job_dir: str, params: dict, doc: dict) -> dict:
    if kind_of(params) == "docdiff":
        import docx
        from .docx_marks import extract_text
        text = extract_text(docx.Document(io.BytesIO(_read(job_dir, DOC, "rb"))))
    else:
        text = _read(job_dir, TEXT)
    return textdiff.compare(text, textdiff.words_of(doc))


def _diff_stats(diff: dict) -> dict:
    return {**diff["stats"], **{k: len(diff[k]) for k in ("removed", "added", "changed")}}


def render(job_dir: str, params: dict, doc: dict, fmt: str,
           fix_rtl: Optional[bool] = None) -> tuple[str | bytes, Optional[dict]]:
    """The worker's document -> one format of the job's kind, and the job's stats if this format
    yields them (None otherwise). Raises KeyError / TypeError / ValueError on a bad document"""
    kind = kind_of(params)
    if fmt not in FORMATS[kind]:
        raise ValueError(f"unknown format {fmt!r} for a {kind} job")
    fix_rtl = params.get("fix_rtl", False) if fix_rtl is None else fix_rtl
    if kind == "diarize":
        labels.check(doc)
        if fmt == "json":
            return json.dumps(doc, ensure_ascii=False), None
        content, stats = labels.labeled_srt(json.loads(_read(job_dir, CUES)), doc)
        stats = {**stats, **{k: doc["stats"][k] for k in ("mixed_cues", "vad_fallback_cues") if k in doc["stats"]}}
        return (fix_rtl_srt(content) if fix_rtl and doc.get("language") == "he" else content), stats
    if fmt == "docx":
        return _docdiff(job_dir, params, doc)
    if fmt == "diff":
        diff = _diff(job_dir, params, doc)
        return json.dumps(diff, ensure_ascii=False), _diff_stats(diff)
    return render_words(doc, fmt, fix_rtl), None
