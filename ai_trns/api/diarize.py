"""
Speaker labels for an existing SRT (moved from stt-api /diarize). A job for the worker engine
"diarize" (ai_trns.diarize.engine) labels every word with its speaker; the labeled SRT is built
from those words here (ai_trns.diarize.labels). Results: GET /jobs/{id}/result.
"""
import json
import logging
import os
import re
from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from .. import sources
from ..diarize import KIND, labels
from ..diarize.srt import parse_srt
from . import config, results
from .auth import Principal, principal
from .submit import clip_of, create_job, job_view

router = APIRouter(prefix="/diarize", tags=["diarize"])
log = logging.getLogger("ai-trns")

_EXT_RE = re.compile(r"^\.[a-z0-9]{1,5}$")


def source_text(source: Optional[UploadFile]) -> Optional[bytes]:
    """The lesson's source text (what the reader reads), from .txt (UTF-8) or .docx; None if not given"""
    if source is None:
        return None
    content = source.file.read()
    if (source.filename or "").lower().endswith(".docx"):
        import io
        import docx
        try:
            document = docx.Document(io.BytesIO(content))
        except Exception:
            raise HTTPException(400, "source: not a valid .docx")
        return "\n".join(p.text for p in document.paragraphs).encode("utf-8")
    try:
        return content.decode("utf-8-sig").encode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "source must be UTF-8 text or .docx")


def fetch_sources(job_dir: str, uid: str, vocab_text: str):
    """In the background after the job is created: the lesson's sources -> <job dir>/sources.json (results.SOURCES).
    Fetched here, not by the worker: the source is used where the result is rendered (results.lesson_source),
    and a failure only leaves the job without a source"""
    cache = sources.Cache(config.SOURCES_CACHE_DIR or os.path.join(config.JOBS_DIR, "sources-cache"),
                          config.SOURCES_CACHE_MB, config.SOURCES_CACHE_DAYS)
    try:
        found = sources.fetch(uid, cache, vocab_text, timeout=config.SOURCES_TIMEOUT)
        if not os.path.isdir(job_dir):
            return
        path = os.path.join(job_dir, results.SOURCES)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(found, f, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    except Exception:
        log.exception("sources %s: not saved", uid)


def _job(request: Request, who: Principal, fmt: str, audio, url, srt, samples, sample_names, lang, speakers,
         paragraphs: bool = False, source: Optional[UploadFile] = None, uid: Optional[str] = None,
         background: Optional[BackgroundTasks] = None, start: Optional[str] = None, end: Optional[str] = None):
    try:
        cues = parse_srt(srt.file.read().decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise HTTPException(400, f"Invalid SRT: {e}")
    uid = (uid or "").strip() or None
    if uid and not sources.UID_RE.match(uid):
        raise HTTPException(400, "uid: a kmedia content unit id (8 letters and digits)")
    if sample_names and len(sample_names) != len(samples):
        raise HTTPException(400, "sample_names must match samples one-to-one")
    uploads, named = {}, []
    for i, sample in enumerate(samples):
        base, ext = os.path.splitext(os.path.basename(sample.filename or f"sample{i}"))
        file = f"sample{i}{ext.lower() if _EXT_RE.match(ext.lower()) else '.bin'}"
        uploads[file] = sample
        named.append({"name": (sample_names[i] if sample_names else base).strip() or base, "file": file})
    store = request.app.state.store
    params = {"kind": KIND, "lang": lang, "formats": [fmt], "profile": None, "fix_rtl": False,
              "samples": named, "speakers": {k: v for k, v in speakers.items() if v is not None},
              "paragraphs": paragraphs}
    part = clip_of(start, end)
    if part:
        params["clip"] = part
    inputs = {results.CUES: json.dumps(cues, ensure_ascii=False).encode("utf-8")}
    text = source_text(source)
    if text:
        inputs[results.SOURCE] = text
        params["source_name"] = os.path.basename(source.filename or "source")
    if uid:
        params["uid"] = uid
    job_id = create_job(store, who, params, audio, url, inputs, uploads)
    if uid and not text and background is not None:
        background.add_task(fetch_sources, store.job_dir(job_id), uid, " ".join(c["text"] for c in cues))
    return job_view(store, store.get(job_id))


# POST /diarize/srt  multipart: audio=<file> or url=<str>, srt=<file>,
#   samples=<file> (0..N, voice samples of known speakers on top of the worker's built-in ones; same name overrides),
#   sample_names=<str> (0..N, same order as samples; default: sample filename without extension),
#   num_speakers / min_speakers / max_speakers=<int> (optional hints), lang (default he: CTC romanization)
# -> 202 + job. GET /jobs/{id}/result: the same SRT, each cue's text prefixed with "<speaker>: "
# (see labels.display_name), cues where the speaker changes split. ?format=json: the words, as /diarize/words.
# Counts in the job's "stats" and the result's X-Diarize-* headers.
@router.post("/srt", status_code=202)
def diarize_srt(request: Request, who: Principal = Depends(principal),
                audio: Optional[UploadFile] = File(None), url: Optional[str] = Form(None), srt: UploadFile = File(...),
                samples: list[UploadFile] = File(default=[]), sample_names: list[str] = Form(default=[]),
                num_speakers: Optional[int] = Form(None, ge=1), min_speakers: Optional[int] = Form(None, ge=1),
                max_speakers: Optional[int] = Form(None, ge=1), lang: Literal["en", "he", "ru"] = Form("he"),
                paragraphs: bool = Form(False), source: Optional[UploadFile] = File(None),
                uid: Optional[str] = Form(None), start: Optional[str] = Form(None), end: Optional[str] = Form(None),
                background: BackgroundTasks = None):
    return _job(request, who, "srt", audio, url, srt, samples, sample_names, lang,
                dict(num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers), paragraphs, source,
                uid, background, start, end)


# POST /diarize/words  same fields as /diarize/srt, the same job
# GET /jobs/{id}/result: the diarization document (ai_trns.diarize):
#   {"words": [{"word", "start", "end", "speaker", "cue"}], "stats": {"speakers", "named", "mixed_cues",
#    "vad_fallback_cues"}, ...}; every word timed by CTC forced alignment (VAD fallback per cue),
#   cue = 0-based SRT cue index. ?format=srt: the labeled SRT.
#   paragraphs=true: the worker also splits the Rav's turns into paragraphs (the mdb-lora model, ai_trns.paragraphs);
#   ?format=paragraphs: [{"speaker", "label", "start", "end", "text", "words"}] -- a new paragraph at every speaker
#   change, inside the Rav's turns where the model puts one (without paragraphs=true: speaker turns only).
#   source=<file> (.txt or .docx, optional): the lesson's source text; a turn of someone else that reads it is
#   labeled קריין (the reader) in ?format=paragraphs.
#   uid=<kmedia content unit id> (optional; a `source` file wins): the server finds the lesson's sources itself
#   (its SOURCE units, LIKUTIM -- ai_trns.sources), in the background; the texts the transcript doesn't read
#   (another part's LIKUTIM) are dropped. No source found / kmedia down: the job goes on without one.
#   Which sources were used: the job's stats "sources": [{"id", "name", "unit", "kind", "matched_words", "used"}]
#   (+ "sources_error"), and the same in ?format=paragraphs "stats". An uploaded file: kind "upload".
#   start / end (optional; seconds, 3m17s or 03:17; no end: to the end): only this clip of the media is diarized;
#   the SRT's cues (in the media's time) outside it are dropped, the result's times stay the media's.
@router.post("/words", status_code=202)
def diarize_words(request: Request, who: Principal = Depends(principal),
                  audio: Optional[UploadFile] = File(None), url: Optional[str] = Form(None), srt: UploadFile = File(...),
                  samples: list[UploadFile] = File(default=[]), sample_names: list[str] = Form(default=[]),
                  num_speakers: Optional[int] = Form(None, ge=1), min_speakers: Optional[int] = Form(None, ge=1),
                  max_speakers: Optional[int] = Form(None, ge=1), lang: Literal["en", "he", "ru"] = Form("he"),
                  paragraphs: bool = Form(False), source: Optional[UploadFile] = File(None),
                  uid: Optional[str] = Form(None), start: Optional[str] = Form(None),
                  end: Optional[str] = Form(None), background: BackgroundTasks = None):
    return _job(request, who, "json", audio, url, srt, samples, sample_names, lang,
                dict(num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers), paragraphs, source,
                uid, background, start, end)


class _Word(BaseModel):
    word: str
    start: float
    end: float
    speaker: Optional[str] = None


class _WordsSrtRequest(BaseModel):
    words: list[_Word]
    max_chars: Optional[int] = None
    max_lines: Optional[int] = None
    labels: bool = True
    pad: bool = True


# POST /diarize/words/srt  JSON: {"words": [{"word", "start", "end", "speaker"}, ...]} (a diarize job's JSON
#   result as is; other fields ignored), optional max_chars (per line, 42), max_lines (2), labels (default true),
#   pad (default true: cues extended into the pauses around them, see resegment.pad_cues)
# -> SRT rebuilt from the words right away (see ai_trns.diarize.resegment): <= max_lines lines of ~max_chars,
# a new line at each speaker change, cues cut at the longest pause, then at punctuation.
@router.post("/words/srt", dependencies=[Depends(principal)])
def words_to_srt(req: _WordsSrtRequest = Body(...)):
    max_chars = req.max_chars or labels.MAX_CHARS
    max_lines = req.max_lines or labels.MAX_LINES
    if max_chars < 1 or max_lines < 1:
        raise HTTPException(400, "max_chars and max_lines must be positive")
    try:
        content = labels.words_srt([w.model_dump() for w in req.words], max_chars, max_lines, req.labels, req.pad)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(content, media_type=results.MEDIA_TYPES["srt"],
                    headers={"Content-Disposition": 'attachment; filename="words.srt"'})
