"""
Speaker labels for an existing SRT (moved from stt-api /diarize). A job for the worker engine
"diarize" (ai_trns.diarize.engine) labels every word with its speaker; the labeled SRT is built
from those words here (ai_trns.diarize.labels). Results: GET /jobs/{id}/result.
"""
import json
import os
import re
from typing import Literal, Optional

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..diarize import KIND, labels
from ..diarize.srt import parse_srt
from . import results
from .auth import Principal, principal
from .submit import create_job, job_view

router = APIRouter(prefix="/diarize", tags=["diarize"])

_EXT_RE = re.compile(r"^\.[a-z0-9]{1,5}$")


def _job(request: Request, who: Principal, fmt: str, audio, url, srt, samples, sample_names, lang, speakers):
    try:
        cues = parse_srt(srt.file.read().decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as e:
        raise HTTPException(400, f"Invalid SRT: {e}")
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
              "samples": named, "speakers": {k: v for k, v in speakers.items() if v is not None}}
    inputs = {results.CUES: json.dumps(cues, ensure_ascii=False).encode("utf-8")}
    return job_view(store, store.get(create_job(store, who, params, audio, url, inputs, uploads)))


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
                max_speakers: Optional[int] = Form(None, ge=1), lang: Literal["en", "he", "ru"] = Form("he")):
    return _job(request, who, "srt", audio, url, srt, samples, sample_names, lang,
                dict(num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers))


# POST /diarize/words  same fields as /diarize/srt, the same job
# GET /jobs/{id}/result: the diarization document (ai_trns.diarize):
#   {"words": [{"word", "start", "end", "speaker", "cue"}], "stats": {"speakers", "named", "mixed_cues",
#    "vad_fallback_cues"}, ...}; every word timed by CTC forced alignment (VAD fallback per cue),
#   cue = 0-based SRT cue index. ?format=srt: the labeled SRT.
@router.post("/words", status_code=202)
def diarize_words(request: Request, who: Principal = Depends(principal),
                  audio: Optional[UploadFile] = File(None), url: Optional[str] = Form(None), srt: UploadFile = File(...),
                  samples: list[UploadFile] = File(default=[]), sample_names: list[str] = Form(default=[]),
                  num_speakers: Optional[int] = Form(None, ge=1), min_speakers: Optional[int] = Form(None, ge=1),
                  max_speakers: Optional[int] = Form(None, ge=1), lang: Literal["en", "he", "ru"] = Form("he")):
    return _job(request, who, "json", audio, url, srt, samples, sample_names, lang,
                dict(num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers))


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
