"""
Text <-> audio diff (moved from stt-api /tadiff and /docdiff): which words of a text were skipped
when it was read aloud, what was said that isn't in it, and what was read differently.

The audio is transcribed by an ordinary ai-trns job (the transcribe engine: same model, no
other GPU work) and, when it finishes, its words are compared with the text (ai_trns.textdiff).
For /docdiff the text comes from a .docx, and the passages skipped in the audio are highlighted
in a copy of it. Results: GET /jobs/{id}/result. /tadiff/compare-timestamps needs no audio and
answers at once.
"""
import io
import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from .. import textdiff
from . import results
from .auth import Principal, principal
from .submit import create_job, job_view

tadiff = APIRouter(prefix="/tadiff", tags=["tadiff"])
docdiff = APIRouter(prefix="/docdiff", tags=["docdiff"])

Lang = Literal["en", "he", "ru", "auto"]


def _job(request: Request, who: Principal, kind: str, fmt: str, audio, url, lang: str, inputs: dict, **params):
    store = request.app.state.store
    params = {"kind": kind, "lang": lang, "formats": [fmt], "profile": None, "fix_rtl": False, **params}
    return job_view(store, store.get(create_job(store, who, params, audio, url, inputs)))


# POST /tadiff/compare  multipart: audio=<file> or url=<str>, text=<str> (a form value, not a file),
#   lang (default he)
# -> 202 + job. GET /jobs/{id}/result: the diff (see ai_trns.textdiff); ?format=srt|txt|json gives the
# transcription. Counts in the job's "stats" and the result's X-Tadiff-* headers.
@tadiff.post("/compare", status_code=202)
def tadiff_compare(request: Request, who: Principal = Depends(principal),
                   audio: Optional[UploadFile] = File(None), url: Optional[str] = Form(None),
                   text: str = Form(...), lang: Lang = Form("he")):
    if not text.strip():
        raise HTTPException(400, "Provide text")
    return _job(request, who, "tadiff", "diff", audio, url, lang, {results.TEXT: text.encode("utf-8")})


# POST /tadiff/compare-timestamps  multipart: text=<file> (plain UTF-8), timestamps=<file> (JSON array of
#   {"word", "start", "end"} in heard order; the "words" of a diarize job's JSON fit)
# -> the diff right away, no job
@tadiff.post("/compare-timestamps", dependencies=[Depends(principal)])
def tadiff_compare_timestamps(text: UploadFile = File(...), timestamps: UploadFile = File(...)):
    try:
        content = text.file.read().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "text must be UTF-8")
    if not content.strip():
        raise HTTPException(400, "Provide text")
    try:
        words = textdiff.load_words(timestamps.file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return textdiff.compare(content, words)


# POST /docdiff/compare  multipart: doc=<.docx>, audio=<file> or url=<str>, lang (default he),
#   min_removed_words (default 5: shorter skipped passages are not highlighted)
# -> 202 + job. GET /jobs/{id}/result: the .docx with the skipped passages highlighted red, named
# <doc>_marked.docx; ?format=diff gives the whole diff. Counts in "stats" and X-Docdiff-* headers.
@docdiff.post("/compare", status_code=202)
def docdiff_compare(request: Request, who: Principal = Depends(principal), doc: UploadFile = File(...),
                    audio: Optional[UploadFile] = File(None), url: Optional[str] = Form(None),
                    lang: Lang = Form("he"), min_removed_words: int = Form(results.MIN_REMOVED_WORDS, ge=1)):
    import docx
    name = os.path.basename(doc.filename or "document.docx")
    if not name.lower().endswith(".docx"):
        raise HTTPException(400, "Only .docx is supported")
    content = doc.file.read()
    try:
        docx.Document(io.BytesIO(content))
    except Exception:
        raise HTTPException(400, "not a valid .docx")
    return _job(request, who, "docdiff", "docx", audio, url, lang, {results.DOC: content},
                doc_name=name, min_removed_words=min_removed_words)
