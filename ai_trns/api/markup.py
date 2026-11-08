"""
Lesson markup by a kmedia lesson uid (ai_trns.markup): chapters and the transcript's paragraphs on the lesson's
time line, read by the kmedia lesson page.

    POST /markup              uid=<kmedia unit>, language=he, force=false   (no login for now, see TODO.md) -> 202 state
    GET  /markup/{uid}        the markup document, no login (lessons are public); 404 until there is one
    GET  /markup/{uid}/status the state of its run, no login

A run is a chain of jobs the server drives itself: the lesson's original-language audio from the CDN is
transcribed (a transcribe job), its words labeled by speaker with paragraphs (a diarize job, which also fetches the
lesson's source by uid), then the markup is built here from those paragraphs and, if the lesson has one, its
transcript as kmedia shows it (archive-unzip doc2html). Pending runs move on on every read and in the maintenance
thread (sync_all), like the review lessons.

Storage (AI_TRNS_MARKUP_DIR), kept without the jobs' TTL:
  state/{uid}.json          -- the last run: status pending -> ready | failed, stage transcribe -> diarize
  runs/{uid}/{run_id}.json  -- every finished run's document, never changed
  {uid}.json                -- the current document (a copy of the last ready run)
"""
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .. import sources
from ..diarize import KIND as DIARIZE
from ..diarize.srt import parse_srt
from ..markup import build
from ..subtitles import render as render_words
from . import config, results
from .auth import Principal, principal
from .diarize import fetch_sources
from .jobs import DONE, FINAL, JobStore
from .submit import create_job

log = logging.getLogger("ai-trns.markup")

router = APIRouter(prefix="/markup", tags=["markup"])

UNIT_URL = "https://kabbalahmedia.info/backend/content_units/{uid}"
CDN_URL = "https://cdn.kabbalahmedia.info/{file_id}"
DOC2HTML_URL = "https://kabbalahmedia.info/assets/api/doc2html/{file_id}"
LANGUAGES = ("he", "en", "ru")
TRANSCRIPT_TYPES = ("tamlil", "", None)
# the markup is public data of public lessons: any site may read it
PUBLIC = {"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=300"}

lock = threading.RLock()

# the owner of runs started without login while authorization is off (start_markup)
ANONYMOUS = Principal(owner="anon:markup", name="anonymous", sees_all=False)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dir(*parts: str) -> str:
    return os.path.join(config.MARKUP_DIR, *parts)


def _read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def load_state(uid: str) -> Optional[dict]:
    try:
        return _read(_dir("state", f"{uid}.json"))
    except FileNotFoundError:
        return None


def save_state(state: dict) -> None:
    _write(_dir("state", f"{state['uid']}.json"), state)


def _check_uid(uid: str) -> str:
    uid = (uid or "").strip()
    if not sources.UID_RE.match(uid):
        raise HTTPException(400, "uid: a kmedia content unit id (8 letters and digits)")
    return uid


# --- kmedia

def fetch_unit(uid: str) -> dict:
    try:
        with urllib.request.urlopen(UNIT_URL.format(uid=uid), timeout=config.SOURCES_TIMEOUT) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(uid)
        raise


def fetch_transcript_html(file_id: str) -> str:
    req = urllib.request.Request(DOC2HTML_URL.format(file_id=file_id), headers={"User-Agent": "ai-trns"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return res.read().decode("utf-8")


def lesson_files(unit: dict, language: str) -> tuple[Optional[dict], Optional[dict]]:
    """The unit's audio in `language` (the original recording -- "_o_" in its name -- before a translation) and its
    transcript in that language (a text file, not a PDF), or None"""
    files = unit.get("files") or []
    audio = [f for f in files if f.get("type") == "audio" and f.get("language") == language
             and f.get("mimetype", "audio/mpeg") == "audio/mpeg"]
    audio.sort(key=lambda f: "_o_" not in (f.get("name") or ""))
    texts = [f for f in files if f.get("type") == "text" and f.get("language") == language
             and f.get("insert_type") in TRANSCRIPT_TYPES and not (f.get("name") or "").lower().endswith(".pdf")]
    texts.sort(key=lambda f: not (f.get("name") or "").lower().endswith(".docx"))
    return (audio[0] if audio else None), (texts[0] if texts else None)


# --- the chain

def _job_done(store: JobStore, job_id: Optional[str]) -> tuple[Optional[dict], Optional[str]]:
    """(the finished job, None), (None, None) while it runs, (None, error) if it failed"""
    job = store.get(job_id) if job_id else None
    if job is None:
        return None, "job is lost"
    if job["status"] not in FINAL:
        return None, None
    if job["status"] != DONE:
        return None, job["error"] or job["status"]
    return job, None


def _start_diarize(store: JobStore, state: dict, words_doc: dict) -> str:
    cues = parse_srt(render_words(words_doc, "srt", False))
    params = {"kind": DIARIZE, "lang": state["language"], "formats": ["paragraphs"], "profile": None,
              "fix_rtl": False, "samples": [], "speakers": {}, "paragraphs": True, "uid": state["uid"]}
    who = Principal(owner=state["owner"], name=state.get("owner_name") or state["owner"], sees_all=False)
    job_id = create_job(store, who, params, None, state["audio_url"],
                        {results.CUES: json.dumps(cues, ensure_ascii=False).encode("utf-8")})
    vocab = " ".join(c["text"] for c in cues)
    threading.Thread(target=fetch_sources, args=(store.job_dir(job_id), state["uid"], vocab),
                     name=f"sources-{state['uid']}", daemon=True).start()
    return job_id


def _build(store: JobStore, state: dict, job: dict) -> dict:
    job_dir = store.job_dir(job["id"])
    doc = _read(os.path.join(job_dir, "result.json"))
    content, _ = results.render(job_dir, job["params"], doc, "paragraphs")
    paragraphs = json.loads(content)["paragraphs"]
    html = None
    if state.get("transcript"):
        try:
            html = fetch_transcript_html(state["transcript"]["file_uid"])
        except (OSError, ValueError) as e:
            log.warning("markup %s: no transcript (%s), built from the audio", state["uid"], e)
    run = {"id": state["run_id"], "created": now(), "model": doc.get("model"),
           "jobs": {"transcribe": state["transcribe_job"], "diarize": state["diarize_job"]}}
    return build.markup(state["uid"], state["language"], state["media"], paragraphs, html, state.get("transcript"),
                        run)


def _advance(store: JobStore, state: dict) -> dict:
    if state["stage"] == "transcribe":
        job, error = _job_done(store, state.get("transcribe_job"))
        if error:
            return {**state, "status": "failed", "error": f"transcription: {error}"}
        if job is None:
            return state
        words_doc = _read(os.path.join(store.job_dir(job["id"]), "result.json"))
        return {**state, "stage": "diarize", "diarize_job": _start_diarize(store, state, words_doc)}
    job, error = _job_done(store, state.get("diarize_job"))
    if error:
        return {**state, "status": "failed", "error": f"speakers: {error}"}
    if job is None:
        return state
    doc = _build(store, state, job)
    _write(_dir("runs", state["uid"], f"{state['run_id']}.json"), doc)
    _write(_dir(f"{state['uid']}.json"), doc)
    return {**state, "status": "ready", "finished": now(), "stats": doc["stats"]}


def sync(store: JobStore, uid: str) -> Optional[dict]:
    """The run's state, moved on if its current job has finished"""
    state = load_state(uid)
    if state is None or state.get("status") != "pending":
        return state
    with lock:
        state = load_state(uid)
        if state.get("status") != "pending":
            return state
        try:
            new = _advance(store, state)
        except Exception as e:
            log.exception("markup %s: failed", uid)
            new = {**state, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        if new != state:
            save_state(new)
            log.info("markup %s: %s %s", uid, new["status"], new.get("stage"))
        return new


def sync_all(store: JobStore) -> None:
    try:
        names = os.listdir(_dir("state"))
    except FileNotFoundError:
        return
    for name in names:
        if name.endswith(".json"):
            sync(store, name[:-5])


def _public(state: dict) -> dict:
    return {k: v for k, v in state.items() if k not in ("owner", "owner_name")}


# --- requests

@router.post("", status_code=202)
def start_markup(request: Request, uid: str = Form(...), language: str = Form("he"), force: bool = Form(False)):
    # TODO: authorization is temporarily off, anyone may start a run (see TODO.md); to restore it put back
    # `who: Principal = Depends(principal)` in the arguments
    who = ANONYMOUS
    request.state.note = f"uid={uid} language={language} force={force}"
    uid = _check_uid(uid)
    if language not in LANGUAGES:
        raise HTTPException(400, f"language: one of {', '.join(LANGUAGES)}")
    store = request.app.state.store
    with lock:
        state = sync(store, uid)
        if state and state["status"] == "pending" and not force:
            return JSONResponse(_public(state), status_code=202)
        try:
            unit = fetch_unit(uid)
        except LookupError:
            raise HTTPException(404, f"no kmedia unit {uid}")
        except (OSError, ValueError) as e:
            raise HTTPException(502, f"kmedia: {e}")
        audio, transcript = lesson_files(unit, language)
        if audio is None:
            raise HTTPException(422, f"unit {uid} has no {language} audio")
        url = CDN_URL.format(file_id=audio["id"])
        params = {"lang": language, "formats": ["json"], "profile": None, "fix_rtl": False}
        job_id = create_job(store, who, params, None, url)
        state = {"uid": uid, "language": language, "run_id": uuid.uuid4().hex[:12], "status": "pending",
                 "stage": "transcribe", "created": now(), "owner": who.owner, "owner_name": who.name,
                 "audio_url": url,
                 "media": {"file_uid": audio["id"], "duration_s": audio.get("duration") or unit.get("duration")},
                 "transcript": {"file_uid": transcript["id"]} if transcript else None,
                 "transcribe_job": job_id}
        save_state(state)
    request.state.note += f" job={job_id}"
    log.info("markup %s: started by %s (job %s)", uid, who.name, job_id)
    return JSONResponse(_public(state), status_code=202)


@router.get("/{uid}")
def get_markup(uid: str, request: Request):
    uid = _check_uid(uid)
    sync(request.app.state.store, uid)
    path = _dir(f"{uid}.json")
    if not os.path.exists(path):
        raise HTTPException(404, "no markup for this lesson", headers=PUBLIC)
    return FileResponse(path, media_type="application/json", headers=PUBLIC)


@router.get("/{uid}/status")
def get_status(uid: str, request: Request):
    uid = _check_uid(uid)
    state = sync(request.app.state.store, uid)
    if state is None:
        raise HTTPException(404, "no markup run for this lesson", headers=PUBLIC)
    return JSONResponse(_public(state), headers=PUBLIC)
