"""
Human review of STT quality (moved from stt-api /review/*): sets of whole lessons, each
transcribed by an ai-trns job, and per-reviewer results.

Reviewers add lessons by kabbalahmedia unit + file ids; the API builds the CDN URL itself and
queues a job for it. A lesson stays "pending" until its job finishes, then gets the job's cues
as "chunks" and the model that made them ("ready"), or the job's error ("failed").
Pending lessons are brought up to date on every read and by the maintenance thread (sync_all).

The reviewer is the caller: a user acts only as themselves (Keycloak sub), a service key may
name any reviewer. Anyone can delete a lesson: it is only marked deleted, so a mistake can be
undone by hand.

Storage (AI_TRNS_REVIEW_DIR), the same files stt-api wrote, so its data can be copied over:
  sets/{set_id}/set.json                          -- {id, created_at}
  sets/{set_id}/segments/{segment_id}.json        -- one lesson; status pending -> ready | failed
  results/{set_id}/{segment_id}/{reviewer}.json   -- one per reviewer and segment
Ids are checked against ID_RE before they become part of a path.
"""
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .. import cues
from . import config
from .auth import Principal, principal
from .jobs import CANCELLED, DONE, FINAL, JobStore

log = logging.getLogger("ai-trns.review")

router = APIRouter(prefix="/review", tags=["review"])

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
KMEDIA_UNIT_URL = "https://kabbalahmedia.info/backend/content_units/{uid}"
KMEDIA_FILE_URL = "https://cdn.kabbalahmedia.info/{file_id}"
_KMEDIA_ID = r"^[A-Za-z0-9]{8}$"
# not shown to reviewers: they must not see the other model (sets from stt-api may have them)
_BASELINE_KEYS = ("baseline_model", "baseline_chunks", "baseline_error")
# what a lesson is transcribed with: words only, cues are built here
JOB_PARAMS = {"lang": "he", "formats": ["json"], "profile": None, "fix_rtl": False}

# serializes read-modify-write of set and segment files (API threads + maintenance)
lock = threading.RLock()


# --- storage

def now(timespec: str = "seconds") -> str:
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def _read(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, data) -> None:
    """Temp file in the same dir + os.replace, so a reader never sees a half-written file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_files(d: str) -> list:
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if n.endswith(".json") and not n.startswith("."))


def _path(*parts: str) -> str:
    return os.path.join(config.REVIEW_DIR, *parts)


def set_ids() -> list:
    root = _path("sets")
    if not os.path.isdir(root):
        return []
    return sorted(n for n in os.listdir(root) if os.path.isfile(os.path.join(root, n, "set.json")))


def load_segments(set_id: str) -> list:
    """Segments in the order they were added"""
    seg_dir = _path("sets", set_id, "segments")
    segments = [_read(os.path.join(seg_dir, n)) for n in _json_files(seg_dir)]
    segments.sort(key=lambda s: (s["created_at"], s["id"]))
    return segments


def load_segment(set_id: str, segment_id: str) -> Optional[dict]:
    path = _path("sets", set_id, "segments", f"{segment_id}.json")
    return _read(path) if os.path.isfile(path) else None


def save_segment(set_id: str, segment: dict) -> None:
    _write(_path("sets", set_id, "segments", f"{segment['id']}.json"), segment)


def result_path(set_id: str, segment_id: str, reviewer: str) -> str:
    return _path("results", set_id, segment_id, f"{reviewer}.json")


def load_result(set_id: str, segment_id: str, reviewer: str) -> Optional[dict]:
    path = result_path(set_id, segment_id, reviewer)
    return _read(path) if os.path.isfile(path) else None


def load_segment_results(set_id: str, segment_id: str) -> list:
    d = _path("results", set_id, segment_id)
    return [_read(os.path.join(d, n)) for n in _json_files(d)]


# --- transcription by jobs

def chunks_from_words(doc: dict) -> list:
    """Words document -> [{start, end, text}]: the same cues as the job's SRT"""
    built, _policy = cues.build_cues(doc)
    chunks = ({"start": c["start_ms"] / 1000, "end": c["end_ms"] / 1000, "text": " ".join(c["text"].split())}
              for c in built)
    return [c for c in chunks if c["text"]]


def _job_outcome(store: JobStore, segment: dict) -> Optional[dict]:
    """Fields that finish a pending lesson, or None while its job is still queued / running"""
    job = store.get(segment["job_id"]) if segment.get("job_id") else None
    if job is None:
        return {"status": "failed", "error": "transcription job is lost"}
    if job["status"] not in FINAL:
        return None
    if job["status"] != DONE:
        return {"status": "failed", "error": job["error"] or job["status"]}
    try:
        doc = _read(os.path.join(store.job_dir(job["id"]), "result.json"))
        chunks = chunks_from_words(doc)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        log.exception("review: cannot read the result of job %s", job["id"])
        return {"status": "failed", "error": f"bad transcription result: {e}"}
    if not chunks:
        return {"status": "failed", "error": "no speech recognized"}
    return {"status": "ready", "chunks": chunks, "end": max(segment["end"], chunks[-1]["end"]),
            "model": doc.get("model"), "transcribed_at": now()}


def sync(store: JobStore, set_id: str, segment: dict) -> dict:
    """The segment with its job's outcome written in, if the job has finished"""
    if segment.get("status") != "pending" or segment.get("deleted"):
        return segment
    with lock:
        segment = load_segment(set_id, segment["id"])
        if segment.get("status") != "pending" or segment.get("deleted"):
            return segment
        outcome = _job_outcome(store, segment)
        if outcome is None:
            return segment
        segment = {k: v for k, v in segment.items() if k != "error"}
        segment.update(outcome)
        save_segment(set_id, segment)
    log.info("review: %s/%s %s", set_id, segment["id"], segment["status"])
    return segment


def sync_all(store: JobStore) -> None:
    """Finish pending lessons before their jobs are removed by the TTL cleanup"""
    for set_id in set_ids():
        for segment in load_segments(set_id):
            sync(store, set_id, segment)


def fetch_unit(uid: str) -> dict:
    try:
        with urllib.request.urlopen(KMEDIA_UNIT_URL.format(uid=uid), timeout=15) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(uid)
        raise


# --- requests

class NewLesson(BaseModel):
    unit_uid: str = Field(pattern=_KMEDIA_ID)
    title: str = Field(min_length=1, max_length=300)
    audio_file_id: str = Field(pattern=_KMEDIA_ID)
    video_file_id: Optional[str] = Field(default=None, pattern=_KMEDIA_ID)
    duration: Optional[float] = Field(default=None, ge=0)  # informational; the backend's value wins
    reviewer: Optional[str] = None
    reviewer_name: str = ""


class ChunkResult(BaseModel):
    index: int = Field(ge=0)
    corrected: str
    category: Optional[Literal["A", "B", "C"]] = None
    comment: str = ""


class SegmentResult(BaseModel):
    reviewer_name: str = ""
    model: Optional[str] = None  # the model the client believes it reviews; the lesson's "model" is authoritative
    status: Literal["in_progress", "done"]
    chunks: list[ChunkResult] = []
    position: Optional[float] = None
    active_seconds: float = Field(default=0, ge=0)


def _validate(model, body: dict):
    """Bad input is a 400, as in stt-api"""
    try:
        return model.model_validate(body)
    except ValidationError as e:
        raise HTTPException(400, e.errors(include_url=False, include_context=False))


def _check_id(value: str, what: str) -> str:
    if not ID_RE.match(value):
        raise HTTPException(400, f"invalid {what}")
    return value


def _reviewer(who: Principal, claimed: Optional[str]) -> str:
    """A user is the reviewer; a service names one"""
    if who.owner.startswith("user:"):
        sub = who.owner[len("user:"):]
        if claimed and claimed != sub:
            raise HTTPException(403, "a reviewer can only act as themselves")
        claimed = sub
    if not claimed:
        raise HTTPException(400, "reviewer is required")
    return _check_id(claimed, "reviewer")


def _load_set(request: Request, set_id: str) -> dict:
    """{id, model, segments: [...]} without deleted segments; a set that doesn't exist yet is empty.
    "model": AI_TRNS_REVIEW_MODEL, or the model of the newest transcribed lesson"""
    _check_id(set_id, "set_id")
    store = request.app.state.store
    meta_path = _path("sets", set_id, "set.json")
    meta = _read(meta_path) if os.path.isfile(meta_path) else {"id": set_id}
    segments = [sync(store, set_id, s) for s in load_segments(set_id) if not s.get("deleted")]
    models = [s["model"] for s in segments if s.get("status", "ready") == "ready" and s.get("model")]
    model = config.REVIEW_MODEL or (models[-1] if models else None) or meta.get("model")
    return {**meta, "model": model, "segments": segments}


def _live_segment(request: Request, set_id: str, segment_id: str) -> dict:
    _check_id(set_id, "set_id")
    _check_id(segment_id, "segment_id")
    segment = load_segment(set_id, segment_id)
    if segment is None or segment.get("deleted"):
        raise HTTPException(404, "segment not found")
    return sync(request.app.state.store, set_id, segment)


def _public(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in _BASELINE_KEYS}


# GET /review/sets/{set_id}
# -> {"id", "model", "segments": [segment, ...]} without baseline_* and deleted segments, in the order added.
# segment: {"id" (= unit_uid), "unit_uid", "title", "media_url", "video_url"?, "start", "end",
#           "chunks": [{"start", "end", "text"}], "status": "pending"|"ready"|"failed", "error"?,
#           "job_id"?, "model"?, "transcribed_at"?, "added_by", "added_by_name", "created_at"}
@router.get("/sets/{set_id}")
def get_set(set_id: str, request: Request, who: Principal = Depends(principal)):
    review_set = _load_set(request, set_id)
    return {**_public(review_set), "segments": [_public(s) for s in review_set["segments"]]}


# POST /review/sets/{set_id}/segments
# Body: {"unit_uid", "title", "audio_file_id", "video_file_id"?, "duration"?, "reviewer_name"}
# -> 202 + segment, status "pending": a job transcribes the whole lesson.
# A failed or deleted lesson is replaced. 409: the lesson is already in the set.
@router.post("/sets/{set_id}/segments", status_code=202)
def add_lesson(set_id: str, request: Request, body: dict = Body(...), who: Principal = Depends(principal)):
    _check_id(set_id, "set_id")
    lesson = _validate(NewLesson, body)
    reviewer = _reviewer(who, lesson.reviewer)
    try:
        unit = fetch_unit(lesson.unit_uid)
    except LookupError:
        raise HTTPException(400, f"unknown unit {lesson.unit_uid}")
    except Exception:
        log.exception("review: kabbalahmedia lookup failed: %s", lesson.unit_uid)
        raise HTTPException(502, "kabbalahmedia backend unavailable")
    files = {f["id"]: f for f in unit.get("files") or []}
    audio = files.get(lesson.audio_file_id)
    if audio is None or audio.get("type") != "audio":
        raise HTTPException(400, "audio_file_id is not an audio file of this unit")
    if lesson.video_file_id and files.get(lesson.video_file_id, {}).get("type") != "video":
        raise HTTPException(400, "video_file_id is not a video file of this unit")

    store = request.app.state.store
    media_url = KMEDIA_FILE_URL.format(file_id=lesson.audio_file_id)
    with lock:
        existing = load_segment(set_id, lesson.unit_uid)
        if existing is not None and not existing.get("deleted") and existing.get("status") != "failed":
            raise HTTPException(409, "this lesson is already in the set")
        meta_path = _path("sets", set_id, "set.json")
        if not os.path.isfile(meta_path):
            _write(meta_path, {"id": set_id, "created_at": now()})
        # the URL is built here from a file of the unit, so it needs no AI_TRNS_ALLOWED_URL_HOSTS
        job_id = store.create(JOB_PARAMS, media_url, owner=who.owner)
        segment = {
            "id": lesson.unit_uid,
            "unit_uid": lesson.unit_uid,
            "title": lesson.title,
            "media_url": media_url,
            **({"video_url": KMEDIA_FILE_URL.format(file_id=lesson.video_file_id)} if lesson.video_file_id else {}),
            "start": 0.0,
            "end": float(audio.get("duration") or unit.get("duration") or lesson.duration or 0),
            "chunks": [],
            "status": "pending",
            "job_id": job_id,
            "added_by": reviewer,
            "added_by_name": lesson.reviewer_name,
            "created_at": now("microseconds"),  # orders the set's segments
        }
        save_segment(set_id, segment)
    log.info("review: %s/%s added by %s, job %s", set_id, segment["id"], who.name, job_id)
    return JSONResponse(segment, status_code=202)


# DELETE /review/sets/{set_id}/segments/{segment_id}
# Soft delete: the segment file is kept with "deleted", "deleted_by", "deleted_at", and so are
# its results; it disappears from both GETs. Its job is cancelled if still running. 404 if missing.
@router.delete("/sets/{set_id}/segments/{segment_id}", status_code=204)
def delete_lesson(set_id: str, segment_id: str, request: Request, reviewer: Optional[str] = None,
                  who: Principal = Depends(principal)):
    reviewer = _reviewer(who, reviewer)
    store = request.app.state.store
    with lock:
        segment = _live_segment(request, set_id, segment_id)
        save_segment(set_id, {**segment, "deleted": True, "deleted_by": reviewer, "deleted_at": now()})
    job = store.get(segment["job_id"]) if segment.get("job_id") else None
    if job and job["status"] not in FINAL:
        store.update(job["id"], status=CANCELLED, finished=time.time())
    return Response(status_code=204)


# GET /review/sets/{set_id}/reviewers/{reviewer}/results
# -> [result, ...]: the reviewer's saved results in this set ([] if none)
@router.get("/sets/{set_id}/reviewers/{reviewer}/results")
def get_reviewer_results(set_id: str, reviewer: str, request: Request, who: Principal = Depends(principal)):
    reviewer = _reviewer(who, reviewer)
    results = (load_result(set_id, s["id"], reviewer) for s in _load_set(request, set_id)["segments"])
    return [r for r in results if r is not None]


# PUT /review/sets/{set_id}/reviewers/{reviewer}/segments/{segment_id}
# Body: {"reviewer_name", "model", "status": "in_progress"|"done",
#        "chunks": [{"index", "corrected", "category": "A"|"B"|"C"|null, "comment"}],
#        "position", "active_seconds"}
# Replaces the stored result as a whole; the server sets reviewer and updated_at.
# With status "done" every chunk whose text was changed must have a category.
# 404 for a deleted segment, 409 while it is not transcribed yet.
@router.put("/sets/{set_id}/reviewers/{reviewer}/segments/{segment_id}")
def put_segment_result(set_id: str, reviewer: str, segment_id: str, request: Request,
                       body: dict = Body(...), who: Principal = Depends(principal)):
    reviewer = _reviewer(who, reviewer)
    segment = _live_segment(request, set_id, segment_id)
    if segment.get("status", "ready") != "ready":
        raise HTTPException(409, f"segment is {segment['status']}, not ready for review")

    result = _validate(SegmentResult, body)
    source = segment.get("chunks", [])
    seen = set()
    for c in result.chunks:
        if c.index >= len(source):
            raise HTTPException(400, f"chunk index {c.index} out of range")
        if c.index in seen:
            raise HTTPException(400, f"duplicate chunk index {c.index}")
        seen.add(c.index)
        changed = c.corrected.strip() != source[c.index]["text"].strip()
        if result.status == "done" and changed and c.category is None:
            raise HTTPException(400, f"chunk {c.index} is changed but has no category")

    stored = {"set_id": set_id, "segment_id": segment_id, "reviewer": reviewer,
              **result.model_dump(), "updated_at": now()}
    _write(result_path(set_id, segment_id, reviewer), stored)
    return stored


# GET /review/sets/{set_id}/results
# -> {"set": <full set incl. baseline_* and pending/failed segments>, "results": [result, ...]}
# Deleted segments and their results are left out.
@router.get("/sets/{set_id}/results")
def get_all_results(set_id: str, request: Request, who: Principal = Depends(principal)):
    review_set = _load_set(request, set_id)
    results = [r for s in review_set["segments"] for r in load_segment_results(set_id, s["id"])]
    return {"set": review_set, "results": results}
