"""
API server: jobs, clients, results. No GPU here: GPU workers (ai_trns.worker) pull jobs
through /worker/* (see ai_trns.protocol) and send back a words document, which is
rendered to the formats a client asked for.
"""
import json
import logging
import os
import secrets
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .. import audio, protocol
from ..subtitles import FORMATS, render
from . import config
from .auth import Principal, can_see, principal
from .jobs import CANCELLED, DONE, ERROR, QUEUED, RUNNING, JobStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns")

MEDIA_TYPES = {"txt": "text/plain; charset=utf-8", "srt": "application/x-subrip; charset=utf-8",
               "json": "application/json"}
UPLOAD = "upload"
# Workers not heard from for this long are dropped from /health
WORKER_SEEN_S = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.OIDC_ISSUER and not config.API_KEYS:
        log.warning("neither AI_TRNS_OIDC_ISSUER nor API keys are set: API is open without auth")
    if not config.WORKER_KEY:
        log.warning("AI_TRNS_WORKER_KEY is not set: no worker can connect")
    store = JobStore()
    stop = threading.Event()

    def maintenance():
        last_cleanup = 0.0
        while not stop.wait(timeout=protocol.HEARTBEAT_S):
            for job_id in store.expire_leases(config.MAX_ATTEMPTS):
                job = store.get(job_id)
                log.warning("job %s: lease expired, now %s", job_id, job["status"])
                if job["status"] == ERROR:
                    drop_upload(job_id)
            if time.time() - last_cleanup > 3600:
                last_cleanup = time.time()
                removed = store.cleanup()
                if removed:
                    log.info("cleanup: removed %d old jobs", removed)

    threading.Thread(target=maintenance, name="maintenance", daemon=True).start()
    app.state.store = store
    app.state.workers = {}
    yield
    stop.set()


app = FastAPI(title="ai-trns", lifespan=lifespan)
if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


def require_worker(request: Request):
    key = request.headers.get(protocol.KEY_HEADER, "")
    if not config.WORKER_KEY or not secrets.compare_digest(key, config.WORKER_KEY):
        raise HTTPException(401, "invalid or missing worker key")
    version = request.headers.get(protocol.VERSION_HEADER)
    if version != str(protocol.VERSION):
        raise HTTPException(409, f"worker protocol {version}, API speaks {protocol.VERSION}: update the worker")


def job_view(job: dict) -> dict:
    view = {k: job[k] for k in ("id", "status", "stage", "progress", "params", "error", "owner",
                                "worker", "attempts", "created", "started", "finished")}
    view["queue_position"] = app.state.store.queue_position(job["id"])
    return view


def get_job(job_id: str, who: Principal) -> dict:
    job = app.state.store.get(job_id)
    # someone else's job looks the same as a missing one
    if job is None or not can_see(who, job):
        raise HTTPException(404, "job not found")
    return job


def drop_upload(job_id: str):
    """The uploaded audio is only kept until the job is final"""
    try:
        os.remove(os.path.join(app.state.store.job_dir(job_id), UPLOAD))
    except FileNotFoundError:
        pass


def save_upload(file: UploadFile, dest: str):
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"file is larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)


@app.get("/health")
def health():
    now = time.time()
    workers = {w: {**info, "seen_s_ago": round(now - info["seen"])}
               for w, info in app.state.workers.items() if now - info["seen"] < WORKER_SEEN_S}
    for info in workers.values():
        del info["seen"]
    store = app.state.store
    return {"status": "ok", "queued": len(store.queued_ids()), "workers": workers}


# --- clients ---

@app.post("/jobs", status_code=202)
def create_job(
    who: Principal = Depends(principal),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    lang: Literal["en", "he", "ru", "auto"] = Form("he"),
    format: str = Form("srt", description="comma-separated: txt, srt, json (words with timecodes)"),
    profile: Optional[Literal["stable", "fast", "served"]] = Form(None, description="default: the model's own"),
    fix_rtl: bool = Form(False),
):
    if (file is None) == (not url):
        raise HTTPException(422, "pass exactly one of `file` or `url`")
    formats = sorted({f.strip() for f in format.split(",") if f.strip()})
    if not formats or any(f not in FORMATS for f in formats):
        raise HTTPException(422, f"format must be a comma-separated subset of {', '.join(FORMATS)}")
    store = app.state.store
    job_id = uuid.uuid4().hex
    job_dir = store.job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        if url:
            if not config.ALLOWED_URL_HOSTS:
                raise HTTPException(422, "URL input is disabled (AI_TRNS_ALLOWED_URL_HOSTS is empty)")
            try:
                audio.check_url(url, config.ALLOWED_URL_HOSTS)
            except ValueError as e:
                raise HTTPException(422, str(e))
            source = url
        else:
            save_upload(file, os.path.join(job_dir, UPLOAD))
            source = UPLOAD
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    params = {"lang": lang, "formats": formats, "profile": profile, "fix_rtl": fix_rtl}
    store.create(params, source, job_id, owner=who.owner)
    log.info("job %s created by %s", job_id, who.name)
    return job_view(store.get(job_id))


@app.get("/jobs")
def list_jobs(who: Principal = Depends(principal), limit: int = Query(50, ge=1, le=500)):
    """The caller's jobs, newest first (everyone's for admins and services)"""
    return [job_view(job) for job in app.state.store.list(None if who.sees_all else who.owner, limit)]


@app.get("/jobs/{job_id}")
def job_status(job_id: str, who: Principal = Depends(principal)):
    return job_view(get_job(job_id, who))


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str, format: Optional[Literal["txt", "srt", "json"]] = None,
               fix_rtl: Optional[bool] = None, who: Principal = Depends(principal)):
    """The result as rendered when the job finished, or rendered now from the job's words"""
    job = get_job(job_id, who)
    if job["status"] != DONE:
        raise HTTPException(409, f"job is {job['status']}")
    params = job["params"]
    fmt = format or params["formats"][0]
    job_dir = app.state.store.job_dir(job_id)
    path = os.path.join(job_dir, f"result.{fmt}")
    if os.path.exists(path) and (fix_rtl is None or fix_rtl == params["fix_rtl"]):
        return FileResponse(path, media_type=MEDIA_TYPES[fmt], filename=f"{job_id}.{fmt}")
    with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
        doc = json.load(f)
    content = render(doc, fmt, fix_rtl if fix_rtl is not None else params["fix_rtl"])
    return Response(content, media_type=MEDIA_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{job_id}.{fmt}"'})


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str, who: Principal = Depends(principal)):
    """A running job's worker learns about it on its next heartbeat"""
    job = get_job(job_id, who)
    if job["status"] not in (QUEUED, RUNNING):
        raise HTTPException(409, "job is already finished")
    app.state.store.update(job_id, status=CANCELLED, finished=time.time())
    drop_upload(job_id)
    return job_view(get_job(job_id, who))


@app.post("/render", dependencies=[Depends(principal)])
def render_words(doc: dict = Body(..., description="words document, as GET /jobs/{id}/result?format=json returns"),
                 format: Literal["txt", "srt", "json"] = "srt",
                 fix_rtl: bool = False):
    """Words -> SRT / txt right away, without a job or a GPU"""
    try:
        content = render(doc, format, fix_rtl)
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        raise HTTPException(422, f"bad words document: {e}")
    return Response(content, media_type=MEDIA_TYPES[format])


# --- GPU workers (ai_trns.protocol) ---

def seen(worker: str, **info):
    app.state.workers.setdefault(worker, {}).update(info, seen=time.time())


def owned_job(job_id: str, worker: str) -> dict:
    job = app.state.store.owned(job_id, worker)
    if job is None:
        raise HTTPException(409, "job is not running on this worker")
    return job


@app.post("/worker/lease", dependencies=[Depends(require_worker)])
def worker_lease(body: dict = Body(...)):
    worker, langs = str(body["worker"]), list(body["langs"])
    seen(worker, langs=langs, job=None)
    job = app.state.store.lease(worker, langs, protocol.LEASE_S)
    if job is None:
        return Response(status_code=204)
    seen(worker, job=job["id"])
    log.info("job %s leased to %s (attempt %d)", job["id"], worker, job["attempts"])
    source = {"upload": f"/worker/jobs/{job['id']}/audio"} if job["source"] == UPLOAD else {"url": job["source"]}
    return {"id": job["id"], "params": job["params"], "source": source, "lease_s": protocol.LEASE_S}


@app.get("/worker/jobs/{job_id}/audio", dependencies=[Depends(require_worker)])
def worker_audio(job_id: str, worker: str):
    owned_job(job_id, worker)
    path = os.path.join(app.state.store.job_dir(job_id), UPLOAD)
    if not os.path.exists(path):
        raise HTTPException(404, "job has no uploaded audio")
    return FileResponse(path, media_type="application/octet-stream")


@app.post("/worker/jobs/{job_id}/heartbeat", dependencies=[Depends(require_worker)])
def worker_heartbeat(job_id: str, body: dict = Body(...)):
    worker = str(body["worker"])
    seen(worker, job=job_id)
    alive = app.state.store.heartbeat(job_id, worker, body.get("stage"), float(body.get("progress", 0)),
                                      protocol.LEASE_S)
    return {"cancel": not alive}


@app.put("/worker/jobs/{job_id}/result", dependencies=[Depends(require_worker)])
def worker_result(job_id: str, worker: str, doc: dict = Body(...)):
    job = owned_job(job_id, worker)
    seen(worker, job=None)
    store, params = app.state.store, job["params"]
    job_dir = store.job_dir(job_id)
    try:
        # result.json is the job's real result: other formats can be rendered from it later
        for fmt in ["json", *params["formats"]]:
            content = render(doc, fmt, params["fix_rtl"])
            with open(os.path.join(job_dir, f"result.{fmt}"), "w", encoding="utf-8") as f:
                f.write(content)
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        log.exception("job %s: bad words document from %s", job_id, worker)
        store.update(job_id, status=ERROR, error=f"bad words document: {e}", finished=time.time())
        drop_upload(job_id)
        raise HTTPException(422, f"bad words document: {e}")
    store.update(job_id, status=DONE, stage=None, progress=1.0, finished=time.time())
    drop_upload(job_id)
    log.info("job %s done by %s", job_id, worker)
    return {"ok": True}


@app.post("/worker/jobs/{job_id}/fail", dependencies=[Depends(require_worker)])
def worker_fail(job_id: str, body: dict = Body(...)):
    worker = str(body["worker"])
    owned_job(job_id, worker)
    seen(worker, job=None)
    app.state.store.update(job_id, status=ERROR, error=str(body.get("error", "worker error")),
                           finished=time.time())
    drop_upload(job_id)
    log.warning("job %s failed on %s: %s", job_id, worker, body.get("error"))
    return {"ok": True}
