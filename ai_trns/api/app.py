"""
API server: jobs, clients, results. No GPU here: GPU workers (ai_trns.worker) pull jobs
through /worker/* (see ai_trns.protocol) and send back a words document (or a diarization
document), which is rendered to the formats a client asked for (ai_trns.api.results).
"""
import json
import logging
import os
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .. import protocol
from ..subtitles import render
from . import config, diarize, live, results, review, tadiff
from .auth import Principal, can_see, principal
from .jobs import CANCELLED, DONE, ERROR, QUEUED, RUNNING, TRANSCRIBE, JobStore, kind_of
from .submit import UPLOAD, create_job as submit_job, job_view as view_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns")

# Job inputs a worker may download (GET /worker/jobs/{id}/files/{name})
WORKER_FILE_RE = re.compile(r"^(cues\.json|sample\d+\.[a-z0-9]{1,5})$")
# Workers not heard from for this long are dropped from /health
WORKER_SEEN_S = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.OIDC_ISSUER and not config.API_KEYS:
        log.warning("neither AI_TRNS_OIDC_ISSUER nor API keys are set: API is open without auth")
    if not config.WORKER_KEY:
        log.warning("AI_TRNS_WORKER_KEY is not set: no worker can connect")
    store = JobStore()
    hubs = live.Live()
    stop = threading.Event()

    def maintenance():
        last_cleanup = 0.0
        while not stop.wait(timeout=protocol.HEARTBEAT_S):
            for job_id in store.expire_leases(config.MAX_ATTEMPTS):
                job = store.get(job_id)
                log.warning("job %s: lease expired, now %s", job_id, job["status"])
                if job["status"] == ERROR:
                    drop_upload(job_id)
                    hubs.publish(job_id, {"type": "error", "message": job["error"]})
            try:
                review.sync_all(store)
            except Exception:
                log.exception("review: sync failed")
            hubs.cleanup(store)
            if time.time() - last_cleanup > 3600:
                last_cleanup = time.time()
                removed = store.cleanup()
                if removed:
                    log.info("cleanup: removed %d old jobs", removed)

    threading.Thread(target=maintenance, name="maintenance", daemon=True).start()
    app.state.store = store
    app.state.live = hubs
    app.state.workers = {}
    yield
    stop.set()


app = FastAPI(title="ai-trns", lifespan=lifespan)
if config.CORS_ORIGINS:
    # exposed: the result's file name and the X-<Kind>-* counters of tadiff / docdiff / diarize results
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["*"],
                       expose_headers=["Content-Disposition", *results.STATS_HEADERS])
app.include_router(review.router)
app.include_router(tadiff.tadiff)
app.include_router(tadiff.docdiff)
app.include_router(diarize.router)
app.include_router(live.router)


def require_worker(request: Request):
    key = request.headers.get(protocol.KEY_HEADER, "")
    if not config.WORKER_KEY or not secrets.compare_digest(key, config.WORKER_KEY):
        raise HTTPException(401, "invalid or missing worker key")
    version = request.headers.get(protocol.VERSION_HEADER)
    if version != str(protocol.VERSION):
        raise HTTPException(409, f"worker protocol {version}, API speaks {protocol.VERSION}: update the worker")


def job_view(job: dict) -> dict:
    return view_of(app.state.store, job)


def get_job(job_id: str, who: Principal, shared: bool = False) -> dict:
    """shared: a broadcast's live job is anyone's to read (ai_trns.api.live), not to cancel"""
    job = app.state.store.get(job_id)
    # someone else's job looks the same as a missing one
    if job is None or not (can_see(who, job) or (shared and live.is_shared(job))):
        raise HTTPException(404, "job not found")
    return job


def drop_upload(job_id: str):
    """The uploaded audio is only kept until the job is final"""
    try:
        os.remove(os.path.join(app.state.store.job_dir(job_id), UPLOAD))
    except FileNotFoundError:
        pass


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
    params = {"lang": lang, "formats": results.parse_formats(format, "transcribe"), "profile": profile,
              "fix_rtl": fix_rtl}
    job_id = submit_job(app.state.store, who, params, file, url)
    return job_view(app.state.store.get(job_id))


@app.get("/jobs")
def list_jobs(who: Principal = Depends(principal), limit: int = Query(50, ge=1, le=500)):
    """The caller's jobs, newest first (everyone's for admins and services)"""
    return [job_view(job) for job in app.state.store.list(None if who.sees_all else who.owner, limit)]


@app.get("/jobs/{job_id}")
def job_status(job_id: str, who: Principal = Depends(principal)):
    return job_view(get_job(job_id, who, shared=True))


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str, format: Optional[str] = Query(None, description="default: the first one asked for; "
                                                          "txt, srt, json, and diff / docx for tadiff / docdiff jobs"),
               fix_rtl: Optional[bool] = None, who: Principal = Depends(principal)):
    """The result as rendered when the job finished, or rendered now from the worker's document.
    The job's stats come as X-<Kind>-* headers too"""
    job = get_job(job_id, who, shared=True)
    if job["status"] != DONE:
        raise HTTPException(409, f"job is {job['status']}")
    params = job["params"]
    fmt = format or params["formats"][0]
    if fmt not in results.formats(params):
        raise HTTPException(422, f"a {kind_of(params)} job has formats {', '.join(results.formats(params))}")
    headers = {"Content-Disposition": results.content_disposition(results.download_name(job, fmt)),
               **results.stats_headers(job)}
    job_dir = app.state.store.job_dir(job_id)
    path = os.path.join(job_dir, results.file_name(fmt))
    if os.path.exists(path) and (fix_rtl is None or fix_rtl == params.get("fix_rtl", False)):
        return FileResponse(path, media_type=results.MEDIA_TYPES[fmt], headers=headers)
    with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
        doc = json.load(f)
    content, _ = results.render(job_dir, params, doc, fmt, fix_rtl)
    return Response(content, media_type=results.MEDIA_TYPES[fmt], headers=headers)


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str, who: Principal = Depends(principal)):
    """A running job's worker learns about it on its next heartbeat"""
    job = get_job(job_id, who)
    if job["status"] not in (QUEUED, RUNNING):
        raise HTTPException(409, "job is already finished")
    app.state.store.update(job_id, status=CANCELLED, finished=time.time())
    drop_upload(job_id)
    app.state.live.publish(job_id, {"type": "error", "message": "cancelled"})
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
    return Response(content, media_type=results.MEDIA_TYPES[format])


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
    engines = list(body.get("engines") or [TRANSCRIBE])
    seen(worker, langs=langs, engines=engines, job=None)
    job = app.state.store.lease(worker, langs, protocol.LEASE_S, engines)
    if job is None:
        return Response(status_code=204)
    seen(worker, job=job["id"])
    log.info("job %s leased to %s (attempt %d)", job["id"], worker, job["attempts"])
    if live.is_live(job):
        app.state.live.leased(job["id"])
    source = {"upload": f"/worker/jobs/{job['id']}/audio"} if job["source"] == UPLOAD else {"url": job["source"]}
    return {"id": job["id"], "params": job["params"], "source": source, "lease_s": protocol.LEASE_S}


@app.get("/worker/jobs/{job_id}/audio", dependencies=[Depends(require_worker)])
def worker_audio(job_id: str, worker: str):
    owned_job(job_id, worker)
    path = os.path.join(app.state.store.job_dir(job_id), UPLOAD)
    if not os.path.exists(path):
        raise HTTPException(404, "job has no uploaded audio")
    return FileResponse(path, media_type="application/octet-stream")


@app.get("/worker/jobs/{job_id}/files/{name}", dependencies=[Depends(require_worker)])
def worker_file(job_id: str, name: str, worker: str):
    """Another input of the job: a diarize job's cues.json and voice samples"""
    owned_job(job_id, worker)
    path = os.path.join(app.state.store.job_dir(job_id), name)
    if not WORKER_FILE_RE.match(name) or not os.path.exists(path):
        raise HTTPException(404, "job has no such file")
    return FileResponse(path, media_type="application/octet-stream")


@app.post("/worker/jobs/{job_id}/heartbeat", dependencies=[Depends(require_worker)])
def worker_heartbeat(job_id: str, body: dict = Body(...)):
    worker = str(body["worker"])
    seen(worker, job=job_id)
    alive = app.state.store.heartbeat(job_id, worker, body.get("stage"), float(body.get("progress", 0)),
                                      protocol.LEASE_S)
    if alive and job_id in app.state.live.stopping:
        return {"cancel": False, "stop": True}
    return {"cancel": not alive}


@app.post("/worker/jobs/{job_id}/events", dependencies=[Depends(require_worker)])
def worker_events(job_id: str, body: dict = Body(...)):
    """A live job's start / final / partial events, passed on to its listeners"""
    owned_job(job_id, str(body["worker"]))
    hubs = app.state.live
    for event in body.get("events", []):
        if event.get("type") in ("start", "final", "partial"):
            hubs.publish(job_id, event)
    return {"stop": job_id in hubs.stopping}


@app.put("/worker/jobs/{job_id}/result", dependencies=[Depends(require_worker)])
def worker_result(job_id: str, worker: str, doc: dict = Body(...)):
    job = owned_job(job_id, worker)
    seen(worker, job=None)
    store, params = app.state.store, job["params"]
    job_dir = store.job_dir(job_id)
    stats = {}
    try:
        # result.json is the job's real result: other formats can be rendered from it later
        for fmt in ["json", *params["formats"]]:
            content, fmt_stats = results.render(job_dir, params, doc, fmt)
            stats.update(fmt_stats or {})
            with open(os.path.join(job_dir, results.file_name(fmt)), "wb") as f:
                f.write(content if isinstance(content, bytes) else content.encode("utf-8"))
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        log.exception("job %s: bad document from %s", job_id, worker)
        store.update(job_id, status=ERROR, error=f"bad document: {e}", finished=time.time())
        drop_upload(job_id)
        app.state.live.publish(job_id, {"type": "error", "message": f"bad document: {e}"})
        raise HTTPException(422, f"bad document: {e}")
    store.update(job_id, status=DONE, stage=None, progress=1.0, finished=time.time(),
                 stats=json.dumps(stats) if stats else None)
    drop_upload(job_id)
    app.state.live.publish(job_id, {"type": "end", "duration_s": doc.get("duration_s")})
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
    app.state.live.publish(job_id, {"type": "error", "message": str(body.get("error", "worker error"))})
    log.warning("job %s failed on %s: %s", job_id, worker, body.get("error"))
    return {"ok": True}
