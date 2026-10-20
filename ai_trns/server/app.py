import logging
import os
import secrets
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .. import audio
from . import config
from .jobs import DONE, JobStore
from .worker import ModelCache, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns")

MEDIA_TYPES = {"txt": "text/plain; charset=utf-8", "srt": "application/x-subrip; charset=utf-8"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.API_KEY:
        log.warning("AI_TRNS_API_KEY is not set: API is open without auth")
    store = JobStore()
    store.recover()
    models = ModelCache()
    for lang in config.PRELOAD_LANGS:
        models.get(lang)
    worker = Worker(store, models)
    worker.start()

    stop_cleanup = threading.Event()

    def cleanup_loop():
        while not stop_cleanup.wait(timeout=3600):
            removed = store.cleanup()
            if removed:
                log.info("cleanup: removed %d old jobs", removed)

    threading.Thread(target=cleanup_loop, name="cleanup", daemon=True).start()
    app.state.store, app.state.models, app.state.worker = store, models, worker
    yield
    stop_cleanup.set()
    worker.stop()


app = FastAPI(title="ai-trns", lifespan=lifespan)
if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


def require_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    if not config.API_KEY:
        return
    token = x_api_key
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if not token or not secrets.compare_digest(token, config.API_KEY):
        raise HTTPException(401, "invalid or missing API key")


def job_view(job: dict) -> dict:
    view = {k: job[k] for k in ("id", "status", "stage", "progress", "params", "error",
                                "created", "started", "finished")}
    view["queue_position"] = app.state.store.queue_position(job["id"])
    return view


def get_job(job_id: str) -> dict:
    job = app.state.store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


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
    worker = app.state.worker
    return {
        "status": "ok",
        "worker_alive": worker.thread.is_alive(),
        "current_job": worker.current,
        "queued": len(app.state.store.queued_ids()),
        "models_loaded": app.state.models.loaded(),
    }


@app.post("/jobs", status_code=202, dependencies=[Depends(require_key)])
def create_job(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    lang: Literal["en", "he", "ru", "auto"] = Form("he"),
    format: str = Form("srt", description="txt, srt or both: srt,txt"),
    impl: Literal[1, 2] = Form(2),
    profile: Literal["stable", "fast"] = Form("stable"),
    fix_rtl: bool = Form(False),
):
    if (file is None) == (not url):
        raise HTTPException(422, "pass exactly one of `file` or `url`")
    formats = sorted({f.strip() for f in format.split(",") if f.strip()})
    if not formats or any(f not in MEDIA_TYPES for f in formats):
        raise HTTPException(422, "format must be txt, srt or srt,txt")
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
            source = os.path.abspath(os.path.join(job_dir, "upload"))
            save_upload(file, source)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    params = {"lang": lang, "formats": formats, "impl": impl, "profile": profile, "fix_rtl": fix_rtl}
    store.create(params, source, job_id)
    app.state.worker.notify()
    return job_view(store.get(job_id))


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str):
    return job_view(get_job(job_id))


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_key)])
def job_result(job_id: str, format: Optional[str] = None):
    job = get_job(job_id)
    if job["status"] != DONE:
        raise HTTPException(409, f"job is {job['status']}")
    formats = job["params"]["formats"]
    fmt = format or formats[0]
    if fmt not in formats:
        raise HTTPException(404, f"job has no {fmt} result, available: {', '.join(formats)}")
    path = os.path.join(app.state.store.job_dir(job_id), f"result.{fmt}")
    return FileResponse(path, media_type=MEDIA_TYPES[fmt], filename=f"{job_id}.{fmt}")


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_key)])
def cancel_job(job_id: str):
    get_job(job_id)
    if not app.state.worker.cancel(job_id):
        raise HTTPException(409, "job is already finished")
    return job_view(get_job(job_id))
