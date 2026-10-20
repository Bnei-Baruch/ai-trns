import json
import logging
import os
import secrets
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import Body, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from .. import audio
from ..subtitles import FORMATS, render
from . import config
from .jobs import DONE, JobStore
from .worker import ModelCache, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns")

MEDIA_TYPES = {"txt": "text/plain; charset=utf-8", "srt": "application/x-subrip; charset=utf-8",
               "json": "application/json"}


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
    format: str = Form("srt", description="comma-separated: txt, srt, json (words with timecodes)"),
    srt_style: Literal["house", "advanced", "standard"] = Form("house"),
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
            source = os.path.abspath(os.path.join(job_dir, "upload"))
            save_upload(file, source)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    params = {"lang": lang, "formats": formats, "srt_style": srt_style, "profile": profile, "fix_rtl": fix_rtl}
    store.create(params, source, job_id)
    app.state.worker.notify()
    return job_view(store.get(job_id))


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def job_status(job_id: str):
    return job_view(get_job(job_id))


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_key)])
def job_result(job_id: str, format: Optional[Literal["txt", "srt", "json"]] = None,
               srt_style: Optional[Literal["house", "advanced", "standard"]] = None,
               fix_rtl: Optional[bool] = None):
    """The result as rendered at job time, or rendered now from the job's words in another way"""
    job = get_job(job_id)
    if job["status"] != DONE:
        raise HTTPException(409, f"job is {job['status']}")
    params = job["params"]
    fmt = format or params["formats"][0]
    job_dir = app.state.store.job_dir(job_id)
    path = os.path.join(job_dir, f"result.{fmt}")
    rerender = ((srt_style or params["srt_style"]) != params["srt_style"]
                or (fix_rtl if fix_rtl is not None else params["fix_rtl"]) != params["fix_rtl"])
    if os.path.exists(path) and not rerender:
        return FileResponse(path, media_type=MEDIA_TYPES[fmt], filename=f"{job_id}.{fmt}")
    with open(os.path.join(job_dir, "result.json"), encoding="utf-8") as f:
        doc = json.load(f)
    content = render(doc, fmt, srt_style or params["srt_style"],
                     fix_rtl if fix_rtl is not None else params["fix_rtl"])
    return Response(content, media_type=MEDIA_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{job_id}.{fmt}"'})


@app.post("/render", dependencies=[Depends(require_key)])
def render_words(doc: dict = Body(..., description="words document, as GET /jobs/{id}/result?format=json returns"),
                 format: Literal["txt", "srt", "json"] = "srt",
                 srt_style: Literal["house", "advanced", "standard"] = "house",
                 fix_rtl: bool = False):
    """Words -> SRT / txt right away, without a job or the GPU"""
    try:
        content = render(doc, format, srt_style, fix_rtl)
    except (KeyError, TypeError, ValueError, RuntimeError) as e:
        raise HTTPException(422, f"bad words document: {e}")
    return Response(content, media_type=MEDIA_TYPES[format])


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_key)])
def cancel_job(job_id: str):
    get_job(job_id)
    if not app.state.worker.cancel(job_id):
        raise HTTPException(409, "job is already finished")
    return job_view(get_job(job_id))
