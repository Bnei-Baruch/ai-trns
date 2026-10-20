"""Creating jobs from client input: an uploaded file or a URL, plus the inputs a job kind needs"""
import logging
import os
import shutil
import uuid
from typing import Optional

from fastapi import HTTPException, UploadFile

from .. import audio
from . import config
from .auth import Principal
from .jobs import JobStore

log = logging.getLogger("ai-trns")

UPLOAD = "upload"


def save_upload(file: UploadFile, dest: str):
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with open(dest, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(413, f"file is larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)


def create_job(store: JobStore, who: Principal, params: dict, file: Optional[UploadFile], url: Optional[str],
               inputs: Optional[dict] = None, uploads: Optional[dict] = None) -> str:
    """A queued job for the audio in `file` or at `url` (exactly one). inputs: {name: bytes} and
    uploads: {name: UploadFile} are saved in the job dir for the worker or the result renderer"""
    if (file is None) == (not url):
        raise HTTPException(422, "pass exactly one of an audio file or `url`")
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
        for name, content in (inputs or {}).items():
            with open(os.path.join(job_dir, name), "wb") as f:
                f.write(content)
        for name, upload in (uploads or {}).items():
            save_upload(upload, os.path.join(job_dir, name))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    store.create(params, source, job_id, owner=who.owner)
    log.info("job %s (%s) created by %s", job_id, params.get("kind", "transcribe"), who.name)
    return job_id


def job_view(store: JobStore, job: dict) -> dict:
    view = {k: job[k] for k in ("id", "status", "stage", "progress", "params", "error", "owner",
                                "worker", "attempts", "created", "started", "finished", "stats")}
    view["queue_position"] = store.queue_position(job["id"])
    return view
