"""
Live transcription (TASK-live-transcription.md): a job of kind "live" for the worker engine "live"
(ai_trns.live), whose text reaches clients while it is being recognized.

    POST /live                 url, lang -> 202 job (as POST /jobs)
    GET  /live/{id}/events     server-sent events (ai_trns.live.hub): start, final, partial, end, error;
                               a client joining late first gets the final words so far
    POST /live/{id}/stop       leave; the last listener finishes the job (the worker sends what it has,
                               the job ends done as usual)
    PUT  /live/{id}/doc        url, title, page: a listener's page writes the text into this Google Doc of
                               its user's; the page repeats it every BROWSER_DOC_S while it writes.
                               One page per doc: 409 while another page writes it (or it is the
                               server's doc), since each page appends the same text
    DELETE /live/{id}/doc      url, page: it stopped
    The server's own doc (ai_trns.api.gdoc, if configured) is created with the job. Listeners learn
    about the docs from "docs" events: {"server": {url, title} | {error} | null, "browser": [{url, title, by}]}

A broadcast (an HLS / DASH stream, ai_trns.audio.is_live) is recognized once for everyone: POST /live
with a stream that is already being recognized returns that job, and any logged-in user may listen
to it and get its result. Nobody listening for IDLE_STOP_S finishes it too (tabs closed).
A recording is a job of its own, as before.

The worker posts its events to POST /worker/jobs/{id}/events; the result (srt, txt, json) is the
usual GET /jobs/{id}/result. Events are kept in memory, one Hub per live job: the API runs as one
process. DELETE /jobs/{id} cancels without a result (owner only).
"""
import json
import logging
import os
import threading
import time
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..audio import is_live as is_stream
from ..live.hub import Hub
from . import config, gdoc
from .auth import Principal, can_see, principal
from .jobs import CANCELLED, DONE, FINAL, QUEUED, RUNNING, kind_of
from .submit import create_job, job_view

log = logging.getLogger("ai-trns.live")

KIND = "live"
FORMATS = ["srt", "txt", "json"]
# Events of a finished live job stay in memory this long; later, a client gets them from the result
KEEP_S = 600
# A running live job nobody has listened to for this long is finished (a reload is not leaving)
IDLE_STOP_S = 60
# A listener's doc not confirmed for this long is dropped from the list (its page is gone)
BROWSER_DOC_S = 60

router = APIRouter(prefix="/live", tags=["live"])


class Live:
    """Hubs of live jobs and the jobs asked to stop"""

    def __init__(self):
        self.hubs: dict[str, Hub] = {}
        self.stopping: set[str] = set()
        # the server's Google Doc of a job, and its writer
        self.server_docs: dict[str, dict] = {}
        self.writers: dict[str, gdoc.Writer] = {}
        # the docs listeners' pages write: job id -> doc url -> {url, title, by, owner, page, seen}
        self.browser_docs: dict[str, dict[str, dict]] = {}

    def hub(self, job_id: str) -> Hub:
        return self.hubs.setdefault(job_id, Hub())

    def publish(self, job_id: str, event: dict):
        """Only to a hub someone may be listening to"""
        if job_id in self.hubs:
            self.hubs[job_id].publish({"words": [], **event})

    def leased(self, job_id: str):
        """A worker starts the stream (again, after a lost worker): listeners drop what they have"""
        hub = self.hubs.get(job_id)
        if hub is not None and hub.events:
            hub.publish({"type": "reset", "words": []})

    def cleanup(self, store):
        now = time.time()
        for job_id in list(self.hubs):
            job = store.get(job_id)
            if job and job["status"] == RUNNING and self.hubs[job_id].idle_s() > IDLE_STOP_S:
                self.stopping.add(job_id)
            if job is None or (job["status"] in FINAL and now - (job["finished"] or now) > KEEP_S):
                self.hubs.pop(job_id, None)
                self.stopping.discard(job_id)
                writer = self.writers.pop(job_id, None)
                if writer:
                    writer.stop()
                self.server_docs.pop(job_id, None)
                self.browser_docs.pop(job_id, None)
                continue
            docs = self.browser_docs.get(job_id, {})
            gone = [url for url, d in docs.items() if now - d["seen"] > BROWSER_DOC_S]
            for url in gone:
                del docs[url]
            if gone:
                self.publish_docs(job_id)

    def publish_docs(self, job_id: str):
        browser = [{k: d[k] for k in ("url", "title", "by")} for d in self.browser_docs.get(job_id, {}).values()]
        self.publish(job_id, {"type": "docs", "server": self.server_docs.get(job_id), "browser": browser})

    def server_doc(self, job_id: str, title: str):
        """Creates the job's doc on the server's account and writes the text into it, in the background"""
        docs = gdoc.service()
        if docs is None:
            return

        def create():
            try:
                doc = docs.create(title, config.GOOGLE_FOLDER, config.GOOGLE_DOC_ROLE)
                _, empty = docs.open(doc)
            except gdoc.DocError as e:
                log.warning("live %s: no doc: %s", job_id, e)
                self.server_docs[job_id] = {"error": str(e)}
                self.publish_docs(job_id)
                return

            def failed(_status, message):
                self.server_docs[job_id] = {**self.server_docs.get(job_id, {}), "error": message}
                self.publish_docs(job_id)

            self.server_docs[job_id] = {"url": gdoc.doc_url(doc), "title": title}
            self.writers[job_id] = gdoc.Writer(docs, doc, self.hub(job_id), failed, empty)
            log.info("live %s: writing doc %s", job_id, doc)
            self.publish_docs(job_id)

        threading.Thread(target=create, name=f"gdoc-new-{job_id[:8]}", daemon=True).start()


def is_live(job: dict) -> bool:
    return kind_of(job["params"]) == KIND


def is_shared(job: dict) -> bool:
    """A broadcast's job: anyone may listen to it and get its result"""
    return is_live(job) and is_stream(job["source"])


def visible(who: Principal, job: dict) -> bool:
    return can_see(who, job) or is_shared(job)


def _job(request: Request, job_id: str, who: Principal) -> dict:
    job = request.app.state.store.get(job_id)
    if job is None or not visible(who, job) or not is_live(job):
        raise HTTPException(404, "live job not found")
    return job


def _running(store, url: str, lang: str) -> Optional[dict]:
    """The job already recognizing this broadcast"""
    for job_id in store.queued_ids() + store.running_ids():
        job = store.get(job_id)
        if job and is_live(job) and job["source"] == url and job["params"]["lang"] == lang:
            return job
    return None


def _from_result(store, job: dict) -> Hub:
    """A hub for a job whose events are gone (finished long ago, API restarted)"""
    hub = Hub()
    if job["status"] == DONE:
        with open(os.path.join(store.job_dir(job["id"]), "result.json"), encoding="utf-8") as f:
            doc = json.load(f)
        words = [{k: w[k] for k in ("word", "start_s", "end_s", "probability")} for w in doc["words"]]
        hub.publish({"type": "final", "words": words})
        hub.publish({"type": "end", "words": [], "duration_s": doc["duration_s"]})
    else:
        hub.publish({"type": "error", "words": [], "message": job["error"] or f"job is {job['status']}"})
    return hub


@router.post("", status_code=202)
def start(request: Request, who: Principal = Depends(principal), url: str = Form(...),
          lang: Literal["he", "en", "ru"] = Form("he"), fix_rtl: bool = Form(False)):
    """A live job for a stream or a media URL (read at its natural pace, as if it were live)"""
    store = request.app.state.store
    if is_stream(url):
        job = _running(store, url, lang)
        if job is not None and job["id"] not in request.app.state.live.stopping:
            return job_view(store, job)
    params = {"kind": KIND, "lang": lang, "formats": FORMATS, "profile": None, "fix_rtl": fix_rtl}
    job_id = create_job(store, who, params, None, url)
    live = request.app.state.live
    live.hub(job_id)
    live.server_doc(job_id, f"תמלול חי {time.strftime('%Y-%m-%d %H:%M')}")
    return job_view(store, store.get(job_id))


@router.get("/{job_id}/events")
def events(job_id: str, request: Request, who: Principal = Depends(principal)):
    job = _job(request, job_id, who)
    live, store = request.app.state.live, request.app.state.store
    hub = live.hubs.get(job_id)
    if hub is None:
        hub = live.hub(job_id) if job["status"] in (QUEUED, RUNNING) else _from_result(store, job)
    # X-Accel-Buffering: Angie / nginx pass each event on at once
    return StreamingResponse(hub.sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/{job_id}/stop")
def stop(job_id: str, request: Request, who: Principal = Depends(principal)):
    """Leave the stream. The caller is the last one listening (its own events stream still counts):
    the job finishes, the worker sends what it heard. Others listening: it goes on for them
    ("stopping": false), the caller just closes its events"""
    job = _job(request, job_id, who)
    store, live = request.app.state.store, request.app.state.live
    hub = live.hubs.get(job_id)
    others = max(0, (hub.listeners if hub else 0) - 1)
    if job["status"] not in (QUEUED, RUNNING):
        raise HTTPException(409, "job is already finished")
    if others:
        return {**job_view(store, job), "stopping": False, "listeners": others}
    if job["status"] == QUEUED:
        store.update(job_id, status=CANCELLED, finished=time.time())
        live.publish(job_id, {"type": "error", "message": "stopped before it started"})
    else:
        live.stopping.add(job_id)
    return {**job_view(store, store.get(job_id)), "stopping": True, "listeners": 0}


@router.put("/{job_id}/doc")
def browser_doc(job_id: str, request: Request, who: Principal = Depends(principal), url: str = Form(...),
                title: str = Form(""), page: str = Form("")):
    """A listener's page writes the text into this doc of its user's (and says so again every
    BROWSER_DOC_S); the other listeners see it in the list. page: the writing page's id (a page that
    sends none is told apart by its user only)"""
    job = _job(request, job_id, who)
    if job["status"] not in (QUEUED, RUNNING):
        raise HTTPException(409, "job is already finished")
    try:
        doc = gdoc.doc_id(url)
    except gdoc.DocError as e:
        raise HTTPException(422, str(e))
    live = request.app.state.live
    docs = live.browser_docs.setdefault(job_id, {})
    key = gdoc.doc_url(doc)
    server = live.server_docs.get(job_id) or {}
    if server.get("url") == key:
        raise HTTPException(409, "the server already writes this doc")
    held = docs.get(key)
    if held and (held["owner"], held["page"]) != (who.owner, page) and time.time() - held["seen"] <= BROWSER_DOC_S:
        raise HTTPException(409, f"{held['by']} already writes this doc")
    new = held is None or held["title"] != title[:200]
    docs[key] = {"url": key, "title": title[:200], "by": who.name, "owner": who.owner, "page": page,
                 "seen": time.time()}
    if new:
        live.publish_docs(job_id)
    return {"ok": True}


@router.delete("/{job_id}/doc")
def drop_browser_doc(job_id: str, request: Request, url: str, page: str = "", who: Principal = Depends(principal)):
    _job(request, job_id, who)
    live = request.app.state.live
    try:
        key = gdoc.doc_url(gdoc.doc_id(url))
    except gdoc.DocError as e:
        raise HTTPException(422, str(e))
    entry = live.browser_docs.get(job_id, {}).get(key)
    if entry and ((entry["owner"], entry["page"]) == (who.owner, page) or who.sees_all):
        del live.browser_docs[job_id][key]
        live.publish_docs(job_id)
    return {"ok": True}
