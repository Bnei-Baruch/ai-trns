"""
GPU worker: lease a job from the API, get its audio, transcribe it to a words document,
send it back. One job at a time per worker (one worker per GPU). Run: python -m ai_trns.worker
"""
import gc
import logging
import os
import shutil
import signal
import threading
import time
from typing import Optional

from .. import audio, protocol
from . import config
from .client import ApiError, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns.worker")

# Share of the job's progress taken by audio conversion; the rest is Whisper
CONVERT_SHARE = 0.1


class Models:
    """Loaded Transcribers by language, unloaded after IDLE_UNLOAD_S so the GPU is free for others"""

    def __init__(self):
        self.loaded = {}
        self.last_used = 0.0

    def get(self, lang: str):
        if lang not in self.loaded:
            # one model at a time on an 8 GB GPU
            self.unload()
            log.info("loading model %s", lang)
            from ..transcriber import Transcriber
            self.loaded[lang] = Transcriber(lang, config.MODELS_DIR, config.DEVICE, config.COMPUTE_TYPE)
        self.last_used = time.time()
        return self.loaded[lang]

    def unload_if_idle(self):
        if self.loaded and time.time() - self.last_used > config.IDLE_UNLOAD_S:
            self.unload()

    def unload(self):
        if self.loaded:
            log.info("unloading models %s", ", ".join(self.loaded))
            self.loaded.clear()
            gc.collect()


def available_langs() -> list:
    if config.LANGS:
        return config.LANGS
    langs = [lang for lang in ("he", "en", "ru") if os.path.isdir(os.path.join(config.MODELS_DIR, lang))]
    return langs + (["auto"] if "en" in langs else [])


class Heartbeat:
    """Reports progress and keeps the lease alive from a thread, so a long step cannot lose the job"""

    def __init__(self, client: Client, job_id: str):
        self.client, self.job_id = client, job_id
        self.stage, self.progress = "starting", 0.0
        self.cancelled = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="heartbeat", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=5)

    def update(self, stage: str, progress: float):
        self.stage, self.progress = stage, progress

    def _loop(self):
        while not self.stopped.is_set():
            try:
                if self.client.heartbeat(self.job_id, self.stage, self.progress):
                    self.cancelled.set()
            except Exception as e:
                # the lease survives a few missed beats
                log.warning("job %s: heartbeat failed: %s", self.job_id, e)
            self.stopped.wait(protocol.HEARTBEAT_S)


def run_job(client: Client, models: Models, job: dict):
    job_id, params = job["id"], job["params"]
    job_dir = os.path.join(config.WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        with Heartbeat(client, job_id) as beat:
            if "upload" in job["source"]:
                beat.update("downloading", 0.0)
                source = os.path.join(job_dir, "upload")
                client.download_audio(job["source"]["upload"], source)
            else:
                source = job["source"]["url"]
            duration = audio.get_duration(source)
            if duration == 0:
                raise RuntimeError("could not determine audio duration")
            wav_file = os.path.join(job_dir, "audio.wav")
            if not audio.convert_to_wav(source, wav_file, duration,
                                        lambda p: beat.update("converting", CONVERT_SHARE * p),
                                        beat.cancelled.is_set):
                log.info("job %s cancelled", job_id)
                return
            beat.update("loading_model", CONVERT_SHARE)
            transcriber = models.get(params["lang"])
            beat.update("transcribing", CONVERT_SHARE)
            from ..transcriber import Cancelled
            try:
                doc = transcriber.transcribe(
                    wav_file, profile=params.get("profile"), batch_size=config.BATCH_SIZE,
                    on_progress=lambda p: beat.update("transcribing", CONVERT_SHARE + (1 - CONVERT_SHARE) * p),
                    should_cancel=beat.cancelled.is_set)
            except Cancelled:
                log.info("job %s cancelled", job_id)
                return
            models.last_used = time.time()
        client.result(job_id, doc)
        log.info("job %s done: %.0f s of audio, %d words", job_id, doc["duration_s"], len(doc["words"]))
    except ApiError as e:
        # 409: the job was cancelled or given to another worker meanwhile; nothing to report
        log.warning("job %s: %s", job_id, e)
    except Exception as e:
        log.exception("job %s failed", job_id)
        try:
            client.fail(job_id, f"{type(e).__name__}: {e}")
        except Exception as report_error:
            log.warning("job %s: could not report the failure: %s", job_id, report_error)
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def main(stop: Optional[threading.Event] = None):
    if not config.WORKER_KEY:
        raise SystemExit("AI_TRNS_WORKER_KEY is not set")
    langs = available_langs()
    if not langs:
        raise SystemExit(f"no models in {config.MODELS_DIR}")
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        # finish the running job, then exit; if killed first, its lease expires and it is retried
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    client = Client(config.API_URL, config.WORKER_KEY, config.WORKER_ID)
    models = Models()
    log.info("worker %s: API %s, langs %s", config.WORKER_ID, config.API_URL, ",".join(langs))
    while not stop.is_set():
        try:
            job = client.lease(langs)
        except Exception as e:
            log.warning("lease failed: %s", e)
            job = None
            if isinstance(e, ApiError) and e.status in (401, 409):
                # wrong key or protocol version: retrying fast will not help
                stop.wait(60)
        if job is None:
            models.unload_if_idle()
            stop.wait(config.POLL_S)
            continue
        log.info("job %s: lang %s, formats %s", job["id"], job["params"]["lang"], job["params"]["formats"])
        run_job(client, models, job)


if __name__ == "__main__":
    main()
