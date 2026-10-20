import logging
import os
import threading
import time
from collections import OrderedDict

from .. import audio
from . import config
from .jobs import CANCELLED, DONE, ERROR, QUEUED, RUNNING, JobStore

log = logging.getLogger("ai-trns.worker")

# Share of overall progress taken by ffmpeg conversion; the rest is Whisper
CONVERT_SHARE = 0.1


class ModelCache:
    """Keeps up to MAX_MODELS Transcriber instances, least recently used is evicted"""

    def __init__(self, max_models: int = config.MAX_MODELS):
        self.max_models = max(1, max_models)
        self.models = OrderedDict()

    def get(self, lang: str):
        if lang in self.models:
            self.models.move_to_end(lang)
            return self.models[lang]
        while len(self.models) >= self.max_models:
            old_lang, _ = self.models.popitem(last=False)
            log.info("unloading model %s", old_lang)
        log.info("loading model %s", lang)
        from ..transcriber import Transcriber
        self.models[lang] = Transcriber(lang, config.MODELS_DIR, config.DEVICE, config.COMPUTE_TYPE)
        return self.models[lang]

    def loaded(self) -> list:
        return list(self.models)


class Worker:
    """Single GPU worker: jobs run one after another"""

    def __init__(self, store: JobStore, models: ModelCache):
        self.store = store
        self.models = models
        self.wakeup = threading.Event()
        self.cancel_requested = set()
        self.current = None
        self.stopping = False
        self.thread = threading.Thread(target=self._loop, name="worker", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping = True
        if self.current:
            self.cancel_requested.add(self.current)
        self.wakeup.set()
        self.thread.join(timeout=30)

    def notify(self):
        self.wakeup.set()

    def cancel(self, job_id: str) -> bool:
        """Returns False if the job is already finished"""
        job = self.store.get(job_id)
        if job is None or job["status"] not in (QUEUED, RUNNING):
            return False
        if job["status"] == QUEUED:
            self.store.update(job_id, status=CANCELLED, finished=time.time())
        else:
            self.cancel_requested.add(job_id)
        return True

    def _loop(self):
        while not self.stopping:
            ids = self.store.queued_ids()
            if not ids:
                self.wakeup.wait(timeout=5)
                self.wakeup.clear()
                continue
            self._run(ids[0])

    def _run(self, job_id: str):
        job = self.store.get(job_id)
        if job is None or job["status"] != QUEUED:
            return
        self.current = job_id
        self.store.update(job_id, status=RUNNING, stage="converting", started=time.time())
        should_cancel = lambda: job_id in self.cancel_requested
        last_write = [0.0]

        def progress(stage: str, base: float, share: float):
            def on_progress(p: float):
                now = time.time()
                if now - last_write[0] >= 1 or p >= 1:
                    last_write[0] = now
                    self.store.update(job_id, stage=stage, progress=round(base + share * p, 4))
            return on_progress

        params = job["params"]
        job_dir = self.store.job_dir(job_id)
        wav_file = os.path.join(job_dir, "audio.wav")
        try:
            duration = audio.get_duration(job["source"])
            if duration == 0:
                raise RuntimeError("could not determine audio duration")
            if not audio.convert_to_wav(job["source"], wav_file, duration,
                                        progress("converting", 0, CONVERT_SHARE), should_cancel):
                raise _Cancelled()

            self.store.update(job_id, stage="loading_model", progress=CONVERT_SHARE)
            transcriber = self.models.get(params["lang"])

            self.store.update(job_id, stage="transcribing")
            from ..transcriber import Cancelled
            try:
                results = transcriber.transcribe(
                    wav_file, formats=params["formats"], profile=params["profile"],
                    impl=params["impl"], fix_rtl=params["fix_rtl"],
                    on_progress=progress("transcribing", CONVERT_SHARE, 1 - CONVERT_SHARE),
                    should_cancel=should_cancel)
            except Cancelled:
                raise _Cancelled()

            for fmt, content in results.items():
                with open(os.path.join(job_dir, f"result.{fmt}"), "w", encoding="utf-8") as f:
                    f.write(content)
            self.store.update(job_id, status=DONE, stage=None, progress=1.0, finished=time.time())
            log.info("job %s done", job_id)
        except _Cancelled:
            self.store.update(job_id, status=CANCELLED, finished=time.time())
            log.info("job %s cancelled", job_id)
        except Exception as e:
            log.exception("job %s failed", job_id)
            self.store.update(job_id, status=ERROR, error=str(e), finished=time.time())
        finally:
            self.cancel_requested.discard(job_id)
            self.current = None
            for name in ("audio.wav", "upload"):
                try:
                    os.remove(os.path.join(job_dir, name))
                except FileNotFoundError:
                    pass


class _Cancelled(Exception):
    pass
