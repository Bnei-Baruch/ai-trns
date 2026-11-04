"""
GPU worker: lease a job from the API, get its audio, run it through an engine, send back the
document. Engines: transcribe (faster-whisper -> words document, or whisper_trainer's service
with AI_TRNS_STT_URL: ai_trns.stt_service), diarize (pyannote + MMS -> diarization document,
ai_trns.diarize) and live (a stream -> events while it plays, then a words document: ai_trns.live).
One job at a time per worker (one worker per GPU).
Run: python -m ai_trns.worker
"""
import gc
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from typing import Optional

from .. import audio, protocol
from . import config
from .client import ApiError, Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns.worker")

# Share of the job's progress taken by audio conversion; the rest is the engine
CONVERT_SHARE = 0.1
DIARIZE, LIVE = "diarize", "live"


class Models:
    """Loaded Transcribers by language and the Diarizer, unloaded after IDLE_UNLOAD_S so the GPU is
    free for others"""

    def __init__(self):
        self.loaded = {}
        self.last_used = 0.0

    def _load(self, key: str, make):
        if key not in self.loaded:
            # one model at a time on an 8 GB GPU
            self.unload()
            log.info("loading model %s", key)
            self.loaded[key] = make()
        self.last_used = time.time()
        return self.loaded[key]

    def get(self, lang: str):
        from ..transcriber import Transcriber
        return self._load(lang, lambda: Transcriber(lang, config.MODELS_DIR, config.DEVICE, config.COMPUTE_TYPE))

    def diarizer(self):
        from ..diarize.engine import Diarizer
        return self._load(DIARIZE, lambda: Diarizer(config.SPEAKERS_DIR, config.DEVICE, config.DIARIZE_PIPELINE,
                                                    config.HF_TOKEN, config.DIARIZE_EXTRA_THRESHOLD))

    def unload_if_idle(self):
        if self.loaded and time.time() - self.last_used > config.IDLE_UNLOAD_S:
            self.unload()

    def unload(self):
        if self.loaded:
            log.info("unloading models %s", ", ".join(self.loaded))
            self.loaded.clear()
            gc.collect()
            if "torch" in sys.modules:
                # give the memory back to the other services on the GPU, not only to torch's cache
                sys.modules["torch"].cuda.empty_cache()


def available_langs() -> list:
    """Languages of the models here; with AI_TRNS_STT_URL also he, which the service transcribes"""
    if config.LANGS:
        return config.LANGS
    langs = [lang for lang in ("he", "en", "ru") if os.path.isdir(os.path.join(config.MODELS_DIR, lang))]
    if config.STT_URL and "he" not in langs:
        langs.insert(0, "he")
    return langs + (["auto"] if "en" in langs else [])


def by_service(job: dict) -> bool:
    """With AI_TRNS_STT_URL, Hebrew URL jobs go to the service; uploads, other languages, diarize and live run
    here (the service takes Hebrew media URLs only)"""
    params = job["params"]
    return bool(config.STT_URL) and params.get("kind") not in (DIARIZE, LIVE) and params["lang"] == "he" \
        and "url" in job["source"]


def free_gpu_mb() -> Optional[int]:
    """Free memory of this worker's GPU by nvidia-smi; None if it cannot tell"""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "-i", config.GPU_INDEX, "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def gpu_has_room(models: "Models") -> bool:
    """Whether to ask for a job now: always with AI_TRNS_MIN_FREE_GPU_MB=0 or a model loaded here already"""
    if config.MIN_FREE_GPU_MB <= 0 or models.loaded:
        return True
    free = free_gpu_mb()
    return free is None or free >= config.MIN_FREE_GPU_MB


class Heartbeat:
    """Reports progress and keeps the lease alive from a thread, so a long step cannot lose the job"""

    def __init__(self, client: Client, job_id: str):
        self.client, self.job_id = client, job_id
        self.stage, self.progress = "starting", 0.0
        self.cancelled = threading.Event()
        # a live job is to finish now, with what it has
        self.finish = threading.Event()
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
                reply = self.client.heartbeat(self.job_id, self.stage, self.progress)
                if reply.get("cancel"):
                    self.cancelled.set()
                if reply.get("stop"):
                    self.finish.set()
            except Exception as e:
                # the lease survives a few missed beats
                log.warning("job %s: heartbeat failed: %s", self.job_id, e)
            self.stopped.wait(protocol.HEARTBEAT_S)


def diarize(client: Client, models: Models, job: dict, job_dir: str, wav_file: str, beat: "Heartbeat") -> dict:
    """The diarize engine: the job's cues and voice samples come from the API next to its audio"""
    job_id, params = job["id"], job["params"]
    beat.update("downloading", CONVERT_SHARE)
    cues_file = os.path.join(job_dir, "cues.json")
    client.download(client.file_path(job_id, "cues.json"), cues_file)
    with open(cues_file, encoding="utf-8") as f:
        cues = json.load(f)
    samples = []
    for sample in params.get("samples", []):
        path = os.path.join(job_dir, sample["file"])
        client.download(client.file_path(job_id, sample["file"]), path)
        samples.append((sample["name"], path))
    beat.update("loading_model", CONVERT_SHARE)
    diarizer = models.diarizer()
    beat.update("diarizing", CONVERT_SHARE)
    # with paragraphs, the last share of the progress is the paragraphing service
    share = (1 - CONVERT_SHARE) * (0.85 if params.get("paragraphs") else 1.0)
    doc = diarizer.diarize(
        wav_file, cues, samples, lang=params["lang"], speakers=params.get("speakers"),
        on_progress=lambda p: beat.update("diarizing", CONVERT_SHARE + share * p),
        should_cancel=beat.cancelled.is_set)
    if params.get("paragraphs"):
        beat.update("paragraphs", CONVERT_SHARE + share)
        add_paragraphs(doc)
    return doc


def add_paragraphs(doc: dict) -> None:
    """The Rav's paragraph starts from the paragraphing service into the diarization document
    (ai_trns.paragraphs); if the service cannot be used, the reason goes to its stats and the job still succeeds"""
    from ..paragraphs import build, client
    stats = doc.setdefault("stats", {})
    if not config.PARAGRAPHS_URL:
        stats["paragraphs_error"] = "no paragraphing service (AI_TRNS_PARAGRAPHS_URL)"
        return
    turns = build.speaker_turns(doc["words"])
    text, index, offsets = build.model_text(doc["words"], turns)
    if not text:
        doc["paragraph_starts"] = []
        return
    try:
        reply = client.split(config.PARAGRAPHS_URL, text, config.PARAGRAPHS_KEY)
    except RuntimeError as e:
        log.warning("paragraphs: %s", e)
        stats["paragraphs_error"] = str(e)
        return
    doc["paragraph_starts"] = build.starts_from_sentences(index, offsets, reply["sentences"], reply["breaks"])
    doc["paragraphs_model"] = "mdb-lora paragraphing"
    log.info("paragraphs: %d breaks in %s s", len(doc["paragraph_starts"]), reply.get("seconds"))


def live(client: Client, models: Models, job: dict, beat: "Heartbeat",
         shutdown: Optional[threading.Event] = None) -> Optional[dict]:
    """The live engine: the job's stream (or a recording at its natural pace), events to the API after every decode;
    None if cancelled. Asked to stop, or the worker shutting down: the result is what was heard so far"""
    from ..live.engine import run
    from ..live.source import AudioSource, is_live
    from ..live.stream import StreamProcessor, WhisperAsr, build_live_document
    job_id, lang = job["id"], job["params"]["lang"]
    if "url" not in job["source"]:
        raise RuntimeError("a live job needs a URL")
    beat.update("loading_model", 0.0)
    transcriber = models.get(lang)
    processor = StreamProcessor(WhisperAsr(transcriber))
    pending = []

    def on_event(e: dict):
        # partials only matter until the next one: keep the latest
        if e["type"] == "partial":
            pending[:] = [p for p in pending if p["type"] != "partial"]
        pending.append(e)
        try:
            if client.events(job_id, pending):
                beat.finish.set()
            pending.clear()
        except ApiError as err:
            if err.status == 409:
                beat.cancelled.set()
            log.warning("job %s: events not sent: %s", job_id, err)
        except Exception as err:
            # kept and sent with the next ones
            log.warning("job %s: events not sent: %s", job_id, err)
        beat.update("live", 0.0)
        models.last_used = time.time()

    beat.update("live", 0.0)
    url = job["source"]["url"]
    # a recording is read at its natural pace, as if it were broadcast
    with AudioSource(url, realtime=not is_live(url)) as source:
        duration = run(source, processor, on_event,
                       should_stop=lambda: beat.cancelled.is_set() or beat.finish.is_set()
                       or bool(shutdown and shutdown.is_set()))
    if beat.cancelled.is_set():
        log.info("job %s cancelled", job_id)
        return None
    models.last_used = time.time()
    return build_live_document(processor.final, duration, lang, transcriber.name,
                               transcriber.meta.get("boundary_correction_s"))


class ServiceUnavailable(Exception):
    """The STT service could not take the job; it may run on the local model instead"""


def transcribe_by_service(job: dict, beat: "Heartbeat") -> Optional[dict]:
    """The transcribe engine with AI_TRNS_STT_URL: the service fetches the media itself"""
    from .. import stt_service
    if "url" not in job["source"]:
        raise RuntimeError("the STT service takes media URLs only, not uploaded files")
    try:
        return stt_service.transcribe(config.STT_URL, job["source"]["url"], job["params"]["lang"],
                                      config.STT_TIMEOUT_S, beat.update, beat.cancelled.is_set)
    except stt_service.Cancelled:
        log.info("job %s cancelled", job["id"])
        return None
    except stt_service.Unavailable as e:
        raise ServiceUnavailable(str(e)) from None


def can_fall_back(job: dict) -> bool:
    """A local model for the job's language, and fallback allowed (AI_TRNS_STT_FALLBACK)"""
    return config.STT_FALLBACK and os.path.isdir(os.path.join(config.MODELS_DIR, job["params"]["lang"]))


def run_locally(client: Client, models: Models, job: dict, job_dir: str, beat: "Heartbeat") -> Optional[dict]:
    """The job's audio here (downloaded or read from its URL) through an engine on this GPU;
    None if cancelled"""
    job_id, params = job["id"], job["params"]
    if "upload" in job["source"]:
        beat.update("downloading", 0.0)
        source = os.path.join(job_dir, "upload")
        client.download(job["source"]["upload"], source)
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
        return None
    from ..transcriber import Cancelled
    try:
        if params.get("kind") == DIARIZE:
            doc = diarize(client, models, job, job_dir, wav_file, beat)
        else:
            beat.update("loading_model", CONVERT_SHARE)
            transcriber = models.get(params["lang"])
            beat.update("transcribing", CONVERT_SHARE)
            doc = transcriber.transcribe(
                wav_file, profile=params.get("profile"), batch_size=config.BATCH_SIZE,
                on_progress=lambda p: beat.update("transcribing", CONVERT_SHARE + (1 - CONVERT_SHARE) * p),
                should_cancel=beat.cancelled.is_set)
    except Cancelled:
        log.info("job %s cancelled", job_id)
        return None
    models.last_used = time.time()
    return doc


RESULT_TRIES, RESULT_RETRY_S = 6, 10.0


def send_result(client: Client, job_id: str, doc: dict):
    """The finished document to the API; an API restarting meanwhile does not lose the work"""
    for attempt in range(RESULT_TRIES):
        try:
            return client.result(job_id, doc)
        except ApiError:
            raise
        except OSError as e:
            if attempt == RESULT_TRIES - 1:
                raise
            log.warning("job %s: result not sent (%s), again in %.0f s", job_id, e, RESULT_RETRY_S)
            time.sleep(RESULT_RETRY_S)


def run_job(client: Client, models: Models, job: dict, shutdown: Optional[threading.Event] = None):
    job_id, params = job["id"], job["params"]
    job_dir = os.path.join(config.WORK_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    try:
        with Heartbeat(client, job_id) as beat:
            if params.get("kind") == LIVE:
                doc = live(client, models, job, beat, shutdown)
            elif by_service(job):
                try:
                    doc = transcribe_by_service(job, beat)
                except ServiceUnavailable as e:
                    if not can_fall_back(job):
                        raise
                    log.warning("job %s: %s, transcribing on the local model", job_id, e)
                    doc = run_locally(client, models, job, job_dir, beat)
            else:
                doc = run_locally(client, models, job, job_dir, beat)
        if doc is None:
            return
        send_result(client, job_id, doc)
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
    engines = config.ENGINES
    langs = available_langs() if "transcribe" in engines else []
    if "transcribe" in engines and not langs:
        raise SystemExit(f"no models in {config.MODELS_DIR}")
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        # finish the running job (a live one: with what it heard so far), then exit; if killed first,
        # its lease expires and it is retried
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    client = Client(config.API_URL, config.WORKER_KEY, config.WORKER_ID)
    models = Models()
    log.info("worker %s: API %s, engines %s, langs %s%s", config.WORKER_ID, config.API_URL, ",".join(engines),
             ",".join(langs), f", transcribing by {config.STT_URL}" if config.STT_URL else "")
    waiting_for_gpu = False
    while not stop.is_set():
        if not gpu_has_room(models):
            if not waiting_for_gpu:
                log.info("less than %d MiB GPU memory free: not taking jobs for now", config.MIN_FREE_GPU_MB)
            waiting_for_gpu = True
            stop.wait(config.POLL_S)
            continue
        waiting_for_gpu = False
        try:
            job = client.lease(langs, engines)
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
        params = job["params"]
        log.info("job %s: %s, lang %s, formats %s", job["id"], params.get("kind", "transcribe"), params["lang"],
                 params["formats"])
        run_job(client, models, job, stop)


if __name__ == "__main__":
    main()
