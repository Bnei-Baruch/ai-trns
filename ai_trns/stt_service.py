"""
Transcription by whisper_trainer's job service (transcription_service.py, e.g. http://10.66.1.53:8765)
instead of a model on this host. Stdlib only.

    POST /transcribe {"urls": [url]} -> 202 + Location: /jobs/<id>
    GET  /jobs/<id>  -> {"status", "results": [{"url", "status", "duration_s", "srt_url", "srt"?}]}
    GET  /results/<request_id>.srt

The service takes media URLs only and gives an SRT, no word times. Its SRT is kept as is
(doc["srt"]) and served for format=srt; the words of the document are spread over each cue's
window (interpolated), which is enough for txt and the text diffs.
"""
import json
import time
import urllib.error
import urllib.request
from typing import Callable, Optional
from urllib.parse import urljoin

from .diarize.srt import parse_srt
from .words import VERSION, _interpolated_timings

POLL_S = 5
SOURCE = "service_cues"


class Cancelled(Exception):
    """Raised when the job is cancelled while the service transcribes it"""


def _get(url: str, timeout: float = 60):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}),
                                timeout=timeout) as response:
        return response.read()


def model_name(service_url: str) -> str:
    """whisper_trainer@<bundle sha>, so a review knows which bundle transcribed it"""
    try:
        sha = json.loads(_get(service_url.rstrip("/") + "/health", timeout=10)).get("bundle_sha256") or ""
    except (OSError, ValueError):
        sha = ""
    return f"whisper_trainer@{sha[:12]}" if sha else "whisper_trainer"


def document_from_srt(srt: str, duration_s: Optional[float], language: str, model: str) -> dict:
    """The service's SRT -> words document (ai_trns.words) carrying that SRT"""
    cues = [c for c in parse_srt(srt) if c["text"].split()]
    duration_s = float(duration_s or (cues[-1]["end"] if cues else 0.0))
    segments, words = [], []
    for c in cues:
        text = " ".join(c["text"].split())
        index = len(segments)
        segments.append({"start_s": c["start"], "end_s": c["end"], "text": text})
        for w in _interpolated_timings(c["start"], c["end"], text, duration_s):
            w["segment"] = index
            words.append(w)
    return {
        "version": VERSION,
        "duration_s": duration_s,
        "language": language,
        "model": model,
        "timing": {"source": SOURCE, "boundary_correction_s": {"start": 0.0, "end": 0.0}},
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "words": words,
        "srt": srt,
    }


def transcribe(service_url: str, media_url: str, language: str = "he", timeout_s: float = 3 * 3600,
               on_progress: Optional[Callable[[str, float], None]] = None,
               should_cancel: Optional[Callable[[], bool]] = None) -> dict:
    """One media URL through the service -> words document. on_progress(stage, 0..1): the service
    reports no progress inside a file, only queued / running / completed"""
    base = service_url.rstrip("/") + "/"
    body = json.dumps({"urls": [media_url]}).encode("utf-8")
    request = urllib.request.Request(base + "transcribe", data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            job_url = urljoin(base, "." + response.headers["Location"])
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"STT service: {e.code} {e.read().decode('utf-8', 'replace')[:300]}") from None
    model = model_name(service_url)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if should_cancel and should_cancel():
            raise Cancelled()
        job = json.loads(_get(job_url))
        results = job.get("results") or []
        item = next((r for r in results if r.get("url") == media_url), results[0] if results else None)
        status = item.get("status") if item else job.get("status")
        if status in ("failed", "skipped"):
            raise RuntimeError(f"STT service: {item.get('reason') or item.get('warnings') or status}")
        if job.get("status") == "failed":
            raise RuntimeError("STT service: job failed")
        if status == "completed":
            srt = item.get("srt")
            if srt is None:
                srt = _get(urljoin(base, "." + item["srt_url"])).decode("utf-8-sig")
            return document_from_srt(srt, item.get("duration_s"), language, model)
        if on_progress:
            on_progress("transcribing" if status == "running" else "queued_upstream",
                        0.5 if status == "running" else 0.0)
        time.sleep(POLL_S)
    raise RuntimeError(f"STT service: no result after {timeout_s:.0f} s")
