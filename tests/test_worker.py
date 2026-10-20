"""The real worker loop against a real API server over HTTP, with a fake model"""
import socket
import threading
import time

import pytest

from tests.conftest import H, WORKER_KEY, words_doc


class FakeTranscriber:
    def __init__(self, delay=0.0):
        self.delay = delay

    def transcribe(self, audio, profile=None, batch_size=8, on_progress=None, should_cancel=None):
        for step in range(5):
            time.sleep(self.delay)
            if should_cancel and should_cancel():
                from ai_trns.transcriber import Cancelled
                raise Cancelled()
            on_progress((step + 1) / 5)
        return words_doc()


@pytest.fixture
def server(api_config):
    import uvicorn
    from ai_trns.api.app import app
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    while not srv.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def worker(server, tmp_path, monkeypatch):
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    monkeypatch.setattr(config, "API_URL", server)
    monkeypatch.setattr(config, "WORKER_KEY", WORKER_KEY)
    monkeypatch.setattr(config, "WORKER_ID", "test-worker")
    monkeypatch.setattr(config, "LANGS", ["he"])
    monkeypatch.setattr(config, "POLL_S", 0.1)
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
    fake = FakeTranscriber()
    monkeypatch.setattr(w.Models, "get", lambda self, lang: fake)
    stop = threading.Event()
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    yield fake
    stop.set()
    thread.join(timeout=10)


def wait_status(http, job_id, statuses, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = http.get(f"/jobs/{job_id}", headers=H).json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(job)


def test_worker_transcribes_an_uploaded_file(server, worker, audio_file):
    import httpx
    with httpx.Client(base_url=server) as http:
        with open(audio_file, "rb") as f:
            job_id = http.post("/jobs", headers=H, files={"file": ("in.mp3", f)}, data={"format": "srt"}).json()["id"]
        job = wait_status(http, job_id, ("done", "error"))
        assert (job["status"], job["worker"]) == ("done", "test-worker"), job
        assert http.get(f"/jobs/{job_id}/result", headers=H).text.endswith("שלום עולם.\n")


def test_worker_reports_a_bad_input(server, worker, tmp_path):
    import httpx
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    with httpx.Client(base_url=server) as http:
        job_id = http.post("/jobs", headers=H, files={"file": ("bad.mp3", bad.read_bytes())}).json()["id"]
        job = wait_status(http, job_id, ("done", "error"))
        assert job["status"] == "error" and "duration" in job["error"]


def test_cancel_stops_the_worker_between_segments(server, worker, audio_file, monkeypatch):
    import httpx
    from ai_trns import protocol
    monkeypatch.setattr(protocol, "HEARTBEAT_S", 0.2)
    worker.delay = 0.5
    with httpx.Client(base_url=server) as http:
        with open(audio_file, "rb") as f:
            job_id = http.post("/jobs", headers=H, files={"file": ("in.mp3", f)}).json()["id"]
        wait_status(http, job_id, ("running",))
        time.sleep(0.6)
        http.delete(f"/jobs/{job_id}", headers=H)
        time.sleep(1.5)
        job = http.get(f"/jobs/{job_id}", headers=H).json()
        assert job["status"] == "cancelled"
        assert http.get("/health").json()["workers"]["test-worker"]["job"] is None


def test_worker_diarizes_with_the_job_cues_and_samples(server, worker, audio_file, monkeypatch):
    import httpx
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    from tests.test_tools import SRT, diarize_doc
    monkeypatch.setattr(config, "ENGINES", ["transcribe", "diarize"])
    calls = []

    class FakeDiarizer:
        def diarize(self, wav_file, cues, samples, lang, speakers, on_progress, should_cancel):
            calls.append((len(cues), [(name, open(path, "rb").read()) for name, path in samples], lang, speakers))
            on_progress(0.5)
            return diarize_doc()

    monkeypatch.setattr(w.Models, "diarizer", lambda self: FakeDiarizer())
    # the loop already running leases with the engines it read at start: run a second one
    stop = threading.Event()
    monkeypatch.setattr(config, "WORKER_ID", "diarize-worker")
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=server) as http, open(audio_file, "rb") as f:
            job_id = http.post("/diarize/srt", headers=H, data={"sample_names": ["Guest"], "num_speakers": "2"},
                               files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode())),
                                      ("samples", ("g.wav", b"voice"))]).json()["id"]
            job = wait_status(http, job_id, ("done", "error"))
            assert (job["status"], job["worker"]) == ("done", "diarize-worker"), job
            assert calls == [(2, [("Guest", b"voice")], "he", {"num_speakers": 2})]
            assert http.get(f"/jobs/{job_id}/result", headers=H).text.startswith("1\n00:00:00,000 --> 00:00:01,000\nרב: ")
    finally:
        stop.set()
        thread.join(timeout=10)
