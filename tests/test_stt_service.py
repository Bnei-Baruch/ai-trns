"""Transcription by whisper_trainer's job service (ai_trns.stt_service), against a fake of the service"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ai_trns import stt_service
from ai_trns.api.review import chunks_from_words
from ai_trns.subtitles import render
from tests.conftest import H
from tests.test_worker import server, wait_status  # noqa: F401 (fixture)

MEDIA = "https://cdn.example.org/lesson.mp3"
SRT = ("1\n00:00:00,370 --> 00:00:02,100\nשלום,\nעולם.\n\n"
       "2\n00:00:02,500 --> 00:00:03,900\nלחו\"ל היום\n\n")


class FakeService:
    """POST /transcribe -> a job that is running for `polls` reads, then completed"""

    def __init__(self, polls=1, status="completed"):
        self.polls, self.status, self.urls = polls, status, []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body, ctype="application/json", **headers):
                data = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.urls += body["urls"]
                self._send(202, {"job_id": "j1", "status": "running"}, Location="/jobs/j1")

            def do_GET(self):
                if self.path == "/health":
                    return self._send(200, {"ok": True, "bundle_sha256": "a037e65c8bd8618bfd70"})
                if self.path == "/results/r1.srt":
                    return self._send(200, SRT, "application/x-subrip")
                if self.path != "/jobs/j1":
                    return self._send(404, {"error": "not found"})
                done = service.polls <= 0
                service.polls -= 1
                item = {"url": service.urls[-1], "request_id": "r1",
                        "status": service.status if done else "running"}
                if done and service.status == "completed":
                    item.update(duration_s=4.0, srt_url="/results/r1.srt")
                elif done:
                    item["reason"] = "not_hebrew"
                self._send(200, {"job_id": "j1", "status": "done" if done else "running", "results": [item]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(stt_service, "POLL_S", 0.05)
    s = FakeService()
    yield s
    s.close()


def test_the_service_srt_becomes_a_words_document(service):
    doc = stt_service.transcribe(service.url, MEDIA)
    assert service.urls == [MEDIA]
    assert doc["model"] == "whisper_trainer@a037e65c8bd8"
    assert doc["duration_s"] == 4.0 and doc["timing"]["source"] == "service_cues"
    assert doc["text"] == 'שלום, עולם. לחו"ל היום'
    assert [w["word"] for w in doc["words"]] == doc["text"].split()
    assert all(w["interpolated"] for w in doc["words"])
    first = [w for w in doc["words"] if w["segment"] == 0]
    assert first[0]["start_s"] == pytest.approx(0.37) and first[-1]["end_s"] == pytest.approx(2.1)
    # SRT as the service made it, not cues rebuilt from interpolated words
    assert render(doc, "srt") == SRT
    assert render(doc, "txt") == doc["text"]
    assert chunks_from_words(doc) == [{"start": 0.37, "end": 2.1, "text": "שלום, עולם."},
                                      {"start": 2.5, "end": 3.9, "text": 'לחו"ל היום'}]


def test_a_skipped_file_is_an_error(monkeypatch):
    monkeypatch.setattr(stt_service, "POLL_S", 0.05)
    s = FakeService(status="skipped")
    try:
        with pytest.raises(RuntimeError, match="not_hebrew"):
            stt_service.transcribe(s.url, MEDIA)
    finally:
        s.close()


def test_cancel_stops_waiting_for_the_service(service):
    service.polls = 1000
    with pytest.raises(stt_service.Cancelled):
        stt_service.transcribe(service.url, MEDIA, should_cancel=lambda: service.polls < 998)


def test_worker_transcribes_a_url_job_by_the_service(server, service, tmp_path, monkeypatch):  # noqa: F811
    import httpx
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    from tests.conftest import WORKER_KEY
    monkeypatch.setattr(config, "API_URL", server)
    monkeypatch.setattr(config, "WORKER_KEY", WORKER_KEY)
    monkeypatch.setattr(config, "WORKER_ID", "service-worker")
    monkeypatch.setattr(config, "STT_URL", service.url)
    monkeypatch.setattr(config, "LANGS", [])
    monkeypatch.setattr(config, "POLL_S", 0.1)
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setattr(w.Models, "get", lambda self, lang: pytest.fail("no local model with AI_TRNS_STT_URL"))
    stop = threading.Event()
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=server) as http:
            job_id = http.post("/jobs", headers=H, data={"url": MEDIA, "format": "srt,txt"}).json()["id"]
            job = wait_status(http, job_id, ("done", "error"))
            assert (job["status"], job["worker"]) == ("done", "service-worker"), job
            assert http.get(f"/jobs/{job_id}/result", headers=H).text == SRT
            assert http.get(f"/jobs/{job_id}/result", headers=H, params={"format": "txt"}).text.startswith("שלום,")
    finally:
        stop.set()
        thread.join(timeout=10)
    assert service.urls == [MEDIA]


def test_uploads_for_transcription_are_refused_when_off(client, api_config, audio_file, monkeypatch):
    monkeypatch.setattr(api_config, "TRANSCRIBE_UPLOADS", False)
    with open(audio_file, "rb") as f:
        r = client.post("/jobs", headers=H, files={"file": ("in.mp3", f)})
    assert r.status_code == 422 and "url" in r.json()["detail"]
    with open(audio_file, "rb") as f:
        r = client.post("/tadiff/compare", headers=H, files={"audio": ("in.mp3", f)}, data={"text": "שלום"})
    assert r.status_code == 422
    assert client.post("/jobs", headers=H, data={"url": MEDIA}).status_code == 202
    # diarize runs on this GPU, its audio may still be uploaded
    with open(audio_file, "rb") as f:
        r = client.post("/diarize/srt", headers=H, files={"audio": ("in.mp3", f),
                                                          "srt": ("in.srt", SRT.encode("utf-8"))})
    assert r.status_code == 202


def test_uploads_and_other_languages_run_on_the_local_model(server, service, tmp_path, audio_file,  # noqa: F811
                                                            monkeypatch):
    """With AI_TRNS_STT_URL the service takes Hebrew URLs only; an upload or another language uses the model here"""
    import httpx
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    from tests.conftest import WORKER_KEY
    from tests.test_worker import FakeTranscriber
    monkeypatch.setattr(config, "API_URL", server)
    monkeypatch.setattr(config, "WORKER_KEY", WORKER_KEY)
    monkeypatch.setattr(config, "WORKER_ID", "mixed-worker")
    monkeypatch.setattr(config, "STT_URL", service.url)
    monkeypatch.setattr(config, "LANGS", [])
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path / "models"))
    for lang in ("he", "en"):
        (tmp_path / "models" / lang).mkdir(parents=True)
    monkeypatch.setattr(config, "POLL_S", 0.1)
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
    used = []
    fake = FakeTranscriber()
    monkeypatch.setattr(w.Models, "get", lambda self, lang: used.append(lang) or fake)
    assert w.available_langs() == ["he", "en", "auto"]
    stop = threading.Event()
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=server) as http:
            with open(audio_file, "rb") as f:
                upload = http.post("/jobs", headers=H, files={"file": ("in.mp3", f)}).json()["id"]
            with open(audio_file, "rb") as f:
                english = http.post("/jobs", headers=H, files={"file": ("in.mp3", f)}, data={"lang": "en"}).json()["id"]
            hebrew = http.post("/jobs", headers=H, data={"url": MEDIA}).json()["id"]
            for job_id in (upload, english, hebrew):
                job = wait_status(http, job_id, ("done", "error"))
                assert job["status"] == "done", job
            assert http.get(f"/jobs/{hebrew}/result", headers=H).text == SRT
    finally:
        stop.set()
        thread.join(timeout=10)
    assert service.urls == [MEDIA]
    assert used == ["he", "en"]


def test_without_a_local_model_the_service_worker_takes_hebrew_only(tmp_path, monkeypatch):
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    monkeypatch.setattr(config, "STT_URL", "http://stt.example")
    monkeypatch.setattr(config, "LANGS", [])
    monkeypatch.setattr(config, "MODELS_DIR", str(tmp_path))
    assert w.available_langs() == ["he"]
