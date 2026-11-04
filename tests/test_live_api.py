"""Live jobs: the API's events and stop, and the real worker's live engine against a real API server"""
import functools
import http.server
import json
import threading
import time
from types import SimpleNamespace as NS

import pytest

from tests.conftest import H, W, words_doc
from tests.test_auth import oidc  # noqa: F401  (fixture)
from tests.test_worker import server, wait_status  # noqa: F401  (fixtures)


def sse(text: str) -> list:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def start_live(client):
    r = client.post("/live", headers=H, data={"url": "http://cdn.example.org/live.mp3"})
    assert r.status_code == 202, r.text
    return r.json()


def test_live_job_events_stop_and_result(client):
    job = start_live(client)
    job_id = job["id"]
    assert job["params"]["kind"] == "live" and job["params"]["formats"] == ["srt", "txt", "json"]
    # a transcription worker does not take it
    assert client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": ["he"]}).status_code == 204
    leased = client.post("/worker/lease", headers=W, json={"worker": "w2", "langs": [], "engines": ["live"]}).json()
    assert leased["id"] == job_id and leased["source"] == {"url": "http://cdn.example.org/live.mp3"}

    word = {"word": "שלום", "start_s": 0.0, "end_s": 0.4, "probability": 0.9}
    events = [{"type": "start", "words": []}, {"type": "final", "words": [word]},
              {"type": "partial", "words": [{**word, "word": "עו"}]}]
    assert client.post(f"/worker/jobs/{job_id}/events", headers=W,
                       json={"worker": "w2", "events": events}).json() == {"stop": False}
    assert client.post(f"/worker/jobs/{job_id}/events", headers=W,
                       json={"worker": "other", "events": events}).status_code == 409

    assert client.post(f"/live/{job_id}/stop", headers=H).status_code == 200
    assert client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W, json={"worker": "w2"}).json() == \
        {"cancel": False, "stop": True}
    assert client.post(f"/worker/jobs/{job_id}/events", headers=W,
                       json={"worker": "w2", "events": []}).json() == {"stop": True}

    assert client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w2"},
                      json=words_doc()).status_code == 200
    got = sse(client.get(f"/live/{job_id}/events", headers=H).text)
    assert [e["type"] for e in got] == ["start", "final", "end"]
    assert got[1]["words"] == [word]
    assert client.get(f"/jobs/{job_id}/result", headers=H).text.startswith("1\n")

    # events gone from memory: rebuilt from the result
    client.app.state.live.hubs.clear()
    got = sse(client.get(f"/live/{job_id}/events", headers=H).text)
    assert [e["type"] for e in got] == ["final", "end"]
    assert [w["word"] for w in got[0]["words"]] == ["שלום", "עולם."]


def test_cancelled_live_job_ends_its_events_with_an_error(client):
    job_id = start_live(client)["id"]
    client.post("/worker/lease", headers=W, json={"worker": "w2", "langs": [], "engines": ["live"]})
    assert client.delete(f"/jobs/{job_id}", headers=H).status_code == 200
    got = sse(client.get(f"/live/{job_id}/events", headers=H).text)
    assert got[-1] == {"type": "error", "words": [], "message": "cancelled"}
    assert client.post(f"/live/{job_id}/stop", headers=H).status_code == 409


def test_live_events_of_a_plain_job_are_not_found(client):
    job_id = client.post("/jobs", headers=H, data={"url": "http://cdn.example.org/a.mp3"}).json()["id"]
    assert client.get(f"/live/{job_id}/events", headers=H).status_code == 404


# --- the real worker ---

class FakeLiveTranscriber:
    """Hears four words at fixed times of a stream that is never trimmed (shorter than TRIM_S)"""
    lang, name, meta = "he", "fake", {}
    WORDS = [(" שלום", 0.2, 0.6), (" עולם.", 1.0, 1.5), (" מה", 2.0, 2.4), (" נשמע", 3.0, 3.4)]

    def __init__(self):
        self.model = self

    def transcribe(self, audio, **kwargs):
        end = len(audio) / 16000
        words = [NS(word=w, start=s, end=e, probability=0.9) for w, s, e in self.WORDS if e <= end]
        return [NS(text="".join(w.word for w in words), words=words)], None


@pytest.fixture
def media_url(audio_file, monkeypatch, api_config):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(audio_file.parent))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(api_config, "ALLOWED_URL_HOSTS", ["127.0.0.1"])
    yield f"http://127.0.0.1:{httpd.server_port}/{audio_file.name}"
    httpd.shutdown()


@pytest.fixture
def live_worker(server, tmp_path, monkeypatch):  # noqa: F811
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    monkeypatch.setattr(config, "API_URL", server)
    monkeypatch.setattr(config, "WORKER_KEY", "worker-key")
    monkeypatch.setattr(config, "WORKER_ID", "live-worker")
    monkeypatch.setattr(config, "ENGINES", ["live"])
    monkeypatch.setattr(config, "POLL_S", 0.1)
    monkeypatch.setattr(config, "WORK_DIR", str(tmp_path / "work"))
    fake = FakeLiveTranscriber()
    monkeypatch.setattr(w.Models, "get", lambda self, lang: fake)
    stop = threading.Event()
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    yield fake
    stop.set()
    thread.join(timeout=10)


def test_worker_streams_a_live_job(server, live_worker, media_url):  # noqa: F811
    import httpx
    with httpx.Client(base_url=server, timeout=30) as http:
        job_id = http.post("/live", headers=H, data={"url": media_url}).json()["id"]
        started = time.time()
        got = sse(http.get(f"/live/{job_id}/events", headers=H).text)
        # the 4 s file is read at its natural pace
        assert time.time() - started > 3
        types = [e["type"] for e in got]
        assert types[0] == "start" and types[-1] == "end" and "partial" in types
        finals = [w["word"] for e in got if e["type"] == "final" for w in e["words"]]
        assert finals == ["שלום", "עולם.", "מה", "נשמע"]
        job = wait_status(http, job_id, ("done", "error"))
        assert job["status"] == "done", job
        assert http.get(f"/jobs/{job_id}/result", params={"format": "txt"}, headers=H).text == "שלום עולם. מה נשמע"


def test_worker_stops_a_live_job_with_what_it_heard(server, live_worker, media_url):  # noqa: F811
    import httpx
    with httpx.Client(base_url=server, timeout=30) as http:
        job_id = http.post("/live", headers=H, data={"url": media_url}).json()["id"]
        wait_status(http, job_id, ("running",))
        time.sleep(2.2)
        http.post(f"/live/{job_id}/stop", headers=H)
        job = wait_status(http, job_id, ("done", "error"))
        assert job["status"] == "done", job
        text = http.get(f"/jobs/{job_id}/result", params={"format": "txt"}, headers=H).text
        assert text.startswith("שלום") and "נשמע" not in text


# --- a broadcast is shared ---

STREAM = "http://cdn.example.org/live/heb/playlist.m3u8"


def test_a_broadcast_is_one_job_for_everyone(client, oidc):  # noqa: F811
    from tests.test_auth import token
    u1, u2 = token("u1"), token("u2")
    job = client.post("/live", headers=u1, data={"url": STREAM}).json()
    assert client.post("/live", headers=u2, data={"url": STREAM}).json()["id"] == job["id"]
    # anyone may read it, only its owner may cancel it
    assert client.get(f"/jobs/{job['id']}", headers=u2).status_code == 200
    assert client.delete(f"/jobs/{job['id']}", headers=u2).status_code == 404
    # a recording is still everyone's own
    rec = "http://cdn.example.org/lesson.mp3"
    mine = client.post("/live", headers=u1, data={"url": rec}).json()["id"]
    assert client.post("/live", headers=u2, data={"url": rec}).json()["id"] != mine
    assert client.get(f"/live/{mine}/events", headers=u2).status_code == 404


def test_stop_leaves_while_others_listen_and_finishes_for_the_last(client):
    job_id = client.post("/live", headers=H, data={"url": STREAM}).json()["id"]
    client.post("/worker/lease", headers=W, json={"worker": "w2", "langs": [], "engines": ["live"]})
    hub = client.app.state.live.hubs[job_id]
    hub.listeners = 2  # the caller and someone else
    reply = client.post(f"/live/{job_id}/stop", headers=H).json()
    assert (reply["stopping"], reply["listeners"]) == (False, 1)
    assert "stop" not in client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W, json={"worker": "w2"}).json()
    hub.listeners = 1
    assert client.post(f"/live/{job_id}/stop", headers=H).json()["stopping"] is True
    assert client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W, json={"worker": "w2"}).json()["stop"] is True
    # a new listener meanwhile starts a new job instead of joining one that is finishing
    assert client.post("/live", headers=H, data={"url": STREAM}).json()["id"] != job_id


def test_a_broadcast_nobody_listens_to_is_finished(client):
    from ai_trns.api import live
    job_id = client.post("/live", headers=H, data={"url": STREAM}).json()["id"]
    client.post("/worker/lease", headers=W, json={"worker": "w2", "langs": [], "engines": ["live"]})
    state = client.app.state.live
    state.cleanup(client.app.state.store)
    assert job_id not in state.stopping
    state.hubs[job_id].idle_since -= live.IDLE_STOP_S + 1
    state.cleanup(client.app.state.store)
    assert job_id in state.stopping


def test_hub_counts_its_listeners():
    from ai_trns.live.hub import Hub
    hub = Hub()
    hub.publish({"type": "start", "words": []})
    stream = hub.sse(ping_s=0.01)
    next(stream)
    assert hub.listeners == 1 and hub.idle_s() == 0
    stream.close()
    assert hub.listeners == 0
