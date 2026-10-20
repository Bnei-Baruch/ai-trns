import os
import time

from ai_trns import protocol
from tests.conftest import H, W, words_doc


def create(client, audio_file, **data):
    with open(audio_file, "rb") as f:
        r = client.post("/jobs", headers=H, files={"file": ("in.mp3", f)},
                        data={"lang": "he", "format": "srt,txt", **data})
    assert r.status_code == 202, r.text
    return r.json()["id"]


def lease(client, worker="w1", langs=("he",)):
    return client.post("/worker/lease", headers=W, json={"worker": worker, "langs": list(langs)})


def test_keys_and_protocol_version_are_checked(client, audio_file):
    assert client.post("/jobs", data={"url": "http://cdn.example.org/a.mp3"}).status_code == 401
    assert client.post("/worker/lease", headers=H, json={"worker": "w", "langs": ["he"]}).status_code == 401
    old = {**W, protocol.VERSION_HEADER: "0"}
    assert client.post("/worker/lease", headers=old, json={"worker": "w", "langs": ["he"]}).status_code == 409


def test_job_goes_through_a_worker(client, audio_file, api_config):
    job_id = create(client, audio_file)
    assert lease(client, langs=["ru"]).status_code == 204  # no job in a language the worker lacks

    job = lease(client).json()
    assert job["id"] == job_id and job["params"]["formats"] == ["srt", "txt"]
    audio = client.get(job["source"]["upload"], headers=W, params={"worker": "w1"})
    assert audio.content == audio_file.read_bytes()
    assert client.get(job["source"]["upload"], headers=W, params={"worker": "other"}).status_code == 409

    beat = client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W,
                       json={"worker": "w1", "stage": "transcribing", "progress": 0.5})
    assert beat.json() == {"cancel": False}
    status = client.get(f"/jobs/{job_id}", headers=H).json()
    assert (status["status"], status["stage"], status["progress"], status["worker"]) == ("running", "transcribing", 0.5, "w1")
    assert client.get("/health").json()["workers"]["w1"]["job"] == job_id

    r = client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=words_doc())
    assert r.status_code == 200, r.text
    assert client.get(f"/jobs/{job_id}", headers=H).json()["status"] == "done"
    assert client.get(f"/jobs/{job_id}/result", headers=H).text == "1\n00:00:00,000 --> 00:00:00,900\nשלום עולם.\n"
    assert client.get(f"/jobs/{job_id}/result?format=txt", headers=H).text == "שלום עולם."
    assert "‫" in client.get(f"/jobs/{job_id}/result?format=srt&fix_rtl=true", headers=H).text
    assert client.get(f"/jobs/{job_id}/result?format=json", headers=H).json()["words"][1]["word"] == "עולם."
    assert not os.path.exists(os.path.join(api_config.JOBS_DIR, job_id, "upload"))


def test_cancel_reaches_the_worker_and_its_result_is_refused(client, audio_file):
    job_id = create(client, audio_file)
    lease(client)
    assert client.delete(f"/jobs/{job_id}", headers=H).json()["status"] == "cancelled"
    beat = client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W, json={"worker": "w1", "progress": 0.3})
    assert beat.json() == {"cancel": True}
    r = client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=words_doc())
    assert r.status_code == 409
    assert client.delete(f"/jobs/{job_id}", headers=H).status_code == 409


def test_lost_worker_job_is_retried_then_failed(client, audio_file, api_config, monkeypatch):
    monkeypatch.setattr(api_config, "MAX_ATTEMPTS", 2)
    store = client.app.state.store
    job_id = create(client, audio_file)
    for attempt, worker in ((1, "w1"), (2, "w2")):
        assert lease(client, worker).json()["id"] == job_id
        store.update(job_id, lease_until=time.time() - 1)
        store.expire_leases(api_config.MAX_ATTEMPTS)
    job = client.get(f"/jobs/{job_id}", headers=H).json()
    assert (job["status"], job["attempts"]) == ("error", 2)
    # the first worker coming back late cannot hand in a result any more
    assert client.post(f"/worker/jobs/{job_id}/heartbeat", headers=W, json={"worker": "w1"}).json() == {"cancel": True}


def test_expired_lease_goes_back_to_the_queue(client, audio_file, api_config):
    store = client.app.state.store
    job_id = create(client, audio_file)
    lease(client, "w1")
    store.update(job_id, lease_until=time.time() - 1)
    assert store.expire_leases(api_config.MAX_ATTEMPTS) == [job_id]
    assert client.get(f"/jobs/{job_id}", headers=H).json()["status"] == "queued"
    assert lease(client, "w2").json()["id"] == job_id


def test_worker_failure_and_bad_document(client, audio_file):
    first, second = create(client, audio_file), create(client, audio_file)
    lease(client)
    client.post(f"/worker/jobs/{first}/fail", headers=W, json={"worker": "w1", "error": "RuntimeError: boom"})
    job = client.get(f"/jobs/{first}", headers=H).json()
    assert (job["status"], job["error"]) == ("error", "RuntimeError: boom")
    lease(client)
    r = client.put(f"/worker/jobs/{second}/result", headers=W, params={"worker": "w1"}, json={"words": []})
    assert r.status_code == 422
    assert client.get(f"/jobs/{second}", headers=H).json()["status"] == "error"


def test_url_jobs_and_render(client):
    assert client.post("/jobs", headers=H, data={"url": "http://evil.example.com/a.mp3"}).status_code == 422
    r = client.post("/jobs", headers=H, data={"url": "https://cdn.example.org/a.mp3"})
    job = lease(client).json()
    assert job["id"] == r.json()["id"] and job["source"] == {"url": "https://cdn.example.org/a.mp3"}
    assert client.post("/render?format=txt", headers=H, json=words_doc()).text == "שלום עולם."
    assert client.post("/render", headers=H, json={"x": 1}).status_code == 422
