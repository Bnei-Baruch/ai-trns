"""Review sets (moved from stt-api): lessons are transcribed by jobs, reviewers act as themselves"""
import pytest

from ai_trns.api import review
from tests.conftest import H, W, words_doc
from tests.test_auth import oidc, token  # noqa: F401 (fixture)

ALICE, BOB = "alice", "bob"
UNITS = {
    "5JBUHZoo": {"duration": 4710, "files": [
        {"id": "1lb9IZdV", "type": "audio", "duration": 4710},
        {"id": "PpgIuLT6", "type": "video", "duration": 4710},
    ]},
    "Unit0002": {"duration": 60, "files": [{"id": "Audio002", "type": "audio", "duration": 60}]},
}
SET = "/review/sets/set1"


@pytest.fixture
def rc(client, api_config, oidc, tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "REVIEW_DIR", str(tmp_path / "review"))
    monkeypatch.setattr(api_config, "REVIEW_MODEL", "")

    def fetch_unit(uid):
        if uid not in UNITS:
            raise LookupError(uid)
        return UNITS[uid]

    monkeypatch.setattr(review, "fetch_unit", fetch_unit)
    return client


def lesson(uid="5JBUHZoo", audio="1lb9IZdV", **extra):
    return {"unit_uid": uid, "title": f"lesson {uid}", "audio_file_id": audio, "duration": 4710,
            "reviewer_name": "Alice", **extra}


def add(client, body=None, who=ALICE):
    return client.post(f"{SET}/segments", headers=token(who), json=body or lesson())


def run_job(client, job_id, doc=None, error=None):
    job = client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": ["he"]}).json()
    assert job["id"] == job_id and job["source"]["url"].startswith("https://cdn.kabbalahmedia.info/")
    if error:
        client.post(f"/worker/jobs/{job_id}/fail", headers=W, json={"worker": "w1", "error": error})
    else:
        r = client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=doc or words_doc())
        assert r.status_code == 200, r.text


def ready(client, body=None):
    segment = add(client, body).json()
    run_job(client, segment["job_id"])
    return segment["id"]


def put(client, segment, who=ALICE, reviewer=None, **body):
    body.setdefault("status", "in_progress")
    return client.put(f"{SET}/reviewers/{reviewer or who}/segments/{segment}", headers=token(who), json=body)


def test_login_is_required(rc):
    assert rc.get(SET).status_code == 401
    assert rc.get(SET, headers=token(ALICE)).json() == {"id": "set1", "model": None, "segments": []}


def test_lesson_is_pending_until_its_job_is_done(rc):
    r = add(rc)
    assert r.status_code == 202, r.text
    segment = r.json()
    assert (segment["status"], segment["added_by"], segment["end"]) == ("pending", ALICE, 4710)
    assert segment["media_url"] == "https://cdn.kabbalahmedia.info/1lb9IZdV"
    # the reviewer can follow the job like their own
    assert rc.get(f"/jobs/{segment['job_id']}", headers=token(ALICE)).json()["params"]["formats"] == ["json"]
    assert put(rc, segment["id"]).status_code == 409

    run_job(rc, segment["job_id"])
    review_set = rc.get(SET, headers=token(BOB)).json()
    assert review_set["model"] == "fake"
    [segment] = review_set["segments"]
    assert segment["status"] == "ready" and segment["model"] == "fake"
    assert segment["chunks"] == [{"start": 0.0, "end": 0.9, "text": "שלום עולם."}]


def test_maintenance_sync_finishes_lessons(rc):
    segment = add(rc).json()
    run_job(rc, segment["job_id"])
    review.sync_all(rc.app.state.store)
    assert review.load_segment("set1", segment["id"])["status"] == "ready"


def test_failed_job_fails_the_lesson_and_it_can_be_added_again(rc):
    segment = add(rc).json()
    assert add(rc).status_code == 409
    run_job(rc, segment["job_id"], error="ffmpeg: bad file")
    [failed] = rc.get(SET, headers=token(ALICE)).json()["segments"]
    assert (failed["status"], failed["error"]) == ("failed", "ffmpeg: bad file")
    again = add(rc).json()
    assert again["status"] == "pending" and again["job_id"] != segment["job_id"] and "error" not in again


def test_lesson_without_job_fails(rc):
    """Pending lessons copied from stt-api had no job"""
    segment = add(rc).json()
    review.save_segment("set1", {k: v for k, v in segment.items() if k != "job_id"})
    assert rc.get(SET, headers=token(ALICE)).json()["segments"][0]["status"] == "failed"


def test_delete_is_soft_and_cancels_the_job(rc):
    segment = add(rc).json()
    assert rc.delete(f"{SET}/segments/{segment['id']}", headers=token(BOB)).status_code == 204
    assert rc.get(SET, headers=token(ALICE)).json()["segments"] == []
    stored = review.load_segment("set1", segment["id"])
    assert stored["deleted"] and stored["deleted_by"] == BOB
    assert rc.get(f"/jobs/{segment['job_id']}", headers=H).json()["status"] == "cancelled"
    assert rc.delete(f"{SET}/segments/{segment['id']}", headers=token(BOB)).status_code == 404
    assert add(rc).status_code == 202  # a deleted lesson can be added again


@pytest.mark.parametrize("change", [{"unit_uid": "Missing0"}, {"audio_file_id": "PpgIuLT6"},
                                    {"video_file_id": "1lb9IZdV"}, {"unit_uid": "../etc"}, {"title": ""}])
def test_invalid_lesson_is_400(rc, change):
    assert add(rc, lesson(**change)).status_code == 400


def test_users_act_only_as_themselves(rc):
    seg = ready(rc)
    assert put(rc, seg, who=BOB, reviewer=ALICE).status_code == 403
    assert rc.get(f"{SET}/reviewers/{ALICE}/results", headers=token(BOB)).status_code == 403
    assert add(rc, lesson("Unit0002", "Audio002", reviewer=ALICE), who=BOB).status_code == 403
    # a service names the reviewer
    r = rc.put(f"{SET}/reviewers/{BOB}/segments/{seg}", headers=H, json={"status": "in_progress"})
    assert r.json()["reviewer"] == BOB
    assert rc.put(f"{SET}/reviewers/..%2Fx/segments/{seg}", headers=H, json={"status": "done"}).status_code in (400, 404)


def test_results(rc):
    seg = ready(rc)
    r = put(rc, seg, chunks=[{"index": 0, "corrected": "שלום עולם!", "category": None}], position=0.5)
    assert r.status_code == 200 and r.json()["reviewer"] == ALICE
    # done: a changed chunk needs a category
    assert put(rc, seg, status="done", chunks=[{"index": 0, "corrected": "x"}]).status_code == 400
    assert put(rc, seg, status="done", chunks=[{"index": 1, "corrected": "x", "category": "A"}]).status_code == 400
    assert put(rc, seg, status="done", chunks=[{"index": 0, "corrected": "x", "category": "A"}]).status_code == 200
    put(rc, seg, who=BOB, status="in_progress")

    mine = rc.get(f"{SET}/reviewers/{ALICE}/results", headers=token(ALICE)).json()
    assert [(r["segment_id"], r["status"]) for r in mine] == [(seg, "done")]
    everything = rc.get(f"{SET}/results", headers=token(BOB)).json()
    assert {r["reviewer"] for r in everything["results"]} == {ALICE, BOB}
    assert everything["set"]["segments"][0]["id"] == seg


def test_baseline_is_hidden_from_reviewers(rc):
    seg = ready(rc)
    # the finished job's outcome is written in first (the API does it as soon as the job is done)
    assert rc.get(SET, headers=token(ALICE)).json()["segments"][0]["status"] == "ready"
    review.save_segment("set1", {**review.load_segment("set1", seg), "baseline_chunks": [{"start": 0, "end": 1, "text": "b"}]})
    assert "baseline_chunks" not in rc.get(SET, headers=token(ALICE)).json()["segments"][0]
    assert "baseline_chunks" in rc.get(f"{SET}/results", headers=token(ALICE)).json()["set"]["segments"][0]


def test_configured_model_wins(rc, api_config, monkeypatch):
    monkeypatch.setattr(api_config, "REVIEW_MODEL", "kabbalah-v3")
    ready(rc)
    assert rc.get(SET, headers=token(ALICE)).json()["model"] == "kabbalah-v3"


def test_bad_set_id(rc):
    assert rc.get("/review/sets/.hidden", headers=token(ALICE)).status_code == 400
