"""Lesson markup (ai_trns.markup, ai_trns.api.markup): chapters by the transcript's marks, paragraphs timed by the
heard words, the transcribe -> diarize -> build chain driven by the server"""
import json

import pytest

from ai_trns.api import markup as api
from ai_trns.markup import build
from tests.conftest import H, W, words_doc

PARAS = [
    "",
    "שיעור בוקר - אחרי עריכה",
    "קריין: שיעור בנושא חוק הערבות.",
    "נמשיך אותו כי אנחנו צריכים חיבור בינינו",
    "קריין: קטע מס' 46 מתוך חוק הערבות.",
    '"כל ישראל יש להם חלק לעולם הבא"',
    "(הרב\"ש. מאמר 3)",
    "תתאר לעצמך שאתה הולך להתחתן עם החברים",
    "שאלה: מה זו אותה מערכת שקושרת בין דברים?",
    "זו המערכת של הבורא שקושרת הכול",
    "תלמיד: ולמה דווקא הבורא?",
    "כי הוא המקור של הכול",
    "קריין: קטע הבא, מס' 47, מאור ושמש.",
    '"העיקר הוא שכל אחד ואחד אל יחשוב לעצמו"',
    "מי שעושה מעצמו אפס בחברה מקשר את כולם",
]


def test_chapters_by_the_transcript_marks():
    chapters = build.marked_chapters(PARAS)
    assert [(c["type"], c["p_start"], c["p_end"]) for c in chapters] == [
        ("other", 1, 3), ("reading", 4, 6), ("commentary", 7, 7), ("qa", 8, 11), ("reading", 12, 13),
        ("commentary", 14, 14)]
    # the reader's announcement as written, not a parsed number; the commentary carries its reading's
    assert chapters[1]["title"] == chapters[2]["title"] == "קטע מס' 46 מתוך חוק הערבות."
    assert chapters[3]["title"] == "מה זו אותה מערכת שקושרת בין דברים?"
    assert all("item" not in c for c in chapters)
    assert build.marked_chapters(["רק טקסט", "בלי סימנים"]) is None


def test_items_numbered_any_way_and_students_without_question_marks():
    paras = ["קריין: \"תלמוד עשר הספירות\", חלק שני.", "קריין: אות ט\"ו.", '"טקסט האות"', "הסבר של הרב",
             "תלמיד: שאלה ראשונה?", "תשובה", "תלמיד: שאלה שנייה?", "תלמיד: והמשך שלה?", "תשובה"]
    chapters = build.marked_chapters(paras)
    assert [(c["type"], c["p_start"]) for c in chapters] == [
        ("reading", 0), ("commentary", 3), ("qa", 4), ("qa", 6)]
    assert chapters[0]["title"] == chapters[1]["title"] == '"תלמוד עשר הספירות", חלק שני. · אות ט"ו.'


def spoken(paras, skip=(), gap=0.3):
    """The heard words of the paragraphs, in order, without the marks and the paragraphs in `skip`"""
    words, t = [], 0.0
    for i, p in enumerate(paras):
        if i in skip:
            t += 5
            continue
        label = build.READER if p.startswith(("קריין", '"', "(")) else build.STUDENT if p.startswith(
            ("שאלה", "תלמיד")) else build.RAV
        for tok in build._MARK_RE.sub("", p).split():
            words.append({"word": tok, "start": t, "end": t + 0.25, "label": label, "reading": False})
            t += gap
    return words


def test_paragraphs_get_their_times_from_the_heard_words():
    words = spoken(PARAS, skip={1})  # the header is not spoken
    paragraphs, chapters, stats = build.from_transcript(PARAS, words, duration=60.0)
    at = {p["i"]: p for p in paragraphs}
    assert stats == {"transcript_paragraphs": 14, "aligned_paragraphs": 13}
    assert at[4]["start"] < at[5]["start"] < at[7]["start"] < at[8]["start"]
    assert at[1]["start"] == 0.0 and at[1]["end"] <= at[2]["start"]  # not heard: the gap before the next one
    assert [at[i]["speaker"] for i in (2, 5, 7, 8, 9, 10)] == [build.READER, build.READER, build.RAV,
                                                              build.STUDENT, build.RAV, build.STUDENT]
    # a chapter runs until the next one; the first from 0, the last to the end of the media
    assert chapters[0]["start"] == 0.0 and chapters[-1]["end"] == 60.0
    assert all(a["end"] == b["start"] for a, b in zip(chapters, chapters[1:]))
    assert chapters[1]["start"] == at[4]["start"]


def test_a_repeated_phrase_does_not_move_a_paragraph_back():
    paras = ["קריין: קטע מס' 1.", '"אחד שתיים שלוש ארבע"', "הסבר ארוך על הקטע הזה", "שאלה: אחד שתיים שלוש ארבע?"]
    words = spoken(paras[:3])
    spans = build.align(paras, words)
    assert spans[3] is None or spans[3][0] >= spans[2][0]


def test_chapters_from_the_audio_without_a_transcript():
    ps = [{"label": "קריין", "reading": False, "start": 0, "end": 5, "text": "קטע מס 3", "words": []},
          {"label": "קריין", "reading": True, "start": 5, "end": 30, "text": "טקסט המקור", "words": []},
          {"label": "הרב", "reading": False, "start": 30, "end": 90, "text": "הסבר", "words": []},
          {"label": "תלמיד", "reading": False, "start": 90, "end": 95, "text": "למה?", "words": []},
          {"label": "הרב", "reading": False, "start": 95, "end": 120, "text": "כי", "words": []},
          {"label": "תלמיד", "reading": False, "start": 120, "end": 125, "text": "ועוד?", "words": []}]
    paragraphs, chapters, _ = build.from_audio(ps, 130.0)
    assert [(c["type"], c["p_start"], c["p_end"]) for c in chapters] == [
        ("other", 0, 0), ("reading", 1, 1), ("commentary", 2, 2), ("qa", 3, 4), ("qa", 5, 5)]
    assert paragraphs[3] == {"i": 3, "start": 90, "end": 95, "speaker": "תלמיד", "text": "למה?"}


def test_markup_document_falls_back_to_the_audio_without_marks():
    ps = [{"label": "הרב", "reading": False, "start": 0, "end": 5, "text": "שלום",
           "words": [{"word": "שלום", "start": 0, "end": 1}]}]
    doc = build.markup("Unit0001", "he", {"file_uid": "Audio001", "duration_s": 10}, ps, "<p>בלי סימנים</p>",
                       {"file_uid": "Text0001"}, {"id": "r1"})
    assert doc["run"]["source"] == "audio" and doc["transcript"] is None
    assert [c["type"] for c in doc["chapters"]] == ["other"]


# --- the API

UNIT = {"id": "Unit0001", "duration": 60, "files": [
    {"id": "AudioTr1", "type": "audio", "language": "he", "mimetype": "audio/mpeg", "name": "heb_t_x.mp3",
     "duration": 60},
    {"id": "AudioOr1", "type": "audio", "language": "he", "mimetype": "audio/mpeg", "name": "heb_o_x.mp3",
     "duration": 60},
    {"id": "TextPdf1", "type": "text", "language": "he", "insert_type": "tamlil", "name": "heb_o_x.pdf"},
    {"id": "TextDoc1", "type": "text", "language": "he", "insert_type": "tamlil", "name": "heb_o_x.doc"},
]}


@pytest.fixture
def mc(client, api_config, tmp_path, monkeypatch):
    monkeypatch.setattr(api_config, "MARKUP_DIR", str(tmp_path / "markup"))
    monkeypatch.setattr(api_config, "ALLOWED_URL_HOSTS", ["cdn.kabbalahmedia.info"])

    def fetch_unit(uid):
        if uid != "Unit0001":
            raise LookupError(uid)
        return UNIT

    monkeypatch.setattr(api, "fetch_unit", fetch_unit)
    monkeypatch.setattr(api, "fetch_transcript_html", lambda file_id: "".join(f"<p>{p}</p>" for p in PARAS))
    monkeypatch.setattr(api, "fetch_sources", lambda *a: None)
    return client


def lease(client, engine):
    r = client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": ["he"], "engines": [engine]})
    assert r.status_code == 200, r.text
    return r.json()


def diarize_doc(paras):
    words = spoken(paras, skip={1})
    speaker = {build.RAV: "Rav", build.READER: "Reader", build.STUDENT: "SPEAKER_01"}
    return {"version": 1, "kind": "diarize", "duration_s": 60.0, "language": "he", "model": "fake",
            "words": [{"word": w["word"], "start": w["start"], "end": w["end"], "speaker": speaker[w["label"]],
                       "cue": 0} for w in words],
            "stats": {"speakers": 3, "named": 2}}


def test_no_login_to_start_or_read(mc, api_config):
    # authorization is temporarily off for POST /markup (TODO.md)
    assert mc.get("/markup/Unit0001").status_code == 404
    r = mc.post("/markup", data={"uid": "Unit0001"})
    assert r.status_code == 202
    # who started it is in the request log
    with open(f"{api_config.LOG_DIR}/requests.log", encoding="utf-8") as f:
        last = json.loads(f.readlines()[-1])
    assert (last["path"], last["note"]) == \
           ("/markup", f"uid=Unit0001 language=he force=False job={r.json()['transcribe_job']}")
    assert mc.get("/markup/bad").status_code == 400
    assert mc.post("/markup", headers=H, data={"uid": "Missing1"}).status_code == 404


def test_the_server_drives_transcribe_then_diarize_then_builds(mc):
    r = mc.post("/markup", headers=H, data={"uid": "Unit0001"})
    assert r.status_code == 202, r.text
    state = r.json()
    assert (state["status"], state["stage"], state["media"]["file_uid"], state["transcript"]) == \
           ("pending", "transcribe", "AudioOr1", {"file_uid": "TextDoc1"})
    assert "owner" not in state
    # a second start while it runs is the same run
    assert mc.post("/markup", headers=H, data={"uid": "Unit0001"}).json()["run_id"] == state["run_id"]

    job = lease(mc, "transcribe")
    assert job["id"] == state["transcribe_job"] and job["source"]["url"] == "https://cdn.kabbalahmedia.info/AudioOr1"
    assert mc.put(f"/worker/jobs/{job['id']}/result", headers=W, params={"worker": "w1"},
                  json=words_doc()).status_code == 200
    state = mc.get("/markup/Unit0001/status").json()
    assert state["stage"] == "diarize" and state["diarize_job"]

    job = lease(mc, "diarize")
    assert job["id"] == state["diarize_job"] and job["params"]["paragraphs"] is True
    assert job["params"]["uid"] == "Unit0001"
    assert mc.put(f"/worker/jobs/{job['id']}/result", headers=W, params={"worker": "w1"},
                  json=diarize_doc(PARAS)).status_code == 200
    api.sync_all(mc.app.state.store)
    state = mc.get("/markup/Unit0001/status").json()
    assert state["status"] == "ready", state

    r = mc.get("/markup/Unit0001")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    doc = r.json()
    assert doc["run"]["source"] == "transcript" and doc["run"]["id"] == state["run_id"]
    assert doc["transcript"] == {"file_uid": "TextDoc1"}
    assert [c["type"] for c in doc["chapters"]] == ["other", "reading", "commentary", "qa", "reading", "commentary"]
    assert doc["stats"]["aligned_paragraphs"] == 13


def test_a_failed_job_fails_the_run(mc):
    state = mc.post("/markup", headers=H, data={"uid": "Unit0001"}).json()
    lease(mc, "transcribe")
    mc.post(f"/worker/jobs/{state['transcribe_job']}/fail", headers=W, json={"worker": "w1", "error": "boom"})
    state = mc.get("/markup/Unit0001/status").json()
    assert state["status"] == "failed" and "boom" in state["error"]
    # and can be started again
    assert mc.post("/markup", headers=H, data={"uid": "Unit0001"}).json()["status"] == "pending"
