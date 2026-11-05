"""A lesson's source text found by its uid (ai_trns.sources) and used by /diarize/* (no network: kmedia mocked)"""
import io
import json
import os
import time

import docx
import pytest

from ai_trns import sources
from tests.conftest import H, W

SOURCE = ("ובזה תבין את דברי חז\"ל שאמרו שהעולם נברא בשביל התורה והישראל ולכן צריכים להבין את המטרה הזו "
          "ושכל הבריאה כולה אינה אלא כדי להביא את האדם לדבקות בבורא")
OTHER = "מאמר אחר לגמרי על נושא שונה שאין לו קשר לשיעור הזה בכלל ולא נקרא בו אף פעם אחת"


def docx_bytes(*paragraphs):
    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def he(fid, name, **kw):
    return {"id": fid, "name": name, "language": "he", "type": "text", **kw}


LESSON = {
    "id": "Lesson01", "sources": ["Src00001"],
    "derived_units": {"x": {"id": "Likut001", "content_type": "LIKUTIM",
                            "files": [he("LikDocx1", "likut.docx"), {**he("LikEng01", "likut.docx"), "language": "en"}]},
                      "y": {"id": "Other001", "content_type": "ARTICLE", "files": [he("ArtDocx1", "a.docx")]}},
    "files": [he("Ins00001", "ins.docx", insert_type="likutim"), he("Tml00001", "t.doc", insert_type="tamlil")],
}
SRC_UNIT = {"content_units": [{"id": "Src00001", "files": [
    he("SrcPdf01", "s.pdf"), he("SrcDocx1", "s.docx"), he("Scan0001", "scan.docx", insert_type="source-scan"),
    {**he("SrcMp301", "s.mp3"), "type": "audio"}]}]}


@pytest.fixture
def kmedia(monkeypatch):
    """URL -> body (bytes, or an exception to raise); the URLs asked, in order"""
    routes = {
        f"{sources.BACKEND}/content_units/Lesson01": json.dumps(LESSON).encode(),
        f"{sources.BACKEND}/content_units?id=Src00001&with_files=true": json.dumps(SRC_UNIT).encode(),
        f"{sources.CDN}/SrcDocx1": docx_bytes("כותרת", SOURCE),
        f"{sources.CDN}/LikDocx1": docx_bytes(OTHER),
        f"{sources.CDN}/Ins00001": docx_bytes("קטע קצר"),
    }
    asked = []

    def get(url, timeout):
        asked.append(url)
        body = routes.get(url, OSError(f"404 {url}"))
        if isinstance(body, Exception):
            raise body
        return body
    monkeypatch.setattr(sources, "_get", get)
    return routes, asked


def test_text_files_of_a_lesson(kmedia):
    assert sources.text_files("Lesson01") == [
        {"id": "SrcDocx1", "name": "s.docx", "unit": "Src00001", "kind": "source"},   # .docx over .pdf, no scan
        {"id": "LikDocx1", "name": "likut.docx", "unit": "Likut001", "kind": "likutim"},
        {"id": "Ins00001", "name": "ins.docx", "unit": "Lesson01", "kind": "likutim"},
    ]


def test_fetch_reads_the_texts_and_caches_them(kmedia, tmp_path):
    routes, asked = kmedia
    cache = sources.Cache(str(tmp_path / "cache"))
    found = sources.fetch("Lesson01", cache)
    assert [f["id"] for f in found["files"]] == ["SrcDocx1", "LikDocx1", "Ins00001"]
    assert SOURCE in found["files"][0]["text"] and "error" not in found and found["skipped"] == []
    asked.clear()
    again = sources.fetch("Lesson01", cache)
    assert again["files"] == found["files"]
    assert not any(u.startswith(sources.CDN) for u in asked)   # the files from the cache


def test_fetch_never_raises(kmedia, tmp_path):
    routes, _ = kmedia
    routes[f"{sources.CDN}/LikDocx1"] = b"not a docx"
    found = sources.fetch("Lesson01")
    assert [f["id"] for f in found["files"]] == ["SrcDocx1", "Ins00001"]
    assert found["skipped"][0]["id"] == "LikDocx1"
    routes[f"{sources.BACKEND}/content_units/Lesson01"] = TimeoutError("timed out")
    assert sources.fetch("Lesson01") == {"uid": "Lesson01", "files": [], "skipped": [], "error": "kmedia: timed out"}
    assert sources.fetch("../../x")["error"] == "bad uid"


def test_a_pdf_source_is_split_into_words_or_skipped(kmedia, monkeypatch):
    routes, _ = kmedia
    lesson = {**LESSON, "derived_units": {}, "files": []}
    routes[f"{sources.BACKEND}/content_units/Lesson01"] = json.dumps(lesson).encode()
    unit = {"content_units": [{"id": "Src00001", "files": [he("SrcPdf01", "s.pdf")]}]}
    routes[f"{sources.BACKEND}/content_units?id=Src00001&with_files=true"] = json.dumps(unit).encode()
    routes[f"{sources.CDN}/SrcPdf01"] = b"%PDF"
    monkeypatch.setattr(sources, "pdf_letters", lambda content: "ובזהתביןאתדברי\nשהעולםנברא")
    found = sources.fetch("Lesson01", vocab_text="ובזה תבין את דברי חז\"ל שאמרו שהעולם נברא")
    assert found["files"][0]["text"] == "ובזה תבין את דברי\nשהעולם נברא"
    assert sources.fetch("Lesson01")["skipped"][0]["reason"] == "pdf: no words"   # nothing to split by

    def no_pypdf(content):
        raise ImportError("pypdf")
    monkeypatch.setattr(sources, "pdf_letters", no_pypdf)
    assert sources.fetch("Lesson01", vocab_text="ובזה")["skipped"][0]["reason"] == "pypdf not installed"


def test_segment_splits_letters_by_the_vocabulary():
    assert sources.segment("ובזהתבין", {"ובזה", "תבין"}, 5) == "ובזה תבין"
    assert sources.segment("אבג", set(), 5) == "א ב ג"


def test_cache_keeps_its_size_and_age(tmp_path):
    cache = sources.Cache(str(tmp_path), max_mb=0.002, max_days=1)    # ~2 KB
    cache.put("Aaaaaaa1", "א" * 500)       # 1000 bytes
    old = time.time() - 100
    os.utime(tmp_path / "Aaaaaaa1.txt", (old, old))
    cache.put("Bbbbbbb2", "ב" * 500)
    cache.put("Ccccccc3", "ג" * 500)       # 3000 bytes > 2 KB: the least recently used goes
    assert cache.get("Aaaaaaa1") is None and cache.get("Ccccccc3") == "ג" * 500
    stale = time.time() - 2 * 86400
    os.utime(tmp_path / "Bbbbbbb2.txt", (stale, stale))
    assert cache.get("Bbbbbbb2") is None
    cache.prune()
    assert sorted(os.listdir(tmp_path)) == ["Ccccccc3.txt"]
    with pytest.raises(ValueError):
        cache.get("../x")


def test_select_drops_a_text_the_transcript_does_not_read():
    words = [{"word": w} for w in ("שלום חברים " + SOURCE + " נכון").split()]
    text, info = sources.select([{"id": "a", "name": "s.docx", "kind": "source", "text": SOURCE},
                                 {"id": "b", "name": "likut.docx", "kind": "likutim", "text": OTHER}], words)
    assert text == SOURCE
    assert [(i["id"], i["matched_words"], i["used"]) for i in info] == [("a", 28, True), ("b", 0, False)]
    assert sources.select([], words) == (None, [])


# The API: /diarize/words with uid

SRT = "1\n00:00:00,000 --> 00:00:30,000\n" + SOURCE + "\n\n2\n00:00:31,000 --> 00:00:33,000\nשלום חברים היום נלמד\n"


def reading_doc():
    words = [(w, "SPEAKER_02", 0) for w in SOURCE.split()] + [(w, "Rav", 1) for w in "שלום חברים היום נלמד".split()]
    return {"version": 1, "kind": "diarize", "duration_s": 40.0, "language": "he", "model": "fake",
            "words": [{"word": w, "start": float(i), "end": i + 0.5, "speaker": s, "cue": c}
                      for i, (w, s, c) in enumerate(words)],
            "stats": {"speakers": 2, "named": 1, "mixed_cues": 0, "vad_fallback_cues": 0}}


def post(client, audio_file, data, files=()):
    with open(audio_file, "rb") as f:
        return client.post("/diarize/words", headers=H, data={"paragraphs": "true", **data},
                           files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode())), *files])


def finish(client, job_id, doc):
    job = client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": [], "engines": ["diarize"]}).json()
    assert job["id"] == job_id
    assert client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=doc).status_code == 200
    return client.get(f"/jobs/{job_id}", headers=H).json()


def test_diarize_finds_the_source_by_uid(client, audio_file, kmedia, tmp_path):
    r = post(client, audio_file, {"uid": "Lesson01"})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    assert r.json()["params"]["uid"] == "Lesson01"
    assert (tmp_path / "jobs" / job_id / "sources.json").exists()
    assert (tmp_path / "jobs" / "sources-cache" / "SrcDocx1.txt").exists()
    stats = finish(client, job_id, reading_doc())["stats"]
    assert [(s["id"], s["kind"], s["matched_words"], s["used"]) for s in stats["sources"]] == [
        ("SrcDocx1", "source", 28, True), ("LikDocx1", "likutim", 0, False), ("Ins00001", "likutim", 0, False)]
    paragraphs = client.get(f"/jobs/{job_id}/result?format=paragraphs", headers=H).json()
    assert [p["label"] for p in paragraphs["paragraphs"]] == ["קריין", "רב"]
    assert paragraphs["stats"]["sources"] == stats["sources"] and paragraphs["stats"]["reader_turns"] == 1


def test_an_uploaded_source_wins_over_uid(client, audio_file, kmedia, tmp_path):
    _, asked = kmedia
    r = post(client, audio_file, {"uid": "Lesson01"}, [("source", ("mine.txt", SOURCE.encode()))])
    assert r.status_code == 202, r.text
    assert asked == [] and not (tmp_path / "jobs" / r.json()["id"] / "sources.json").exists()
    stats = finish(client, r.json()["id"], reading_doc())["stats"]
    assert stats["sources"] == [{"name": "mine.txt", "kind": "upload", "matched_words": 28, "used": True}]


def test_without_a_source_the_job_goes_on(client, audio_file, kmedia):
    routes, _ = kmedia
    routes[f"{sources.BACKEND}/content_units/Lesson01"] = TimeoutError("timed out")
    r = post(client, audio_file, {"uid": "Lesson01"})
    assert r.status_code == 202, r.text
    job = finish(client, r.json()["id"], reading_doc())
    assert job["status"] == "done"
    assert job["stats"]["sources"] == [] and job["stats"]["sources_error"] == "kmedia: timed out"
    paragraphs = client.get(f"/jobs/{job['id']}/result?format=paragraphs", headers=H).json()
    assert [p["label"] for p in paragraphs["paragraphs"]] == ["תלמיד", "רב"]
    # a job without uid or source: no source stats at all
    r = post(client, audio_file, {})
    assert "sources" not in (finish(client, r.json()["id"], reading_doc())["stats"] or {})
    assert post(client, audio_file, {"uid": "../etc"}).status_code == 400
