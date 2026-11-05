"""/tadiff, /docdiff, /diarize (moved from stt-api) as jobs: created by clients, done by workers, rendered here"""
import io
import json

import docx

from ai_trns.api.docx_marks import RED
from tests.conftest import H, W, words_doc

SRT = "1\n00:00:00,000 --> 00:00:02,000\nשלום עולם טוב\n\n2\n00:00:03,000 --> 00:00:04,000\nמה שלומכם\n"


def finish(client, job_id, doc, engines=("transcribe",)):
    job = client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": ["he"], "engines": list(engines)}).json()
    assert job["id"] == job_id
    r = client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=doc)
    assert r.status_code == 200, r.text
    return job


def test_tadiff_compares_the_job_words_with_the_text(client, audio_file):
    with open(audio_file, "rb") as f:
        r = client.post("/tadiff/compare", headers=H, files={"audio": ("in.mp3", f)}, data={"text": "שלום, יקר עולם."})
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["params"]["kind"] == "tadiff" and job["status"] == "queued"
    finish(client, job["id"], words_doc())

    assert client.get(f"/jobs/{job['id']}", headers=H).json()["stats"] == {
        "text_words": 3, "audio_words": 2, "removed": 1, "added": 0, "changed": 0}
    result = client.get(f"/jobs/{job['id']}/result", headers=H)
    assert result.json()["removed"] == [{"text_idx": [1, 1], "text": "יקר", "at": 0.4}]
    assert result.headers["x-tadiff-removed"] == "1"
    assert client.get(f"/jobs/{job['id']}/result?format=txt", headers=H).text == "שלום עולם."
    assert client.get(f"/jobs/{job['id']}/result?format=docx", headers=H).status_code == 422


def test_tadiff_input_errors(client, audio_file):
    assert client.post("/tadiff/compare", headers=H, data={"text": "  ", "url": "https://cdn.example.org/a"}).status_code == 400
    assert client.post("/tadiff/compare", headers=H, data={"text": "a"}).status_code == 422  # no audio
    assert client.post("/tadiff/compare", data={"text": "a", "url": "https://cdn.example.org/a"}).status_code == 401


def test_tadiff_compare_timestamps_answers_at_once(client):
    words = [{"word": "a", "start": 0, "end": 1}, {"word": "c", "start": 1, "end": 2}]
    r = client.post("/tadiff/compare-timestamps", headers=H,
                    files={"text": ("t.txt", "a b c".encode()), "timestamps": ("w.json", json.dumps(words))})
    assert r.json()["removed"] == [{"text_idx": [1, 1], "text": "b", "at": 1.0}]
    bad = client.post("/tadiff/compare-timestamps", headers=H,
                      files={"text": ("t.txt", b"a"), "timestamps": ("w.json", b"{}")})
    assert bad.status_code == 400


def test_docdiff_highlights_skipped_passages(client, audio_file):
    document = docx.Document()
    document.add_paragraph("שלום אחת שתיים שלוש ארבע חמש")
    document.add_paragraph("עולם.")
    buf = io.BytesIO()
    document.save(buf)
    with open(audio_file, "rb") as f:
        r = client.post("/docdiff/compare", headers=H, data={"url": ""},
                        files={"doc": ("שיעור.docx", buf.getvalue()), "audio": ("in.mp3", f)})
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    finish(client, job_id, words_doc())

    result = client.get(f"/jobs/{job_id}/result", headers=H)
    assert result.headers["content-type"].startswith("application/vnd.openxmlformats")
    assert "filename*=UTF-8''%D7%A9%D7%99%D7%A2%D7%95%D7%A8_marked.docx" in result.headers["content-disposition"]
    assert (result.headers["x-docdiff-removed-total"], result.headers["x-docdiff-removed-marked"]) == ("1", "1")
    marked = docx.Document(io.BytesIO(result.content))
    assert ["".join(r.text for r in p.runs if r.font.highlight_color == RED) for p in marked.paragraphs] == [
        "אחת שתיים שלוש ארבע חמש", ""]
    assert client.get(f"/jobs/{job_id}/result?format=diff", headers=H).json()["stats"]["text_words"] == 7


def test_docdiff_refuses_other_documents(client):
    data = {"url": "https://cdn.example.org/a.mp3"}
    assert client.post("/docdiff/compare", headers=H, data=data, files={"doc": ("a.doc", b"x")}).status_code == 400
    assert client.post("/docdiff/compare", headers=H, data=data, files={"doc": ("a.docx", b"x")}).status_code == 400


def diarize_doc(**stats):
    words = [("שלום", 0.1, 0.5, "Rav", 0), ("עולם", 0.6, 1.0, "Rav", 0), ("טוב", 1.2, 1.9, "SPEAKER_00", 0),
             ("מה", 3.1, 3.4, "SPEAKER_00", 1), ("שלומכם", 3.5, 3.9, "SPEAKER_00", 1)]
    return {"version": 1, "kind": "diarize", "duration_s": 4.0, "language": "he", "model": "fake",
            "words": [{"word": w, "start": s, "end": e, "speaker": sp, "cue": c} for w, s, e, sp, c in words],
            "stats": {"speakers": 2, "named": 1, "mixed_cues": 1, "vad_fallback_cues": 0, **stats}}


def test_diarize_job_goes_to_a_diarize_worker(client, audio_file, api_config):
    with open(audio_file, "rb") as f:
        r = client.post("/diarize/srt", headers=H, data={"sample_names": ["Guest"], "max_speakers": "4"},
                        files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode())),
                               ("samples", ("guest.MP3", b"voice"))])
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["params"]["samples"] == [{"name": "Guest", "file": "sample0.mp3"}]
    assert job["params"]["speakers"] == {"max_speakers": 4}

    # a transcription-only worker doesn't get it
    assert client.post("/worker/lease", headers=W, json={"worker": "w0", "langs": ["he"]}).status_code == 204
    leased = client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": [], "engines": ["diarize"]}).json()
    assert leased["id"] == job["id"]
    files = f"/worker/jobs/{job['id']}/files"
    cues = client.get(f"{files}/cues.json", headers=W, params={"worker": "w1"}).json()
    assert cues[1] == {"start": 3.0, "end": 4.0, "text": "מה שלומכם"}
    assert client.get(f"{files}/sample0.mp3", headers=W, params={"worker": "w1"}).content == b"voice"
    assert client.get(f"{files}/upload", headers=W, params={"worker": "w1"}).status_code == 404
    assert client.get(f"{files}/sample0.mp3", headers=W, params={"worker": "w2"}).status_code == 409

    r = client.put(f"/worker/jobs/{job['id']}/result", headers=W, params={"worker": "w1"}, json=diarize_doc())
    assert r.status_code == 200, r.text
    result = client.get(f"/jobs/{job['id']}/result", headers=H)
    assert result.text == ("1\n00:00:00,000 --> 00:00:01,000\nהרב: שלום עולם\n\n"
                           "2\n00:00:01,200 --> 00:00:02,000\nתלמיד: טוב\n\n"
                           "3\n00:00:03,000 --> 00:00:04,000\nתלמיד: מה שלומכם\n")
    assert (result.headers["x-diarize-speakers"], result.headers["x-diarize-split-cues"]) == ("2", "1")
    words = client.get(f"/jobs/{job['id']}/result?format=json", headers=H).json()
    srt = client.post("/diarize/words/srt", headers=H, json={**words, "pad": False})
    assert srt.text == "1\n00:00:00,100 --> 00:00:03,900\nהרב: שלום עולם\nתלמיד: טוב מה שלומכם\n"


def test_diarize_input_errors_and_a_bad_document(client, audio_file):
    url = {"url": "https://cdn.example.org/a.mp3"}
    assert client.post("/diarize/words", headers=H, data=url, files={"srt": ("a.srt", b"junk")}).status_code == 400
    r = client.post("/diarize/words", headers=H, data={**url, "sample_names": ["A", "B"]},
                    files=[("srt", ("a.srt", SRT.encode())), ("samples", ("a.wav", b"x"))])
    assert r.status_code == 400
    assert client.post("/diarize/words/srt", headers=H, json={"words": []}).status_code == 400

    job_id = client.post("/diarize/words", headers=H, data=url, files={"srt": ("a.srt", SRT.encode())}).json()["id"]
    client.post("/worker/lease", headers=W, json={"worker": "w1", "langs": [], "engines": ["diarize"]})
    r = client.put(f"/worker/jobs/{job_id}/result", headers=W, params={"worker": "w1"}, json=words_doc())
    assert r.status_code == 422
    assert client.get(f"/jobs/{job_id}", headers=H).json()["status"] == "error"
