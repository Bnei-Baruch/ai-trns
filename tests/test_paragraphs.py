"""Paragraphs of a lesson (ai_trns.paragraphs): speaker turns, the Rav's split by a fake paragraphing service"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ai_trns.paragraphs import build, client


def doc_of(words):
    return {"version": 1, "kind": "diarize", "duration_s": 60.0, "language": "he", "model": "fake",
            "words": [{"word": w, "start": float(i), "end": i + 0.5, "speaker": s, "cue": i}
                      for i, (w, s) in enumerate(words)],
            "stats": {"speakers": 2, "named": 1}}


RAV = [("שלום", "Rav"), ("חברים.", "Rav"), ("היום", "Rav"), ("נלמד.", "Rav")]
STUDENT = [("שאלה?", "SPEAKER_01")]
LESSON = RAV + STUDENT + [("כן.", "Rav"), ("עוד", None), ("משפט.", "Rav")]


def test_speaker_turns_join_words_without_a_speaker_to_the_turn_before():
    words = doc_of(LESSON)["words"]
    assert build.speaker_turns(words) == [range(0, 4), range(4, 5), range(5, 8)]
    assert build.speaker_turns([]) == []


def test_model_text_is_the_rav_turns_with_word_offsets():
    words = doc_of(LESSON)["words"]
    text, index, offsets = build.model_text(words, build.speaker_turns(words))
    assert text == "שלום חברים. היום נלמד. כן. עוד משפט."
    assert index == [0, 1, 2, 3, 5, 6, 7]
    assert [text[o:].split()[0] for o in offsets] == [w["word"] for w in (words[i] for i in index)]


def test_sentence_breaks_become_word_starts():
    words = doc_of(LESSON)["words"]
    text, index, offsets = build.model_text(words, build.speaker_turns(words))
    # the service's sentences of that text: "שלום חברים." "היום נלמד." "כן." "עוד משפט."
    sentences = [[0, 11], [12, 22], [23, 26], [27, 36]]
    assert build.starts_from_sentences(index, offsets, sentences, [1, 3, 0, 99]) == [2, 6]


def test_paragraphs_split_at_speaker_changes_and_model_starts():
    doc = doc_of(LESSON)
    assert [p["text"] for p in build.paragraphs(doc)["paragraphs"]] == ["שלום חברים. היום נלמד.", "שאלה?",
                                                                          "כן. עוד משפט."]
    doc["paragraph_starts"] = [2, 6]
    result = build.paragraphs(doc)
    assert [(p["label"], p["text"], p["start"], p["end"]) for p in result["paragraphs"]] == [
        ("רב", "שלום חברים.", 0.0, 1.5), ("רב", "היום נלמד.", 2.0, 3.5), ("תלמיד", "שאלה?", 4.0, 4.5),
        ("רב", "כן.", 5.0, 5.5), ("רב", "עוד משפט.", 6.0, 7.5)]
    assert result["stats"] == {"paragraphs": 5, "turns": 3, "model_breaks": 2}
    assert result["paragraphs"][1]["words"] == [{"word": "היום", "start": 2.0, "end": 2.5},
                                                {"word": "נלמד.", "start": 3.0, "end": 3.5}]


def test_the_reader_label():
    doc = doc_of([("קוראים", "Reader"), ("שאלה", "SPEAKER_00")])
    assert [p["label"] for p in build.paragraphs(doc)["paragraphs"]] == ["קריין", "תלמיד"]


class FakeService:
    """The paragraphing service: sentences end at '.', every second sentence starts a paragraph"""

    def __init__(self, key="k", status=200):
        self.texts = []
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                service.texts.append(body["text"])
                if self.headers.get(client.KEY_HEADER) != key or status != 200:
                    code, reply = (status if status != 200 else 401), {"detail": "no"}
                else:
                    text, sentences, start = body["text"], [], 0
                    for i, ch in enumerate(text):
                        if ch == "." and (i + 1 == len(text) or text[i + 1] == " "):
                            sentences.append([start, i + 1])
                            start = i + 2
                    code, reply = 200, {"sentences": sentences, "breaks": list(range(1, len(sentences), 2)),
                                        "chunks": 1, "seconds": 0.1}
                data = json.dumps(reply).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def service():
    s = FakeService()
    yield s
    s.close()


def test_the_worker_adds_the_services_paragraph_starts(service, monkeypatch):
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    monkeypatch.setattr(config, "PARAGRAPHS_URL", service.url)
    monkeypatch.setattr(config, "PARAGRAPHS_KEY", "k")
    doc = doc_of(LESSON)
    w.add_paragraphs(doc)
    assert service.texts == ["שלום חברים. היום נלמד. כן. עוד משפט."]
    # sentences 1 and 3 start paragraphs: "היום" (word 2) and "עוד" (word 6)
    assert doc["paragraph_starts"] == [2, 6]
    assert "paragraphs_error" not in doc["stats"]


def test_a_failing_service_leaves_speaker_turns(monkeypatch):
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    s = FakeService(status=500)
    try:
        monkeypatch.setattr(config, "PARAGRAPHS_URL", s.url)
        doc = doc_of(LESSON)
        w.add_paragraphs(doc)
    finally:
        s.close()
    assert "paragraph_starts" not in doc and "500" in doc["stats"]["paragraphs_error"]
    result = build.paragraphs(doc)
    assert result["stats"]["turns"] == 3 and "500" in result["stats"]["paragraphs_error"]
    monkeypatch.setattr(config, "PARAGRAPHS_URL", "")
    doc = doc_of(LESSON)
    w.add_paragraphs(doc)
    assert "AI_TRNS_PARAGRAPHS_URL" in doc["stats"]["paragraphs_error"]


def test_a_diarize_job_with_paragraphs_end_to_end(server, worker, audio_file, service, monkeypatch):  # noqa: F811
    import httpx
    from ai_trns.worker import __main__ as w
    from ai_trns.worker import config
    from tests.conftest import H
    from tests.test_tools import SRT
    from tests.test_worker import wait_status
    monkeypatch.setattr(config, "ENGINES", ["diarize"])
    monkeypatch.setattr(config, "PARAGRAPHS_URL", service.url)
    monkeypatch.setattr(config, "PARAGRAPHS_KEY", "k")

    class FakeDiarizer:
        def diarize(self, wav_file, cues, samples, lang, speakers, on_progress, should_cancel):
            return doc_of(LESSON)

    monkeypatch.setattr(w.Models, "diarizer", lambda self: FakeDiarizer())
    stop = threading.Event()
    monkeypatch.setattr(config, "WORKER_ID", "paragraphs-worker")
    thread = threading.Thread(target=w.main, args=(stop,), daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=server) as http, open(audio_file, "rb") as f:
            job_id = http.post("/diarize/words", headers=H, data={"paragraphs": "true"},
                               files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode()))]).json()["id"]
            job = wait_status(http, job_id, ("done", "error"))
            assert job["status"] == "done" and job["params"]["paragraphs"] is True, job
            r = http.get(f"/jobs/{job_id}/result", headers=H, params={"format": "paragraphs"})
            assert r.status_code == 200 and r.headers["content-type"] == "application/json"
            assert [p["text"] for p in r.json()["paragraphs"]] == ["שלום חברים.", "היום נלמד.", "שאלה?", "כן.",
                                                                     "עוד משפט."]
    finally:
        stop.set()
        thread.join(timeout=10)


from tests.test_worker import server, worker  # noqa: E402,F401 (fixtures)
