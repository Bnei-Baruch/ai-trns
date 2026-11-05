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
        ("הרב", "שלום חברים.", 0.0, 1.5), ("הרב", "היום נלמד.", 2.0, 3.5), ("תלמיד", "שאלה?", 4.0, 4.5),
        ("הרב", "כן.", 5.0, 5.5), ("הרב", "עוד משפט.", 6.0, 7.5)]
    assert result["stats"] == {"paragraphs": 5, "turns": 3, "model_breaks": 2, "reader_turns": 0,
                               "reading_words": 0}
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
    # AI_TRNS_PARAGRAPHS_MIN_WORDS: a start too close to the one before it (or to its turn's start) is dropped
    monkeypatch.setattr(config, "PARAGRAPHS_MIN_WORDS", 2)
    doc = doc_of(LESSON)
    w.add_paragraphs(doc)
    assert doc["paragraph_starts"] == [2]  # 6 is one word after its turn's start (5)


def test_thin_starts_keeps_starts_min_words_apart_inside_a_turn():
    words = [{"word": "w", "speaker": s} for s in ["Rav"] * 10 + ["SPEAKER_01"] * 2 + ["Rav"] * 6]
    assert build.thin_starts(words, [2, 3, 6, 9, 13, 16], 0) == [2, 3, 6, 9, 13, 16]
    # turn 0..9: 2 (2 from 0) dropped, then 3, 6, 9 kept; turn 12..17: 13 (1 from 12) dropped, 16 kept
    assert build.thin_starts(words, [2, 3, 6, 9, 13, 16], 3) == [3, 6, 9, 16]


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


def test_evaluate_against_an_edited_transcript():
    from ai_trns.paragraphs import evaluate
    text = ('כותרת השיעור\n\nקריין: קוראים בספר.\n"אור גדול." מאיר.\n\nשלום חברים. היום נלמד.\nעוד משפט.\n'
            'שאלה: למה?\nכי כך.\n')
    human = evaluate.human_words(text)
    assert [(w["key"], w["label"]) for w in human][2:7] == [
        ("קוראים", "קריין"), ("בספר", "קריין"), ("אור", "קריין"), ("גדול", "קריין"), ("מאיר", "קריין")]
    words = lambda s, t0: [{"word": w, "start": t0 + i, "end": t0 + i + 0.5} for i, w in enumerate(s.split())]
    result = {"paragraphs": [
        {"label": "תלמיד", "words": words("קוראים בספר. אור גדול. מאיר.", 0)},
        {"label": "הרב", "words": words("שלום חברים. היום נלמד.", 10)},
        {"label": "הרב", "words": words("עוד משפט.", 20)},
        {"label": "תלמיד", "words": words("למה?", 30)},
        {"label": "הרב", "words": words("כי כך.", 40)},
    ]}
    r = evaluate.evaluate(text, result)
    assert r["speakers"]["הרב"] == {"words": 8, "right": 1.0}
    assert r["speakers"]["קריין"] == {"words": 5, "right": 0.0}
    assert r["speakers"]["תלמיד"] == {"words": 1, "right": 1.0}
    assert r["paragraphs_exact"] == {"p": 1.0, "r": 1.0, "f1": 1.0, "tp": 1, "pred": 1, "gold": 1}
    assert r["speaker_changes"] == {"missed": 0, "extra": 0, "both": 3}
    # the alignment depends on the words only: given once, the same scores
    pairs = evaluate.align(human, evaluate.model_words(result))
    assert evaluate.evaluate(text, result, pairs) == r


def test_the_unmarked_reading_of_the_source_is_the_readers():
    from ai_trns.paragraphs import evaluate
    source = ("ובזה תבין את דברי חז\"ל שאמרו שהעולם נברא בשביל התורה והישראל ולכן צריכים להבין את המטרה הזו "
              "ושכל הבריאה כולה אינה אלא כדי להביא את האדם לדבקות בבורא")
    reading = source.split()[:24]
    reading[11] = "ולכן,"           # punctuation is not a difference
    reading[16:17] = ["אה", "כן"]  # two words the source doesn't have, inside the passage
    text = ("כותרת השיעור\n"
            + " ".join(reading) + "\n"
            "מה זה אומר? כמו שכתוב דברי חז\"ל שאמרו שהעולם נברא בשביל התורה, זה העיקר.\n")
    human = evaluate.human_words(text, source)
    labels = [w["label"] for w in human]
    assert labels[:2] == ["הרב", "הרב"]                     # the title
    assert labels[2:2 + 25] == ["קריין"] * 25             # the reading, the inserted words too
    assert set(labels[2 + 25:]) == {"הרב"}                 # a 7-word quote inside the Rav's words stays his
    assert [w["label"] for w in evaluate.human_words(text)] == ["הרב"] * len(human)
    # a passage shorter than min_words is not a reading
    assert evaluate.source_passages(evaluate.human_words(text), source, min_words=26) == []
    # the evaluation scores the reading as the reader's
    words = lambda s, t0: [{"word": w, "start": t0 + i, "end": t0 + i + 0.5} for i, w in enumerate(s.split())]
    result = {"paragraphs": [{"label": "קריין", "words": words(" ".join(reading), 0)},
                             {"label": "הרב", "words": words(text.split("\n")[2], 30)}]}
    r = evaluate.evaluate(text, result, source=source)
    assert r["speakers"]["קריין"] == {"words": 25, "right": 1.0}
    assert r["speakers"]["הרב"]["right"] == 1.0


SOURCE = "ובזה תבין את דברי חז\"ל, שאמרו שהעולם נברא בשביל התורה והישראל."
READING = [(w, "SPEAKER_02") for w in "שוב ובזה תבין את דברי חז\"ל, שאמרו שהעולם נברא בשביל התורה".split()]


def test_a_turn_reading_the_source_is_the_readers():
    doc = doc_of(READING + RAV + [(w, "SPEAKER_02") for w in "ומה זה אומר על הדרך שלנו היום בבקשה".split()])
    result = build.paragraphs(doc, SOURCE)
    assert [p["label"] for p in result["paragraphs"]] == ["קריין", "הרב", "תלמיד"]
    assert result["paragraphs"][0]["speaker"] == "Reader" and result["stats"]["reader_turns"] == 1
    # without the source: a student
    assert [p["label"] for p in build.paragraphs(doc)["paragraphs"]] == ["תלמיד", "הרב", "תלמיד"]
    # the Rav reading the source stays the Rav
    rav_reads = doc_of([(w, "Rav") for w, _ in READING])
    assert [p["label"] for p in build.paragraphs(rav_reads, SOURCE)["paragraphs"]] == ["הרב"]


def test_the_source_of_a_diarize_job_from_txt_or_docx(client, audio_file, tmp_path):
    import io
    import docx
    from ai_trns.api import config
    from tests.conftest import H
    from tests.test_tools import SRT
    d = docx.Document()
    d.add_paragraph("כותרת")
    d.add_paragraph(SOURCE)
    buf = io.BytesIO()
    d.save(buf)
    for name, content in (("s.docx", buf.getvalue()), ("s.txt", SOURCE.encode())):
        with open(audio_file, "rb") as f:
            r = client.post("/diarize/words", headers=H, data={"paragraphs": "true"},
                            files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode())),
                                   ("source", (name, content))])
        assert r.status_code == 202, r.text
        saved = (tmp_path / "jobs" / r.json()["id"] / "source.txt").read_text(encoding="utf-8")
        assert SOURCE in saved
    with open(audio_file, "rb") as f:
        r = client.post("/diarize/words", headers=H, files=[("audio", ("in.mp3", f)), ("srt", ("a.srt", SRT.encode())),
                                                            ("source", ("s.docx", b"not a docx"))])
    assert r.status_code == 400
    assert config.JOBS_DIR.startswith(str(tmp_path))


LONG_SOURCE = ("ובזה תבין את דברי חז\"ל שאמרו שהעולם נברא בשביל התורה והישראל ולכן צריכים להבין את המטרה הזו "
               "ושכל הבריאה כולה אינה אלא כדי להביא את האדם לדבקות בבורא")


def test_reading_is_told_apart_from_speaking_whoever_speaks():
    read = LONG_SOURCE.split()[:20]
    read[9:10] = ["אה", "בשביל"]  # an ASR slip inside the reading
    announce = [(w, "SPEAKER_03") for w in "קטע מספר 2 נקרא עכשיו".split()]
    reads = [(w, "SPEAKER_03") for w in read]
    talks = [(w, "SPEAKER_03") for w in "ועכשיו שאלה מחבר מאיטליה בבקשה".split()]
    doc = doc_of(announce + reads + talks + RAV + [(w, "Rav") for w in LONG_SOURCE.split()[4:20]] + RAV)
    result = build.paragraphs(doc, LONG_SOURCE)
    assert [(p["label"], p["reading"], len(p["words"])) for p in result["paragraphs"]] == [
        ("קריין", False, 5), ("קריין", True, 21), ("קריין", False, 5),  # the reader announces, reads, talks
        ("הרב", False, 4), ("הרב", True, 16), ("הרב", False, 4)]          # the Rav reads too: still the Rav's
    assert result["stats"]["reading_words"] == 37 and result["stats"]["model_breaks"] == 0
    # a quote shorter than READING_MIN_WORDS is speech
    assert not any(build.reading_words(doc_of([(w, "Rav") for w in LONG_SOURCE.split()[:11]])["words"], LONG_SOURCE))
    # without the source, nothing is reading
    assert not any(p["reading"] for p in build.paragraphs(doc)["paragraphs"])


def test_the_evaluation_scores_reading_apart_from_speakers():
    from ai_trns.paragraphs import evaluate
    text = "תלמיד: כתוב " + LONG_SOURCE + "\nזה העיקר שם.\n"
    words = lambda s, t0: [{"word": w, "start": t0 + i, "end": t0 + i + 0.5} for i, w in enumerate(s.split())]
    result = {"paragraphs": [{"label": "תלמיד", "reading": False, "words": words("כתוב", 0)},
                             {"label": "תלמיד", "reading": True, "words": words(LONG_SOURCE, 1)},
                             {"label": "הרב", "reading": False, "words": words("זה העיקר שם.", 40)}]}
    r = evaluate.evaluate(text, result, source=LONG_SOURCE)
    n = len(LONG_SOURCE.split())
    assert r["speakers"]["תלמיד"] == {"words": n + 1, "right": 1.0}  # a student quoting the source: a student
    assert r["reading"] == {"p": 1.0, "r": 1.0, "f1": 1.0, "tp": n, "pred": n, "gold": n}
    assert evaluate.evaluate(text, result)["reading"] is None


def test_the_reader_is_the_voice_that_announces_the_reading():
    announce = [(w, "SPEAKER_03") for w in "קטע מספר 2 מתוך מאמר שמעתי נקרא עכשיו".split()]
    more = [(w, "SPEAKER_03") for w in "והאדם צריך להאמין למעלה מהדעת תמיד בכל מצב".split()]
    relayed = [(w, "SPEAKER_03") for w in "שאלה מחבר מאיטליה מה זה אמונה למעלה מהדעת".split()]
    asks = [(w, "SPEAKER_04") for w in "ומה זה אומר על הדרך שלנו היום בבקשה".split()]
    doc = doc_of(announce + RAV + more + RAV + relayed + RAV + asks)
    assert [p["label"] for p in build.paragraphs(doc)["paragraphs"]] == [
        "קריין", "הרב", "קריין", "הרב", "תלמיד", "הרב", "תלמיד"]
