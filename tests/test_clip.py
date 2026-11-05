"""A clip of the media (ai_trns.clip): only its part is processed, the result's times stay the media's"""
import threading
import wave

import pytest

from ai_trns import audio, clip
from ai_trns.diarize.srt import parse_srt
from tests.conftest import H, words_doc


@pytest.mark.parametrize("value, seconds", [
    (197, 197.0), ("197.5", 197.5), ("3m17s", 197.0), ("1h2m3s", 3723.0), ("5m", 300.0), ("43s", 43.0),
    ("03:17", 197.0), ("1:02:03", 3723.0), (" 3M17S ", 197.0)])
def test_parse_time(value, seconds):
    assert clip.parse_time(value) == seconds


@pytest.mark.parametrize("value", ["", "abc", "3x", "-5", "1:2:3:4", "m", "nan", "inf"])
def test_parse_time_rejects(value):
    with pytest.raises(ValueError):
        clip.parse_time(value)


def test_make():
    assert clip.make(None, None) is None
    assert clip.make("", "") is None
    assert clip.make("0", None) is None
    assert clip.make("3m17s", "5m43s") == {"start": 197.0, "end": 343.0}
    assert clip.make(None, "10") == {"start": 0.0, "end": 10.0}
    assert clip.make("10", None) == {"start": 10.0, "end": None}
    with pytest.raises(ValueError):
        clip.make("5m43s", "3m17s")


def test_window():
    assert clip.window({"start": 10.0, "end": None}, 60.0) == (10.0, 60.0)
    assert clip.window({"start": 10.0, "end": 90.0}, 60.0) == (10.0, 60.0)
    assert clip.window({"start": 10.0, "end": 20.0}, 60.0) == (10.0, 20.0)
    with pytest.raises(ValueError):
        clip.window({"start": 70.0, "end": None}, 60.0)


def test_cues_to_clip_keeps_one_cue_per_cue():
    cues = [{"start": 0.0, "end": 9.0, "text": "before"}, {"start": 9.0, "end": 12.0, "text": "across"},
            {"start": 12.0, "end": 15.0, "text": "inside"}, {"start": 21.0, "end": 22.0, "text": "after"}]
    assert clip.cues_to_clip(cues, 10.0, 20.0) == [
        {"start": 0.0, "end": 0.0, "text": ""}, {"start": 0.0, "end": 2.0, "text": "across"},
        {"start": 2.0, "end": 5.0, "text": "inside"}, {"start": 0.0, "end": 0.0, "text": ""}]


def test_shift_documents():
    doc = clip.shift_words_doc(words_doc(2.0), 10.0)
    assert doc["duration_s"] == 12.0
    assert [(w["start_s"], w["end_s"]) for w in doc["words"]] == [(10.0, 10.4), (10.5, 10.9)]
    assert doc["segments"][0]["start_s"] == 10.0
    d = clip.shift_diarize_doc({"duration_s": 2.0, "words": [{"word": "a", "start": 0.5, "end": 1.0}]}, 10.0)
    assert d == {"duration_s": 12.0, "words": [{"word": "a", "start": 10.5, "end": 11.0}]}


def test_cut_srt_keeps_the_cues_whose_middle_is_inside():
    srt = ("1\n00:00:01,000 --> 00:00:03,000\nא\n\n2\n00:00:09,000 --> 00:00:12,000\nב\n\n"
           "3\n00:00:15,000 --> 00:00:21,000\nג\n\n4\n00:00:30,000 --> 00:00:31,000\nד\n")
    assert [c["text"] for c in parse_srt(clip.cut_srt(srt, 10.0, 20.0))] == ["ב", "ג"]
    assert [c["start"] for c in parse_srt(clip.cut_srt(srt, 10.0, None))] == [9.0, 15.0, 30.0]
    assert clip.cut_srt(srt, 40.0, 50.0).strip() == ""


def test_convert_to_wav_cuts_the_clip(tmp_path, audio_file):
    wav = tmp_path / "out.wav"
    assert audio.convert_to_wav(str(audio_file), str(wav), 2.0, start=1.0, length=2.0)
    with wave.open(str(wav)) as w:
        assert abs(w.getnframes() / w.getframerate() - 2.0) < 0.05


def test_a_local_job_transcribes_only_the_clip(tmp_path, audio_file, monkeypatch):
    """The worker cuts the clip's audio, the model gets only it, the document is in the media's time"""
    from ai_trns.worker import __main__ as w
    seen = {}

    class Transcriber:
        def transcribe(self, wav, **kw):
            with wave.open(wav) as f:
                seen["seconds"] = f.getnframes() / f.getframerate()
            return words_doc(2.0)

    class Models:
        last_used = 0

        def get(self, lang):
            return Transcriber()

    class Beat:
        cancelled = threading.Event()

        def update(self, stage, progress):
            pass

    job = {"id": "j1", "params": {"lang": "he", "clip": {"start": 1.0, "end": 3.0}}, "source": {"url": str(audio_file)}}
    doc = w.run_locally(None, Models(), job, str(tmp_path), Beat())
    assert abs(seen["seconds"] - 2.0) < 0.05
    assert doc["duration_s"] == 3.0
    assert doc["words"][0]["start_s"] == 1.0


def test_the_api_keeps_the_clip_in_the_params(client):
    r = client.post("/jobs", headers=H, data={"url": "https://cdn.example.org/a.mp3", "start": "3m17s", "end": "5m43s"})
    assert r.status_code == 202, r.text
    assert r.json()["params"]["clip"] == {"start": 197.0, "end": 343.0}
    assert "clip" not in client.post("/jobs", headers=H, data={"url": "https://cdn.example.org/a.mp3"}).json()["params"]
    r = client.post("/jobs", headers=H, data={"url": "https://cdn.example.org/a.mp3", "start": "5m", "end": "1m"})
    assert r.status_code == 422
    srt = b"1\n00:00:01,000 --> 00:00:02,000\nhello\n"
    r = client.post("/diarize/words", headers=H, data={"url": "https://cdn.example.org/a.mp3", "start": "10"},
                    files={"srt": ("a.srt", srt)})
    assert r.status_code == 202, r.text
    assert r.json()["params"]["clip"] == {"start": 10.0, "end": None}
