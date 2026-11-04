import subprocess
from types import SimpleNamespace as NS

import numpy as np
import pytest

from ai_trns.live.engine import run
from ai_trns.live.source import AudioSource, is_live
from ai_trns.live.stream import SAMPLE_RATE, StreamProcessor, Word, build_live_document, merge_tokens
from ai_trns.subtitles import render

# a word every 0.5 s, a 3 s pause after the 20th
SCRIPT = [Word(i * 0.5 + (3.0 if i >= 20 else 0.0), i * 0.5 + 0.4 + (3.0 if i >= 20 else 0.0), f"w{i}")
          for i in range(60)]


class FakeAsr:
    """Hears the script words that fit in the buffer; the last one is misheard differently every call"""

    def __init__(self, words=SCRIPT):
        self.words = words
        self.processor = None
        self.calls = 0
        self.max_buffer_s = 0.0

    def __call__(self, audio, prompt):
        self.calls += 1
        offset, end = self.processor.offset_s, self.processor.end_s
        self.max_buffer_s = max(self.max_buffer_s, len(audio) / SAMPLE_RATE)
        heard = [Word(w.start_s - offset, w.end_s - offset, w.word) for w in self.words
                 if w.start_s >= offset and w.end_s <= end]
        if heard:
            heard[-1] = Word(heard[-1].start_s, heard[-1].end_s, f"{heard[-1].word}~{self.calls}")
        return heard


def stream(asr, seconds, chunk_s=1.0, **kwargs):
    processor = StreamProcessor(asr, **kwargs)
    asr.processor = processor
    events = []
    for _ in range(int(seconds / chunk_s)):
        processor.insert_audio(np.zeros(int(chunk_s * SAMPLE_RATE), np.float32))
        final, partial = processor.process()
        events.append((final, partial))
    events.append((processor.finish(), []))
    return processor, events


def test_every_word_becomes_final_once_in_order():
    asr = FakeAsr()
    processor, events = stream(asr, 35.0)
    assert [w.word for w in processor.final[:-1]] == [w.word for w in SCRIPT[:-1]]
    # the last word was never heard the same way twice: committed as heard at the end
    assert processor.final[-1].word.startswith("w59~")
    assert [w.start_s for w in processor.final] == pytest.approx([w.start_s for w in SCRIPT])
    assert [w for final, _ in events for w in final] == processor.final


def test_unstable_word_stays_partial():
    asr = FakeAsr()
    _, events = stream(asr, 5.0)
    final, partial = events[3]
    assert partial and partial[-1].word.endswith("~4")
    assert not any("~" in w.word for f, _ in events[:-1] for w in f)


def test_buffer_is_trimmed_after_final_words():
    asr = FakeAsr()
    stream(asr, 35.0, trim_s=5.0)
    assert asr.max_buffer_s <= 5.0 + 1.0 + 1e-6


def test_silence_does_not_grow_the_buffer():
    asr = FakeAsr(words=[])
    processor, _ = stream(asr, 60.0, trim_s=10.0)
    assert asr.max_buffer_s <= 11.0 + 1e-6
    assert processor.final == []


def test_endless_disagreement_is_forced_out():
    class Babble(FakeAsr):
        def __call__(self, audio, prompt):
            self.calls += 1
            self.max_buffer_s = max(self.max_buffer_s, len(audio) / SAMPLE_RATE)
            return [Word(0.0, 0.5, f"x{self.calls}")]

    asr = Babble()
    processor, _ = stream(asr, 40.0, max_buffer_s=10.0)
    assert processor.final and asr.max_buffer_s <= 12.0


def test_repeated_last_words_are_not_committed_twice():
    asr = FakeAsr()
    processor = StreamProcessor(asr)
    asr.processor = processor
    h = processor.hypothesis
    h.insert([Word(0.0, 0.4, "a"), Word(0.5, 0.9, "b")], 0.0)
    h.flush()
    h.insert([Word(0.0, 0.4, "a"), Word(0.5, 0.9, "b"), Word(1.0, 1.4, "c")], 0.0)
    assert [w.word for w in h.flush()] == ["a", "b"]
    # decode starting just before the last final word repeats it
    h.insert([Word(0.85, 0.9, "b."), Word(1.0, 1.4, "c"), Word(1.5, 1.9, "d")], 0.0)
    assert [w.word for w in h.flush()] == ["c"]


def test_gershayim_tokens_are_merged():
    tokens = [NS(word=" נסע", start=0.0, end=0.5, probability=0.9), NS(word=" לחו", start=0.6, end=1.0, probability=0.8),
              NS(word='"ל', start=1.0, end=1.3, probability=0.7)]
    assert merge_tokens(tokens) == [Word(0.0, 0.5, "נסע", 0.9), Word(0.6, 1.3, 'לחו"ל', 0.8)]


def test_final_words_make_a_words_document():
    words = [Word(0.0, 0.4, "שלום"), Word(0.5, 0.9, "עולם."), Word(1.0, 1.4, "מה"), Word(4.0, 4.4, "נשמע")]
    doc = build_live_document(words, 5.0, "he", "m")
    assert doc["timing"]["source"] == "live"
    assert [s["text"] for s in doc["segments"]] == ["שלום עולם.", "מה", "נשמע"]
    assert doc["text"].split() == [w["word"] for w in doc["words"]]
    assert not any(w["interpolated"] for w in doc["words"])
    assert render(doc, "srt").startswith("1\n00:00:00,000 --> ")


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "in.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-y", str(path)], check=True)
    return str(path)


def test_source_reads_the_whole_stream(audio_file):
    total = 0
    with AudioSource(audio_file) as source:
        while (samples := source.read(wait_s=0.5)) is not None:
            total += len(samples)
    assert total == pytest.approx(3 * SAMPLE_RATE, abs=SAMPLE_RATE * 0.1)


def test_source_reports_ffmpeg_failure(tmp_path):
    with AudioSource(str(tmp_path / "missing.mp3")) as source:
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            while source.read(wait_s=0.5) is not None:
                pass


def test_run_streams_a_file_to_events(audio_file):
    words = [Word(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(6)]
    asr = FakeAsr(words)
    processor = StreamProcessor(asr)
    asr.processor = processor
    events = []
    with AudioSource(audio_file) as source:
        duration = run(source, processor, events.append, min_chunk_s=0.5)
    assert duration == pytest.approx(3.0, abs=0.1)
    finals = [w["word"] for e in events if e["type"] == "final" for w in e["words"]]
    assert finals[:-1] == [f"w{i}" for i in range(5)] and finals[-1].startswith("w5")
    assert any(e["type"] == "partial" for e in events)


def test_demo_page_streams_events_and_serves_the_result(audio_file):
    import json
    import threading
    import urllib.request

    from ai_trns.live.demo import Session, serve
    from ai_trns.live.stream import build_live_document

    def new_processor():
        asr = FakeAsr([Word(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(6)])
        asr.processor = StreamProcessor(asr)
        return asr.processor

    session = Session(audio_file, new_processor, lambda p, d: build_live_document(p.final, d, "he", "fake"),
                      min_chunk_s=0.5)
    assert session.local_file
    server = serve(session, "127.0.0.1", 0, "he")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert json.load(urllib.request.urlopen(base + "/config")) == {"player": "audio", "lang": "he", "state": "idle"}
        assert "<audio" in urllib.request.urlopen(base + "/").read().decode()
        ranged = urllib.request.urlopen(urllib.request.Request(base + "/audio", headers={"Range": "bytes=0-99"}))
        assert ranged.status == 206 and len(ranged.read()) == 100
        start = urllib.request.Request(base + "/start", method="POST")
        assert json.load(urllib.request.urlopen(start))["started"] is True
        assert json.load(urllib.request.urlopen(start))["started"] is False
        events = [json.loads(line[6:]) for line in urllib.request.urlopen(base + "/events", timeout=10).read()
                  .decode().splitlines() if line.startswith("data: ")]
        assert events[-1]["type"] == "end"
        finals = [w["word"] for e in events if e["type"] == "final" for w in e["words"]]
        assert finals[:5] == [f"w{i}" for i in range(5)]
        srt = urllib.request.urlopen(base + "/result.srt").read().decode()
        assert srt.startswith("1\n") and "w0" in srt
        # a client joining after the end gets the final words, not every partial
        replay = [json.loads(line[6:]) for line in urllib.request.urlopen(base + "/events", timeout=10).read()
                  .decode().splitlines() if line.startswith("data: ")]
        assert [e["type"] for e in replay if e["type"] != "final"] == ["start", "end"]
    finally:
        server.shutdown()


def test_streams_are_told_from_recordings():
    assert is_live("https://edge3.uk.kab.tv/live/heb-medium/playlist.m3u8")
    assert is_live("https://example.org/live/manifest.mpd?token=x")
    assert is_live("/tmp/janus.sdp")
    assert not is_live("https://cdn.kabbalahmedia.info/K8X56H7k")
    assert not is_live("lesson.mp3")


def test_subdomain_entries_allow_stream_hosts():
    from ai_trns.audio import check_url
    allowed = ["cdn.kabbalahmedia.info", ".kab.tv"]
    check_url("https://edge3.uk.kab.tv/live/heb-medium/playlist.m3u8", allowed)
    check_url("https://cdn.kabbalahmedia.info/K8X56H7k", allowed)
    for bad in ("https://kab.tv.evil.org/x.m3u8", "https://evilkab.tv/x.m3u8", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            check_url(bad, allowed)


def test_whisper_asr_splits_words_by_the_text_not_the_tokens():
    from ai_trns.live.stream import WhisperAsr

    class Model:
        # as faster-whisper gives them for the Hebrew model: a segment's first word glued to the
        # previous segment's last token
        def transcribe(self, audio, **kwargs):
            return [NS(text="דרך איך שהבורא", words=[NS(word="דרך", start=0.0, end=0.3, probability=0.9),
                                                     NS(word=" איך", start=0.4, end=0.6, probability=0.9),
                                                     NS(word=" שהבוראעיצב", start=0.7, end=1.7, probability=0.8)]),
                    NS(text='עיצב לחו"ל.', words=[NS(word=" לחו", start=1.8, end=2.0, probability=0.9),
                                                 NS(word='"ל.', start=2.0, end=2.2, probability=0.9)])], None

    asr = WhisperAsr(NS(lang="he", model=Model()))
    words = asr(np.zeros(16000, np.float32), "")
    assert [w.word for w in words] == ["דרך", "איך", "שהבורא", "עיצב", 'לחו"ל.']
    # "שהבוראעיצב" is 6 + 4 letters over 1 s
    assert [(round(w.start_s, 2), round(w.end_s, 2)) for w in words[2:4]] == [(0.7, 1.3), (1.3, 1.7)]
    assert (words[4].start_s, words[4].end_s) == (1.8, 2.2)


def test_tokens_without_space_are_separate_words_unless_a_quote_or_punctuation():
    tokens = [NS(word=" אנחנו", start=0.0, end=0.3, probability=0.9), NS(word="צריכים", start=0.4, end=0.8, probability=0.9),
              NS(word=",", start=0.8, end=0.8, probability=0.9), NS(word=" לחו", start=1.0, end=1.2, probability=0.9),
              NS(word='"ל', start=1.2, end=1.4, probability=0.9)]
    assert [w.word for w in merge_tokens(tokens)] == ["אנחנו", "צריכים,", 'לחו"ל']
