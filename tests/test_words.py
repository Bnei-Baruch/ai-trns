import json
from types import SimpleNamespace as NS

import pytest

from ai_trns.subtitles import render
from ai_trns.words import build_document


def seg(start, end, text, tokens=None):
    words = [NS(word=w, start=s, end=e, probability=0.9) for w, s, e in (tokens or [])]
    return NS(start=start, end=end, text=text, words=words)


def test_gershayim_tokens_are_merged_back_into_one_word():
    # faster-whisper words carry a leading space and split לחו"ל into two tokens
    doc = build_document([seg(0.0, 2.0, ' נסע לחו"ל', [(" נסע", 0.0, 0.5), (" לחו", 0.6, 1.0), ('"ל', 1.0, 1.3)])],
                         duration_s=3.0, language="he", model="m")
    assert [(w["word"], w["start_s"], w["end_s"]) for w in doc["words"]] == [("נסע", 0.0, 0.5), ('לחו"ל', 0.6, 1.3)]
    assert not any(w["interpolated"] for w in doc["words"])


def test_segment_without_usable_times_is_interpolated_by_word_length():
    doc = build_document([seg(1.0, 3.0, "אב אבגד", [(" משהו", 0.0, 1.0)])], duration_s=10.0, language="he", model="m")
    words = doc["words"]
    assert all(w["interpolated"] for w in words)
    assert words[0]["start_s"] == pytest.approx(1.0)
    assert words[0]["end_s"] == pytest.approx(1.0 + 2.0 * 2 / 6)
    assert words[1]["end_s"] == pytest.approx(3.0)


def test_document_text_is_exactly_the_words():
    doc = build_document([seg(0, 1, " שלום עולם.", [(" שלום", 0, 0.4), (" עולם.", 0.5, 0.9)]),
                          seg(1, 2, "  "),
                          seg(2, 3, " מה נשמע", [(" מה", 2.0, 2.3), (" נשמע", 2.4, 2.9)])],
                         duration_s=3.0, language="he", model="m", boundary_correction_s={"start": -0.1, "end": 0.2})
    assert doc["text"] == "שלום עולם. מה נשמע"
    assert doc["text"].split() == [w["word"] for w in doc["words"]]
    assert [w["segment"] for w in doc["words"]] == [0, 0, 1, 1]
    assert doc["timing"]["boundary_correction_s"] == {"start": -0.1, "end": 0.2}


@pytest.mark.parametrize("style", [
    "house",
    pytest.param("advanced", marks=pytest.mark.xfail(reason="ToSubs drops the last word (pre-existing bug)", strict=True)),
    "standard",
])
def test_every_srt_style_renders_from_the_document(style):
    doc = build_document([seg(0, 1, " שלום עולם.", [(" שלום", 0, 0.4), (" עולם.", 0.5, 0.9)]),
                          seg(2, 3, " מה נשמע", [(" מה", 2.0, 2.3), (" נשמע", 2.4, 2.9)])],
                         duration_s=3.0, language="he", model="m")
    srt = render(doc, "srt", style)
    assert "-->" in srt and "שלום" in srt and "נשמע" in srt
    assert render(doc, "txt") == "שלום עולם. מה נשמע"
    assert json.loads(render(doc, "json")) == doc
