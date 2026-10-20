"""Cue aggregation tests, ported with the code from whisper_trainer tests/test_timing_alignment.py"""
import pytest

from ai_trns.cues import (
    HOUSE_POLICY,
    MUSIC_CUE,
    CueAggregationPolicy,
    TimedCue,
    WordTiming,
    aggregate_words_into_cues,
    canonicalize_cues,
    render_srt,
    serialize_srt,
    wrap_cue_lines,
)


def test_gap_and_length_split_cues():
    policy = CueAggregationPolicy(max_chars=10, gap_split_s=1.0)
    words = [
        WordTiming("aaa", 0.0, 0.4),
        WordTiming("bbb", 0.5, 0.9),
        WordTiming("ccc", 2.0, 2.4),
        WordTiming("dddddddddddddddddddd", 2.5, 2.9),
    ]
    cues = aggregate_words_into_cues(words, policy)
    assert [cue.text for cue in cues] == ["aaa bbb", "ccc", "dddddddddddddddddddd"]
    assert cues[0].start_s == pytest.approx(0.0)
    assert cues[0].end_s == pytest.approx(0.9)


def test_duration_cap_splits_long_runs():
    policy = CueAggregationPolicy(max_chars=100, max_duration_s=1.0, gap_split_s=5.0)
    words = [WordTiming(f"w{i}", i * 0.4, i * 0.4 + 0.3) for i in range(6)]
    cues = aggregate_words_into_cues(words, policy)
    assert len(cues) > 1
    assert all(cue.end_s - cue.start_s <= 1.35 for cue in cues)


def test_offsets_shift_boundaries():
    policy = CueAggregationPolicy(start_offset_s=-0.1, end_offset_s=0.2)
    cues = aggregate_words_into_cues([WordTiming("x", 1.0, 2.0)], policy)
    assert cues[0].start_s == pytest.approx(0.9)
    assert cues[0].end_s == pytest.approx(2.2)


def test_sentence_punctuation_ends_a_cue():
    words = [
        WordTiming("שלום", 0.0, 0.4),
        WordTiming("עולם.", 0.5, 0.9),
        WordTiming("מה", 1.0, 1.4),
        WordTiming("נשמע", 1.5, 1.9),
    ]
    policy = CueAggregationPolicy(gap_split_s=5.0, split_after_chars=".,?!", min_split_chars=0)
    assert [cue.text for cue in aggregate_words_into_cues(words, policy)] == ["שלום עולם.", "מה נשמע"]
    # A minimum length keeps punctuation from cutting a cue to one word.
    guarded = CueAggregationPolicy(gap_split_s=5.0, split_after_chars=".,?!", min_split_chars=20)
    assert len(aggregate_words_into_cues(words, guarded)) == 1


def test_rejects_invalid_policy():
    for policy in (CueAggregationPolicy(max_chars=0), CueAggregationPolicy(min_split_chars=-1),
                   CueAggregationPolicy(line_chars=0), CueAggregationPolicy(fit_lines=True),
                   CueAggregationPolicy(pause_split_min_chars=-1)):
        with pytest.raises(ValueError):
            aggregate_words_into_cues([], policy)


def test_wrap_leaves_a_cue_within_the_line_budget_on_one_line():
    assert wrap_cue_lines("שלום עולם", 32) == "שלום עולם"
    assert wrap_cue_lines("שלום עולם ומלואו וכל צבאו והרבה מעבר", None).count("\n") == 0
    assert wrap_cue_lines("א" * 40, 32) == "א" * 40


def test_wrap_balances_two_lines_and_restores_the_text_when_joined():
    text = "אז אתה מתחיל לראות, שההמן הזה מתחיל להיות"
    wrapped = wrap_cue_lines(text, 32)
    first, second = wrapped.split("\n")
    assert wrapped.count("\n") == 1
    assert abs(len(first) - len(second)) <= max(len(word) for word in text.split())
    assert f"{first} {second}" == text


def test_wrap_breaks_a_tie_toward_the_shorter_first_line():
    assert wrap_cue_lines("aa bb cc", 5) == "aa\nbb cc"


def _paced(words, pause_after):
    timings, clock = [], 0.0
    for index, word in enumerate(words):
        timings.append(WordTiming(word, clock, clock + 0.3))
        clock += 0.35 + pause_after.get(index, 0.0)
    return timings


def test_fitting_two_lines_caps_a_cue_below_max_chars():
    words = _paced(["מילה"] * 16, {})
    loose = CueAggregationPolicy(gap_split_s=5.0, max_duration_s=60.0, line_chars=10)
    fitted = CueAggregationPolicy(gap_split_s=5.0, max_duration_s=60.0, line_chars=10, fit_lines=True)
    assert len(aggregate_words_into_cues(words, loose)) == 1
    cues = aggregate_words_into_cues(words, fitted)
    assert [cue.text for cue in cues] == ["מילה מילה מילה מילה"] * 4
    assert all(len(line) <= 10 for cue in cues for line in wrap_cue_lines(cue.text, 10).split("\n"))


def test_a_capacity_split_moves_to_the_longest_pause():
    words = _paced(["אחת", "שתיים", "שלוש", "ארבע", "חמש", "שש"], {1: 0.5})
    common = {"gap_split_s": 5.0, "max_duration_s": 60.0, "max_chars": 24}
    greedy = aggregate_words_into_cues(words, CueAggregationPolicy(**common))
    paused = aggregate_words_into_cues(words, CueAggregationPolicy(**common, pause_split_min_chars=5))
    guarded = aggregate_words_into_cues(words, CueAggregationPolicy(**common, pause_split_min_chars=10))
    assert [cue.text for cue in greedy] == ["אחת שתיים שלוש ארבע חמש", "שש"]
    assert [cue.text for cue in paused] == ["אחת שתיים", "שלוש ארבע חמש שש"]
    assert paused[0].end_s == pytest.approx(words[1].end_s)
    assert [cue.text for cue in guarded] == [cue.text for cue in greedy]


def test_music_is_a_cue_of_its_own():
    words = [WordTiming("לפני", 0.0, 0.4), WordTiming(MUSIC_CUE, 0.5, 9.0), WordTiming("אחרי", 9.1, 9.5)]
    policy = CueAggregationPolicy(gap_split_s=5.0, max_duration_s=60.0)
    assert [c.text for c in aggregate_words_into_cues(words, policy)] == ["לפני", MUSIC_CUE, "אחרי"]


def test_canonicalize_clamps_and_removes_overlap():
    cues, repairs = canonicalize_cues(
        [TimedCue("a", -0.1, 1.0), TimedCue("b", 0.8, 9.0), TimedCue("c", 9.5, 12.0)],
        duration_s=10.0, max_duration_s=7.0, max_chars=84)
    assert [(c["start_ms"], c["end_ms"]) for c in cues] == [(0, 1000), (1000, 8000), (9500, 10000)]
    assert repairs["clamped"] == 2 and repairs["overlap"] == 1 and repairs["duration"] == 1


def test_serialize_srt_wraps_lines():
    text = "הם מרגישים צימאון, אנשים באים, חסר להם, וזה לא נגמר"
    srt = serialize_srt([{"text": text, "start_ms": 0, "end_ms": 2000}], line_chars=32)
    index, timing, *lines = srt.strip().split("\n")
    assert (index, timing) == ("1", "00:00:00,000 --> 00:00:02,000")
    assert len(lines) == 2 and " ".join(lines) == text


def test_render_srt_applies_the_documents_boundary_correction():
    doc = {"duration_s": 5.0, "timing": {"boundary_correction_s": {"start": -0.1, "end": 0.22}},
           "words": [{"word": "שלום", "start_s": 1.0, "end_s": 1.4}, {"word": "עולם", "start_s": 1.5, "end_s": 2.0}]}
    assert render_srt(doc) == "1\n00:00:00,900 --> 00:00:02,220\nשלום עולם\n"
    assert HOUSE_POLICY["line_chars"] == 32
