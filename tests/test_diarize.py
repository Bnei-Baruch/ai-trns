"""Model-free parts of diarize (moved from stt-api): SRT, cue mapping, resegmenting, VAD spreading"""
import numpy as np
import pytest

from ai_trns.diarize import labels, mapping
from ai_trns.diarize.engine import apportion, builtin_samples, distribute_over_regions, spread_evenly
from ai_trns.diarize.resegment import build_cues, pad_cues
from ai_trns.diarize.srt import parse_srt, write_srt

SAMPLE = (
    "﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nשלום עולם\r\nשורה שנייה\r\n\r\n"
    "2\r\n00:00:03.2 --> 00:01:04,050\r\nמה שלומכם\r\n"
)


def test_parse():
    assert parse_srt(SAMPLE) == [
        {"start": 1.0, "end": 2.5, "text": "שלום עולם\nשורה שנייה"},
        {"start": 3.2, "end": 64.05, "text": "מה שלומכם"},
    ]


def test_missing_indices():
    data = "00:00:00,000 --> 00:00:01,000\na\n\n00:00:01,000 --> 00:00:02,000\nb\n"
    assert [c["text"] for c in parse_srt(data)] == ["a", "b"]


def test_numeric_text_in_last_cue_kept():
    data = "1\n00:00:00,000 --> 00:00:01,000\n42\n"
    assert parse_srt(data)[0]["text"] == "42"


def test_round_trip():
    cues = parse_srt(SAMPLE)
    assert parse_srt(write_srt(cues)) == cues


def test_write_format():
    assert write_srt([{"start": 3661.5, "end": 3662.0, "text": "x"}]) == "1\n01:01:01,500 --> 01:01:02,000\nx\n"


def test_garbage_rejected():
    with pytest.raises(ValueError):
        parse_srt("hello world")


def test_empty_rejected():
    with pytest.raises(ValueError, match="No SRT cues"):
        parse_srt("")


def test_start_after_end():
    with pytest.raises(ValueError, match="start after end"):
        parse_srt("1\n00:00:02,000 --> 00:00:01,000\na\n")


TURNS = [(0.0, 5.0, "A"), (5.0, 10.0, "B")]


def _span_words(spans):
    return [{"word": f"w{i}", "start": s, "end": e} for i, (s, e) in enumerate(spans)]


def test_dominant_single_speaker():
    assert mapping.dominant_cluster(1.0, 4.0, TURNS, 0.85) == ("A", False)


def test_dominant_mixed():
    assert mapping.dominant_cluster(3.0, 8.0, TURNS, 0.85) == ("B", True)


def test_dominant_no_overlap_uses_nearest():
    assert mapping.dominant_cluster(11.0, 12.0, TURNS, 0.85) == ("B", False)


def test_dominant_no_turns():
    assert mapping.dominant_cluster(0.0, 1.0, [], 0.85) == (None, False)


def test_split_at_speaker_change():
    cue = {"start": 3.0, "end": 8.0, "text": "w0 w1 w2 w3"}
    words = _span_words([(3.0, 3.8), (3.9, 4.8), (5.2, 6.0), (6.1, 7.9)])
    parts = mapping.split_cue(cue, words, TURNS, min_piece_words=2, min_piece_s=0.6)
    assert parts == [
        {"start": 3.0, "end": 4.8, "text": "w0 w1", "cluster": "A"},
        {"start": 5.2, "end": 8.0, "text": "w2 w3", "cluster": "B"},
    ]


def test_split_short_fragment_merged():
    cue = {"start": 4.0, "end": 9.0, "text": "w0 w1 w2 w3"}
    # one short word in A, then three words in B -> one cue for B
    words = _span_words([(4.5, 4.9), (5.1, 6.0), (6.1, 7.0), (7.1, 8.5)])
    parts = mapping.split_cue(cue, words, TURNS, min_piece_words=2, min_piece_s=0.6)
    assert parts == [{"start": 4.0, "end": 9.0, "text": "w0 w1 w2 w3", "cluster": "B"}]


def test_word_runs_per_speaker():
    words = _span_words([(3.0, 3.8), (3.9, 4.8), (5.2, 6.0), (6.1, 7.9)])
    assert mapping.word_runs(words, TURNS, 2, 0.6) == [([0, 1], "A"), ([2, 3], "B")]


def test_smooth_rejoins_same_cluster():
    words = _span_words([(i, i + 0.9) for i in range(7)])
    clusters = ["A", "A", "A", "B", "A", "A", "A"]
    assert mapping.smooth_runs(words, clusters, 2, 0.6) == [list(range(7))]


def test_split_without_words_keeps_cue():
    cue = {"start": 3.0, "end": 8.0, "text": "x"}
    assert mapping.split_cue(cue, [], TURNS, 2, 0.6) == [{**cue, "cluster": "B"}]


def test_match_samples_one_to_one_with_threshold():
    sim = np.array([[0.8, 0.7, 0.1], [0.75, 0.2, 0.3]])
    # optimal assignment: sample0->cluster1 (0.7) + sample1->cluster0 (0.75)
    assert mapping.match_samples(sim, 0.5) == {1: 0, 0: 1}
    assert mapping.match_samples(sim, 0.72) == {0: 1}


def test_match_samples_extra_clusters_off_by_default():
    # cluster 2 is the Rav's voice too, but one-to-one gives it nobody
    sim = np.array([[0.8, 0.1, 0.6], [0.2, 0.7, 0.1]])
    assert mapping.match_samples(sim, 0.5) == {0: 0, 1: 1}


def test_match_samples_extra_clusters():
    sim = np.array([[0.8, 0.1, 0.45, 0.3], [0.2, 0.7, 0.1, 0.35]])
    # cluster 2 -> sample 0 (0.45 >= 0.4); cluster 3 is closest to sample 1, but 0.35 < 0.4
    assert mapping.match_samples(sim, 0.5, 0.4) == {0: 0, 1: 1, 2: 0}
    assert mapping.match_samples(sim, 0.5, 0.3) == {0: 0, 1: 1, 2: 0, 3: 1}
    assert mapping.match_samples(sim, 0.5, 0.5) == {0: 0, 1: 1}


def test_match_samples_extra_needs_a_matched_speaker():
    # sample 1 is below threshold everywhere: it gets no extra cluster either
    sim = np.array([[0.8, 0.1, 0.2], [0.3, 0.45, 0.48]])
    assert mapping.match_samples(sim, 0.5, 0.4) == {0: 0}


def test_match_samples_empty():
    assert mapping.match_samples(np.zeros((1, 0)), 0.5) == {}


def test_speaker_names():
    names = mapping.speaker_names(["C", "A", "C", "B"], {"A": "Rav"})
    assert names == {"C": "SPEAKER_00", "A": "Rav", "B": "SPEAKER_01"}


def test_cosine_similarity():
    sim = mapping.cosine_similarity(np.array([[1.0, 0.0]]), np.array([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    np.testing.assert_allclose(sim, [[1.0, 0.0, -1.0]])


def test_best_per_speaker():
    sim = np.array([[0.3, 0.9], [0.7, 0.2], [0.1, 0.4]])
    speakers, best = mapping.best_per_speaker(["Rav", "Rav", "Guest"], sim)
    assert speakers == ["Rav", "Guest"]
    np.testing.assert_allclose(best, [[0.7, 0.9], [0.1, 0.4]])


def test_best_per_speaker_empty():
    speakers, best = mapping.best_per_speaker([], np.zeros((0, 3)))
    assert speakers == [] and best.shape == (0, 3)


def _w(word, start, end, speaker="A"):
    return {"word": word, "start": start, "end": end, "speaker": speaker}


def _seq(texts, speaker="A", start=0.0, step=0.5, gaps=None):
    """Words of 0.4s every `step` s; gaps: {index: extra seconds before that word}."""
    out, t = [], start
    for i, text in enumerate(texts):
        t += (gaps or {}).get(i, 0.0)
        out.append(_w(text, t, t + 0.4, speaker))
        t += step
    return out


def test_short_text_one_cue_one_line():
    cues = build_cues(_seq(["aa", "bb", "cc"]), max_chars=20)
    assert cues == [{"start": 0.0, "end": 1.4, "text": "A: aa bb cc"}]


def test_long_line_wrapped_into_two():
    cues = build_cues(_seq(["aaaa"] * 6), max_chars=15, labels=False)
    assert len(cues) == 1
    assert cues[0]["text"] == "aaaa aaaa aaaa\naaaa aaaa aaaa"


def test_speaker_change_new_line():
    words = _seq(["aa", "bb"], "A") + _seq(["cc", "dd"], "B", start=1.0)
    cues = build_cues(words, max_chars=40)
    assert cues == [{"start": 0.0, "end": 1.9, "text": "A: aa bb\nB: cc dd"}]


def test_three_speakers_need_two_cues():
    words = _seq(["aa"], "A") + _seq(["bb"], "B", start=0.5) + _seq(["cc"], "C", start=1.0)
    cues = build_cues(words, max_chars=40)
    assert [c["text"] for c in cues] == ["A: aa\nB: bb", "C: cc"]


def test_long_pause_forces_cut():
    words = _seq(["aa", "bb", "cc", "dd"], gaps={2: 3.0})
    cues = build_cues(words, max_chars=40, labels=False)
    assert [c["text"] for c in cues] == ["aa bb", "cc dd"]


def test_cut_at_biggest_pause_before_punctuation():
    # 12 words of 4 chars don't fit 2x15; a 0.6s pause before word 4 beats the period after word 2
    texts = ["aaaa", "aaa.", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa"]
    cues = build_cues(_seq(texts, gaps={4: 0.6}), max_chars=15, labels=False)
    assert cues[0]["text"].split() == texts[:4]


def test_cut_at_punctuation_without_pause():
    texts = ["aaaa", "aaaa", "aaaa", "aaa.", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa", "aaaa"]
    cues = build_cues(_seq(texts), max_chars=15, labels=False)
    assert cues[0]["text"].split() == texts[:4]


def test_min_fill_avoids_tiny_cue():
    # the period after word 0 is ignored: cutting there would leave the cue < half full
    texts = ["aaa."] + ["aaaa"] * 10
    cues = build_cues(_seq(texts), max_chars=15, labels=False)
    assert len(cues[0]["text"].split()) > 1


def test_every_word_kept_in_order():
    texts = [f"w{i}" for i in range(50)]
    words = _seq(texts, gaps={10: 0.7, 30: 2.5})
    cues = build_cues(words, max_chars=12, labels=False)
    assert " ".join(c["text"].replace("\n", " ") for c in cues).split() == texts
    assert all(len(c["text"].split("\n")) <= 2 for c in cues)
    assert all(len(line) <= 12 for c in cues for line in c["text"].split("\n"))


def test_names_replace_labels():
    words = _seq(["aa"], "Rav") + _seq(["bb"], "SPEAKER_01", start=0.5)
    cues = build_cues(words, max_chars=40, names={"Rav": "רב"})
    assert cues[0]["text"] == "רב: aa\nSPEAKER_01: bb"


def _c(start, end):
    return {"start": start, "end": end, "text": "x"}


def test_pad_lead_and_tail_in_big_pause():
    out = pad_cues([_c(1.0, 3.0), _c(10.0, 12.0)])
    assert [(c["start"], c["end"]) for c in out] == [(0.5, 4.0), (9.5, 13.0)]


def test_pad_small_pause_split_never_overlaps():
    out = pad_cues([_c(0.0, 2.0), _c(2.45, 4.0)])
    assert out[0]["start"] == 0.0 and out[1]["start"] == pytest.approx(2.25)
    assert out[0]["end"] == pytest.approx(2.1)
    assert out[1]["start"] - out[0]["end"] >= 0.05 - 1e-9


def test_pad_short_cue_stretched_both_sides():
    out = pad_cues([_c(0.0, 2.0), _c(5.0, 5.2), _c(8.0, 10.0)], lead_s=0.0, tail_s=0.0)
    assert out[1]["start"] == pytest.approx(4.35) and out[1]["end"] == pytest.approx(5.85)


def test_pad_short_cue_blocked_before_takes_after():
    out = pad_cues([_c(0.0, 2.0), _c(2.05, 2.25), _c(8.0, 10.0)], lead_s=0.0, tail_s=0.0)
    assert out[1]["start"] == pytest.approx(2.05) and out[1]["end"] == pytest.approx(3.55)


def testspread_evenly():
    words = spread_evenly(["a", "b", "c", "d"], 0.0, 4.0)
    assert [(w["word"], w["start"], w["end"]) for w in words] == [
        ("a", 0.0, 1.0), ("b", 1.0, 2.0), ("c", 2.0, 3.0), ("d", 3.0, 4.0),
    ]


def testapportion_sums_to_total_and_is_proportional():
    counts = apportion(100, [1.0, 0.5, 5.0])
    assert sum(counts) == 100
    assert counts[2] > counts[0] > counts[1]


def testapportion_more_weights_than_total():
    counts = apportion(2, [1.0, 1.0, 1.0, 1.0])
    assert sum(counts) == 2
    assert all(c in (0, 1) for c in counts)


def test_distribute_no_word_spans_a_silence_gap():
    regions = [(0.0, 2.0), (3.0, 5.0)]
    words = ["a", "b", "c", "d"]
    result = distribute_over_regions(words, regions, 0.0, 5.0)
    assert [w["word"] for w in result] == words
    assert result[0]["start"] == 0.0
    assert result[-1]["end"] == 5.0
    for w in result:
        assert not (w["start"] < 2.5 < w["end"])


def test_distribute_every_word_contained_within_one_region():
    regions = [(0.0, 1.0), (2.0, 2.5), (3.0, 8.0)]
    words = list("abcdefghij")
    result = distribute_over_regions(words, regions, 0.0, 8.0)
    assert len(result) == len(words)
    assert [w["word"] for w in result] == words
    for w in result:
        assert any(r_start <= w["start"] and w["end"] <= r_end for r_start, r_end in regions)


def test_distribute_falls_back_to_even_spread_with_no_speech_regions():
    result = distribute_over_regions(["x", "y"], [], 10.0, 14.0)
    assert result[0]["start"] == 10.0
    assert result[-1]["end"] == 14.0


def test_builtin_samples(tmp_path):
    for name in ("Rav.wav", "Other.MP3", "notes.txt", "README.md"):
        (tmp_path / name).write_bytes(b"")
    assert builtin_samples(str(tmp_path)) == [
        ("Other", str(tmp_path / "Other.MP3")),
        ("Rav", str(tmp_path / "Rav.wav")),
    ]


def test_builtin_samples_missing_dir(tmp_path):
    assert builtin_samples(str(tmp_path / "nope")) == []


def test_builtin_samples_speaker_dirs(tmp_path):
    (tmp_path / "Rav").mkdir()
    (tmp_path / "Rav" / "rus_2003.mp3").write_bytes(b"")
    (tmp_path / "Rav" / "heb_2018.wav").write_bytes(b"")
    (tmp_path / "Rav" / "notes.txt").write_bytes(b"")
    (tmp_path / "Guest.wav").write_bytes(b"")
    assert builtin_samples(str(tmp_path)) == [
        ("Guest", str(tmp_path / "Guest.wav")),
        ("Rav", str(tmp_path / "Rav" / "heb_2018.wav")),
        ("Rav", str(tmp_path / "Rav" / "rus_2003.mp3")),
    ]


def _doc(*words):
    return {"kind": "diarize", "words": [{"word": w, "start": s, "end": e, "speaker": sp, "cue": c}
                                         for w, s, e, sp, c in words], "stats": {"speakers": 2, "named": 1}}


def test_speaker_cues_keep_single_speaker_cues_and_split_mixed_ones():
    cues = [{"start": 0.0, "end": 2.0, "text": "a b\nc"}, {"start": 3.0, "end": 8.0, "text": "d e f g"},
            {"start": 9.0, "end": 9.5, "text": ""}]
    doc = _doc(("a", 0.1, 0.5, "Rav", 0), ("b", 0.6, 1.0, "Rav", 0), ("c", 1.1, 1.9, "Rav", 0),
               ("d", 3.2, 3.8, "Rav", 1), ("e", 3.9, 4.8, "Rav", 1),
               ("f", 5.2, 6.0, "SPEAKER_00", 1), ("g", 6.1, 7.9, "SPEAKER_00", 1))
    pieces, split = labels.speaker_cues(cues, doc)
    assert split == 1
    assert pieces == [
        {"start": 0.0, "end": 2.0, "text": "a b\nc", "speaker": "Rav"},
        {"start": 3.0, "end": 4.8, "text": "d e", "speaker": "Rav"},
        {"start": 5.2, "end": 8.0, "text": "f g", "speaker": "SPEAKER_00"},
        {"start": 9.0, "end": 9.5, "text": "", "speaker": None},
    ]
    srt, stats = labels.labeled_srt(cues, doc)
    assert stats == {"speakers": 2, "named": 1, "split_cues": 1}
    assert [c["text"] for c in parse_srt(srt)] == ["רב: a b\nc", "רב: d e", "תלמיד: f g", ""]


def test_words_srt_labels_and_rejects_no_words():
    words = [{"word": "aa", "start": 0.0, "end": 0.4, "speaker": "Rav"},
             {"word": "bb", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_03"}]
    assert labels.words_srt(words, pad=False) == "1\n00:00:00,000 --> 00:00:00,900\nרב: aa\nתלמיד: bb\n"
    with pytest.raises(ValueError):
        labels.words_srt([])


def test_check_refuses_other_documents():
    labels.check(_doc(("a", 0.0, 0.5, None, 0)))
    for bad in ({"words": []}, _doc(("a", 1.0, 0.5, None, 0)), {"kind": "diarize", "words": [{"word": "a"}]}):
        with pytest.raises((ValueError, KeyError)):
            labels.check(bad)
