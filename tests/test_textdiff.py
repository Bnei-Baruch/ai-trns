"""Text <-> heard words diff (moved from stt-api tadiff)"""
import pytest

from ai_trns.textdiff import Word, compare, load_words
def _words(text: str, step: float = 1.0) -> list[Word]:
    """Heard words spaced `step` seconds apart: word k spans [k, k + 0.5]."""
    return [Word(w, k * step, k * step + 0.5) for k, w in enumerate(text.split())]


def test_identical():
    r = compare("שלום לכם חברים יקרים", _words("שלום לכם חברים יקרים"))
    assert r["removed"] == r["added"] == r["changed"] == []
    assert r["stats"] == {"text_words": 4, "audio_words": 4}


def test_removed_in_middle_merged_into_range():
    r = compare("a b c d e f", _words("a b e f"))
    assert r["added"] == r["changed"] == []
    assert r["removed"] == [{"text_idx": [2, 3], "text": "c d", "at": 1.5}]


def test_removed_at_start_and_end():
    r = compare("a b c d", _words("b c"))
    assert [e["text_idx"] for e in r["removed"]] == [[0, 0], [3, 3]]
    assert r["removed"][0]["at"] == 0.0
    assert r["removed"][1]["at"] == 1.5


def test_added():
    r = compare("a b c", _words("a x y b c"))
    assert r["removed"] == r["changed"] == []
    assert r["added"] == [{"start": 1.0, "end": 2.5, "heard": "x y", "after_text_idx": 0}]


def test_added_before_first_word():
    r = compare("a b", _words("x a b"))
    assert r["added"][0]["after_text_idx"] == -1


def test_changed_has_text_and_audio_ranges():
    r = compare("a b c d", _words("a q c d"))
    assert r["removed"] == r["added"] == []
    assert r["changed"] == [{"text_idx": [1, 1], "text": "b", "start": 1.0, "end": 1.5, "heard": "q"}]


def test_fuzzy_spelling_variant_is_equal():
    # ktiv male vs haser: one extra letter
    r = compare("אמר שלמה המלך", _words("אמר שלומה המלך"))
    assert r["changed"] == []


def test_fuzzy_threshold_respected():
    r = compare("אמר שלמה המלך", _words("אמר שלומה המלך"), fuzzy_threshold=0.95)
    assert r["changed"][0]["text_idx"] == [1, 1]


def test_fuzzy_splits_replace_block():
    # 'abcdefgh' ~ 'abcdefgx' (fuzzy equal) sits between two real changes
    r = compare("s q abcdefgh w e", _words("s z abcdefgx y e"))
    assert [e["text_idx"] for e in r["changed"]] == [[1, 1], [3, 3]]


def test_punctuation_and_niqqud_do_not_shift_indices():
    text = 'שָׁלוֹם , "עולם" - טוב'
    r = compare(text, _words("שלום עולם"))
    assert r["removed"] == [{"text_idx": [4, 4], "text": "טוב", "at": 1.5}]
    assert r["stats"]["text_words"] == 3  # ',' and '-' are not words


def test_empty_audio():
    r = compare("a b", [])
    assert r["removed"] == [{"text_idx": [0, 1], "text": "a b", "at": 0.0}]


def test_valid_words():
    data = '[{"word": "a", "start": 0.0, "end": 0.5}, {"word": "b", "start": 0.5, "end": 1.0}]'
    words = load_words(data)
    assert [(w.text, w.start, w.end) for w in words] == [("a", 0.0, 0.5), ("b", 0.5, 1.0)]


def test_text_key_accepted():
    data = '[{"text": "a", "start": 0, "end": 1}]'
    words = load_words(data)
    assert words[0].text == "a"


def test_empty_list():
    assert load_words("[]") == []


def test_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_words("not json")


def test_not_a_list():
    with pytest.raises(ValueError, match="JSON array"):
        load_words('{"word": "a", "start": 0, "end": 1}')


def test_missing_word():
    with pytest.raises(ValueError, match="non-empty"):
        load_words('[{"start": 0, "end": 1}]')


def test_missing_start():
    with pytest.raises(ValueError, match="start"):
        load_words('[{"word": "a", "end": 1}]')


def test_start_after_end():
    with pytest.raises(ValueError, match="after end"):
        load_words('[{"word": "a", "start": 1, "end": 0}]')
