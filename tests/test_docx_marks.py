"""Reading and highlighting .docx for /docdiff (moved from stt-api docdiff)"""
import io

import docx

from ai_trns.api.docx_marks import RED, extract_text, mark_red

def _doc(*paragraphs: list[tuple[str, bool]]) -> docx.document.Document:
    """Each paragraph is a list of (run text, bold) pieces."""
    d = docx.Document()
    for runs in paragraphs:
        p = d.add_paragraph()
        for text, bold in runs:
            p.add_run(text).bold = bold
    return docx.Document(_save(d))


def _save(d) -> io.BytesIO:
    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    return buf


def _red_words(d) -> list[str]:
    """Red-highlighted text per paragraph, joined."""
    out = []
    for p in d.paragraphs:
        out.append("".join(r.text for r in p.runs if r.font.highlight_color == RED))
    return out


def test_extract_text_joins_paragraphs_and_runs():
    d = _doc([("שלום ", False), ("עול", True), ("ם טוב", False)], [("שני", False)])
    assert extract_text(d) == "שלום עולם טוב\nשני"
    assert extract_text(d).split() == ["שלום", "עולם", "טוב", "שני"]


def test_mark_single_word_inside_run_splits_it():
    d = _doc([("a bb c", False)])
    mark_red(d, [(1, 1)])
    d = docx.Document(_save(d))
    assert [r.text for r in d.paragraphs[0].runs] == ["a ", "bb", " c"]
    assert _red_words(d) == ["bb"]
    assert extract_text(d) == "a bb c"


def test_mark_word_spanning_runs_keeps_formatting():
    d = _doc([("x ab", False), ("cd", True), (" y", False)])
    mark_red(d, [(1, 1)])
    d = docx.Document(_save(d))
    runs = d.paragraphs[0].runs
    assert _red_words(d) == ["abcd"]
    assert [(r.text, bool(r.bold)) for r in runs if r.font.highlight_color == RED] == [("ab", False), ("cd", True)]


def test_mark_range_across_paragraphs():
    d = _doc([("a b", False)], [("c d", False)])
    mark_red(d, [(1, 2)])
    d = docx.Document(_save(d))
    assert _red_words(d) == ["b", "c"]
    assert extract_text(d) == "a b\nc d"


def test_spaces_inside_range_are_red():
    d = _doc([("a b c d", False)])
    mark_red(d, [(1, 2)])
    d = docx.Document(_save(d))
    assert [r.text for r in d.paragraphs[0].runs] == ["a ", "b c", " d"]
    assert _red_words(d) == ["b c"]


def test_indices_match_split_with_empty_paragraph():
    d = _doc([("a", False)], [], [("b c", False)])
    mark_red(d, [(2, 2)])
    assert _red_words(docx.Document(_save(d))) == ["", "", "c"]


def test_no_ranges_leaves_doc_unchanged():
    d = _doc([("a b", False)])
    mark_red(d, [])
    assert _red_words(d) == [""]
