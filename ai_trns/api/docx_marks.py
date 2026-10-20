"""
Read text out of a .docx and mark word ranges in it with a red highlight (moved from stt-api
docdiff/docx_io.py).

Tokens are whitespace-separated words within one paragraph (a paragraph boundary always ends a
token), taken over the concatenated text of ALL the paragraph's runs -- including runs nested in
hyperlinks, which python-docx's Paragraph.runs skips. extract_text joins paragraphs with "\n", so
extract_text(doc).split() yields exactly the tokens mark_red indexes -- the same indices
ai_trns.textdiff.compare returns as text_idx.

A word can span several runs (mixed formatting mid-word) and a run can hold several words, so
mark_red splits runs at range boundaries, copying the run's formatting onto each piece, and
highlights only the pieces inside a range.
"""
import copy
import re

from docx.document import Document
from docx.oxml.ns import qn
from docx.enum.text import WD_COLOR_INDEX
from docx.text.run import Run

RED = WD_COLOR_INDEX.RED

_TOKEN_RE = re.compile(r"\S+")


def _paragraphs(doc: Document):
    """Every w:p in the body in document order, table cells included."""
    return list(doc.element.body.iter(qn("w:p")))


def _runs(p) -> list[Run]:
    """p's runs, hyperlink runs included; runs of a paragraph nested inside p
    (text box) belong to that inner paragraph, not to p."""
    return [Run(r, None) for r in p.iter(qn("w:r")) if next(r.iterancestors(qn("w:p"))) is p]


def _para_text(runs: list[Run]) -> str:
    return "".join(r.text for r in runs)


def extract_text(doc: Document) -> str:
    return "\n".join(_para_text(_runs(p)) for p in _paragraphs(doc))


def _token_spans(doc: Document) -> list[tuple[int, int, int]]:
    """Token index -> (paragraph index, char start, char end) within that paragraph's text."""
    spans = []
    for pi, p in enumerate(_paragraphs(doc)):
        spans.extend((pi, m.start(), m.end()) for m in _TOKEN_RE.finditer(_para_text(_runs(p))))
    return spans


# Run children that setting run.text would destroy (images, embedded objects, fields).
_NON_TEXT = {qn(t) for t in ("w:drawing", "w:pict", "w:object", "w:fldChar", "w:instrText")}


def _split_run(run: Run, cuts: list[int]) -> list[Run]:
    """Split `run` at char offsets `cuts` (relative to run.text) -> pieces in order.
    A run holding non-text content is never split (it is highlighted whole)."""
    text = run.text
    bounds = [0] + [c for c in cuts if 0 < c < len(text)] + [len(text)]
    if len(bounds) == 2 or any(child.tag in _NON_TEXT for child in run._r):
        return [run]
    pieces = [run]
    run.text = text[bounds[0]:bounds[1]]
    prev = run._r
    for a, b in zip(bounds[1:], bounds[2:]):
        new_r = copy.deepcopy(run._r)
        prev.addnext(new_r)
        piece = Run(new_r, None)
        piece.text = text[a:b]
        pieces.append(piece)
        prev = new_r
    return pieces


def mark_red(doc: Document, ranges: list[tuple[int, int]]) -> None:
    """Highlight tokens in each closed index range [a, b] red."""
    spans = _token_spans(doc)
    by_para: dict[int, list[tuple[int, int]]] = {}
    for a, b in ranges:
        # one interval per paragraph the range touches, so spaces between its words are highlighted too
        per_para: dict[int, tuple[int, int]] = {}
        for pi, start, end in spans[a:b + 1]:
            s, _ = per_para.get(pi, (start, end))
            per_para[pi] = (s, end)
        for pi, interval in per_para.items():
            by_para.setdefault(pi, []).append(interval)

    paragraphs = _paragraphs(doc)
    for pi, intervals in by_para.items():
        pos = 0
        for run in _runs(paragraphs[pi]):
            length = len(run.text)
            r0, r1 = pos, pos + length
            pos = r1
            hits = [(max(s, r0), min(e, r1)) for s, e in intervals if s < r1 and e > r0]
            if not hits:
                continue
            cuts = sorted({c - r0 for s, e in hits for c in (s, e)})
            offset = r0
            pieces = _split_run(run, cuts)
            for piece in pieces:
                p0, p1 = offset, offset + len(piece.text)
                offset = p1
                if len(pieces) == 1 or any(s <= p0 and p1 <= e for s, e in hits):
                    piece.font.highlight_color = RED
