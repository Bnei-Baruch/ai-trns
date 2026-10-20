"""
The words document: the boundary between speech recognition (GPU, slow) and
everything built from it (SRT, txt, speakers; CPU, fast).

    {
      "version": 1,
      "duration_s": float,               # audio length, cues are clamped to it
      "language": "he" | ...,
      "model": str,
      "timing": {"source": "engine_words",
                 "boundary_correction_s": {"start": float, "end": float}},
      "text": str,                       # transcript: segment texts joined by spaces
      "segments": [{"start_s", "end_s", "text"}],
      "words": [{"word", "start_s", "end_s", "probability", "interpolated", "segment"}]
    }

Word times are the engine's own, uncorrected: the calibrated boundary
correction belongs to the model and is applied by the cue builder
(ai_trns.cues), since shifting every word would change the pauses cues are cut at.
Joining the words with spaces gives exactly the transcript split on whitespace.

Word reconstruction is ported from whisper_trainer alignment_sources.py
(engine_segment_timings, interpolated fallback) at commit 29645c8.
"""
from __future__ import annotations

from typing import Any, Optional

VERSION = 1
MIN_ALIGNMENT_WINDOW_S = 1.0
MAX_ALIGNMENT_WINDOW_S = 30.0


def _engine_timings(tokens: list, text: str, duration_s: float) -> Optional[list[dict]]:
    """This segment's words carrying the engine's times, or None.

    The engine times its own tokens, and it splits a Hebrew gershayim word such
    as `לחו"ל` into two of them, so consecutive tokens are merged back onto the
    word they spell. A token list that does not spell the segment is refused
    rather than repaired.
    """
    if not tokens:
        return None
    words = []
    consumed = 0
    for word in text.split():
        first, spelled = consumed, ""
        while consumed < len(tokens) and spelled != word:
            spelled += tokens[consumed]["word"]
            consumed += 1
        if spelled != word:
            return None
        words.append({"word": word, "start_s": tokens[first]["start_s"],
                      "end_s": tokens[consumed - 1]["end_s"],
                      "probability": tokens[first]["probability"], "interpolated": False})
    if consumed != len(tokens):
        return None
    previous_start = 0.0
    for w in words:
        if not (0.0 <= w["start_s"] <= w["end_s"] <= duration_s + 1e-3) or w["start_s"] < previous_start:
            return None
        previous_start = w["start_s"]
    return words


def _interpolated_timings(start_s: float, end_s: float, text: str, duration_s: float) -> list[dict]:
    """No usable engine times: split the segment's window proportionally to word lengths"""
    start_s = max(0.0, min(start_s, duration_s))
    end_s = min(duration_s, max(end_s, start_s + MIN_ALIGNMENT_WINDOW_S), start_s + MAX_ALIGNMENT_WINDOW_S)
    if end_s <= start_s:
        start_s, end_s = max(0.0, duration_s - 1.0), duration_s
    words = text.split()
    weights = [max(len(w), 1) for w in words]
    total, cursor, out = sum(weights), 0, []
    for word, weight in zip(words, weights):
        w_start = start_s + (end_s - start_s) * cursor / total
        cursor += weight
        out.append({"word": word, "start_s": w_start, "end_s": start_s + (end_s - start_s) * cursor / total,
                    "probability": None, "interpolated": True})
    return out


def build_document(segments, duration_s: float, language: Optional[str], model: str,
                   boundary_correction_s: Optional[dict] = None) -> dict[str, Any]:
    """faster-whisper segments (with or without .words) -> words document"""
    doc_segments, doc_words = [], []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        index = len(doc_segments)
        doc_segments.append({"start_s": float(segment.start), "end_s": float(segment.end), "text": text})
        tokens = [{"word": str(w.word).strip(), "start_s": float(w.start), "end_s": float(w.end),
                   "probability": float(w.probability)} for w in (segment.words or [])]
        words = (_engine_timings(tokens, text, duration_s)
                 or _interpolated_timings(float(segment.start), float(segment.end), text, duration_s))
        for w in words:
            w["segment"] = index
        doc_words.extend(words)
    return {
        "version": VERSION,
        "duration_s": float(duration_s),
        "language": language,
        "model": model,
        "timing": {"source": "engine_words",
                   "boundary_correction_s": boundary_correction_s or {"start": 0.0, "end": 0.0}},
        "text": " ".join(s["text"] for s in doc_segments),
        "segments": doc_segments,
        "words": doc_words,
    }
