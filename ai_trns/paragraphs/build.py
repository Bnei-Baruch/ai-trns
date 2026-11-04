"""Diarization document (+ the model's paragraph starts) -> paragraphs. CPU only, stdlib only."""
from typing import Iterable, Optional

from ..diarize import labels

# The speaker whose turns the paragraphing model splits (the model learned on edited lesson texts of the Rav)
MODEL_SPEAKER = "Rav"


def speaker_turns(words: list[dict]) -> list[range]:
    """Runs of words of one speaker, as index ranges. A word without a speaker joins the run before it"""
    turns, start = [], 0
    current = None
    for i, w in enumerate(words):
        speaker = w.get("speaker")
        if i == 0:
            current = speaker
            continue
        if speaker is not None and current is not None and speaker != current:
            turns.append(range(start, i))
            start = i
        if speaker is not None:
            current = speaker
    if words:
        turns.append(range(start, len(words)))
    return turns


def turn_speaker(words: list[dict], turn: range) -> Optional[str]:
    return next((words[i]["speaker"] for i in turn if words[i].get("speaker") is not None), None)


def model_text(words: list[dict], turns: list[range], speaker: str = MODEL_SPEAKER) -> tuple[str, list[int], list[int]]:
    """The text the paragraphing model gets: all turns of `speaker` joined by spaces.
    Returns (text, word index of each token in order, char offset of each of those words in text)"""
    index, offsets, parts, pos = [], [], [], 0
    for turn in turns:
        if turn_speaker(words, turn) != speaker:
            continue
        for i in turn:
            token = " ".join(words[i]["word"].split())
            if not token:
                continue
            index.append(i)
            offsets.append(pos)
            parts.append(token)
            pos += len(token) + 1
    return " ".join(parts), index, offsets


def starts_from_sentences(index: list[int], offsets: list[int], sentences: list[list[int]],
                          breaks: Iterable[int]) -> list[int]:
    """The model's answer (sentence numbers that start a paragraph; sentences as [start, end) char spans of
    the text) -> word indices. A sentence that starts inside a word starts at that word's next word"""
    import bisect
    out = set()
    for b in breaks:
        if not 0 < b < len(sentences):
            continue
        char = sentences[b][0]
        k = bisect.bisect_left(offsets, char)
        if k < len(index):
            out.add(index[k])
    return sorted(out)


def paragraphs(doc: dict) -> dict:
    """The "paragraphs" result of a diarize job"""
    labels.check(doc)
    words = doc["words"]
    model_starts = set(doc.get("paragraph_starts") or [])
    turns = speaker_turns(words)
    out = []
    for turn in turns:
        speaker = turn_speaker(words, turn)
        cuts = [i for i in turn if i in model_starts and i != turn.start]
        bounds = [turn.start, *cuts, turn.stop]
        for a, b in zip(bounds, bounds[1:]):
            ws = words[a:b]
            out.append({
                "speaker": speaker,
                "label": labels.display_name(speaker) if speaker else None,
                "start": ws[0]["start"],
                "end": ws[-1]["end"],
                "text": " ".join(" ".join(w["word"].split()) for w in ws if w["word"].strip()),
                "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in ws],
            })
    stats = {"paragraphs": len(out), "turns": len(turns), "model_breaks": len(out) - len(turns)}
    if (doc.get("stats") or {}).get("paragraphs_error"):
        stats["paragraphs_error"] = doc["stats"]["paragraphs_error"]
    return {"paragraphs": out, "stats": stats}
