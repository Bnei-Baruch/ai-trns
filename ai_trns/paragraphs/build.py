"""Diarization document (+ the model's paragraph starts, + the lesson's source text) -> paragraphs.
CPU only, stdlib only."""
import re
from typing import Iterable, Optional

from ..diarize import labels

# The speaker whose turns the paragraphing model splits (the model learned on edited lesson texts of the Rav)
MODEL_SPEAKER = "Rav"
# The reader (labels.NAMES: קריין): first the voice (cluster) that announces the reading ("קטע מספר 3",
# "אנחנו קוראים") -- its turns, except the ones that relay a question; if no voice does, the turns of someone
# else whose words are mostly in the lesson's source text (runs of READER_NGRAM words the source has too).
# Measured on 3 lessons against edited transcripts (TASK-speakers-paragraphs.md): the announcement alone
# found the reader in all 3, the LIKUTIM of a lesson is sometimes another part's
READER = "Reader"
READER_NGRAM = 4
READER_SHARE = 0.5
READER_MIN_WORDS = 8
READER_CUE_RE = re.compile(r"(קטע|קטעים) (מספר|מס)|נקרא (קטע|קטעים|בספר|מתוך)|אנחנו קוראים")
QUESTION_RE = re.compile(r"שאלה|\?")
READER_CUE_WORDS, QUESTION_WORDS = 20, 12
# Reading the source aloud vs speaking, per word, whoever speaks (the reader also announces, relays questions,
# talks; the Rav sometimes reads himself): runs of at least READING_MIN_WORDS words covered by the source's
# READER_NGRAM-grams, up to READING_GAP other words inside (ASR errors in the read text; not across a speaker
# change). Against the editor's
# reading on 35 lessons (TASK-speakers-paragraphs.md, 2026-11-05): precision 0.99, recall 0.97
READING_MIN_WORDS = 12
READING_GAP = 12

_NIQQUD_RE = re.compile(r"[\u0591-\u05C7]")
_PUNCT_RE = re.compile(r"[^\w\u05D0-\u05EA]+")


def norm(word: str) -> str:
    """A word for comparing texts: no niqqud, no punctuation, lower case"""
    return _PUNCT_RE.sub("", _NIQQUD_RE.sub("", word)).lower()


def _keys(text: str) -> list[str]:
    return [k for k in (norm(w) for w in text.split()) if k]


def _head(words: list[dict], turn: range, n: int) -> str:
    return " ".join(words[i]["word"] for i in turn[:n])


def reader_turns(words: list[dict], turns: list[range], source: Optional[str]) -> set[int]:
    """Indices of the turns (not the Rav's) that read the lesson's source aloud: the turns of the voice that
    announces the reading; if none does, by the source text"""
    others = [(t, turn) for t, turn in enumerate(turns) if turn_speaker(words, turn) not in (MODEL_SPEAKER, None)]
    readers = {turn_speaker(words, turn) for _, turn in others if READER_CUE_RE.search(_head(words, turn, READER_CUE_WORDS))}
    by_voice = {t for t, turn in others if turn_speaker(words, turn) in readers and len(turn) >= READER_MIN_WORDS
                and not QUESTION_RE.search(_head(words, turn, QUESTION_WORDS))}
    return by_voice or (_source_turns(words, turns, source) if source else set())


def reading_words(words: list[dict], source: Optional[str], min_words: int = READING_MIN_WORDS,
                  gap: int = READING_GAP) -> list[bool]:
    """Per word: is it inside a passage read from the source (runs of covered words, see READING_MIN_WORDS)"""
    out = [False] * len(words)
    if not source:
        return out
    index = [i for i, w in enumerate(words) if norm(w["word"])]
    covered = source_covered([norm(words[i]["word"]) for i in index], source_grams(source))
    k, n = 0, len(index)
    while k < n:
        if not covered[k]:
            k += 1
            continue
        last, j = k, k + 1
        while j < n and j - last <= gap + 1:
            if covered[j]:
                last = j
            elif words[index[j]].get("speaker") not in (None, words[index[last]].get("speaker")):
                break  # a gap doesn't go on into someone else's words
            j += 1
        if last + 1 - k >= min_words:
            for i in range(index[k], index[last] + 1):
                out[i] = True
        k = last + 1
    return out


def source_grams(source: str, n: int = READER_NGRAM) -> set[tuple]:
    """The word n-grams of a source text (normalized words)"""
    keys = _keys(source)
    return {tuple(keys[i:i + n]) for i in range(len(keys) - n + 1)}


def source_covered(keys: list[str], grams: set[tuple], n: int = READER_NGRAM) -> list[bool]:
    """Per normalized word: is it inside a run of n words that the source has too"""
    covered = [False] * len(keys)
    for i in range(len(keys) - n + 1):
        if tuple(keys[i:i + n]) in grams:
            covered[i:i + n] = [True] * n
    return covered


def _source_turns(words: list[dict], turns: list[range], source: str) -> set[int]:
    grams = source_grams(source)
    out = set()
    for t, turn in enumerate(turns):
        if turn_speaker(words, turn) in (MODEL_SPEAKER, None):
            continue
        tk = [k for k in (norm(words[i]["word"]) for i in turn) if k]
        if len(tk) < READER_MIN_WORDS:
            continue
        if sum(source_covered(tk, grams)) / len(tk) >= READER_SHARE:
            out.add(t)
    return out


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


def thin_starts(words: list[dict], starts: Iterable[int], min_words: int, by_pause: bool = False) -> list[int]:
    """Paragraph starts at least min_words words after the start before them in the same turn (or the turn's start);
    the others dropped, earliest kept first. by_pause: of two starts too close, the one after the longer pause
    is kept instead (a turn's start always stays). min_words <= 0: all kept"""
    starts = sorted(set(starts))
    if min_words <= 0:
        return starts
    turn_start, out = {}, []
    for turn in speaker_turns(words):
        for i in turn:
            turn_start[i] = turn.start
    last = {}
    for i in starts:
        t = turn_start.get(i)
        if t is None:
            continue
        if i - last.get(t, t) >= min_words:
            out.append(i)
            last[t] = i
        elif by_pause and t in last and _pause(words, i) > _pause(words, last[t]):
            # last[t] is the start kept last (starts come in order): this one takes its place
            out[-1] = last[t] = i
    return out


def _pause(words: list[dict], i: int) -> float:
    """Silence before word i"""
    return words[i]["start"] - words[i - 1]["end"] if i else 0.0


def paragraphs(doc: dict, source: Optional[str] = None) -> dict:
    """The "paragraphs" result of a diarize job; with the lesson's source text, turns reading it are the reader's,
    and every paragraph says if it reads the source (`reading`: a new paragraph where reading starts or stops)"""
    labels.check(doc)
    words = doc["words"]
    model_starts = set(doc.get("paragraph_starts") or [])
    turns = speaker_turns(words)
    readers = reader_turns(words, turns, source)
    reading = reading_words(words, source)
    out, model_breaks = [], 0
    for t, turn in enumerate(turns):
        speaker = READER if t in readers else turn_speaker(words, turn)
        cuts = {i for i in turn if i in model_starts and i != turn.start}
        model_breaks += len(cuts)
        cuts |= {i for i in turn if i != turn.start and reading[i] != reading[i - 1]}
        bounds = [turn.start, *sorted(cuts), turn.stop]
        for a, b in zip(bounds, bounds[1:]):
            ws = words[a:b]
            out.append({
                "speaker": speaker,
                "label": labels.display_name(speaker) if speaker else None,
                "reading": reading[a],
                "start": ws[0]["start"],
                "end": ws[-1]["end"],
                "text": " ".join(" ".join(w["word"].split()) for w in ws if w["word"].strip()),
                "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in ws],
            })
    stats = {"paragraphs": len(out), "turns": len(turns), "model_breaks": model_breaks,
             "reader_turns": len(readers), "reading_words": sum(reading)}
    if (doc.get("stats") or {}).get("paragraphs_error"):
        stats["paragraphs_error"] = doc["stats"]["paragraphs_error"]
    # how sure the speakers' names are (ai_trns.diarize.mapping.voice_report)
    for key in ("voices", "uncertain_voices"):
        if key in (doc.get("stats") or {}):
            stats[key] = doc["stats"][key]
    return {"paragraphs": out, "stats": stats}
