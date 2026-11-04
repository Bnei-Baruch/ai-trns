"""
Streaming recognition with a non-streaming Whisper: LocalAgreement-2 over a growing audio buffer
(after ufal/whisper_streaming, Macháček et al. 2023).

The buffer is re-decoded every chunk. A word becomes final once two consecutive decodes agree on
it (longest common prefix); the disagreeing tail is the partial hypothesis. Once final words cover
enough of the buffer, the audio before the last final word is dropped, so a decode never sees more
than about TRIM_S + one chunk of audio.

Times are seconds from the start of the stream.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from ..words import build_document

SAMPLE_RATE = 16000
# Drop committed audio once the buffer is longer than this
TRIM_S = 15.0
# Nothing agreed for this long (music, endless hallucination): commit the hypothesis as is
MAX_BUFFER_S = 30.0
# Silent buffer with no words: keep only the tail, speech may be starting there
SILENCE_KEEP_S = 2.0
# Words from a decode starting this far before the last final word are a repeat of it
OVERLAP_S = 0.1
# A pause longer than this, or a sentence end, starts a new segment of the words document
SEGMENT_GAP_S = 1.0
SENTENCE_END = (".", "?", "!")


@dataclass
class Word:
    start_s: float
    end_s: float
    word: str
    probability: float = 1.0


# audio (16 kHz float32), prompt -> words with times relative to the start of the audio
Asr = Callable[[np.ndarray, str], "list[Word]"]


def merge_tokens(tokens) -> list[Word]:
    """One segment's faster-whisper words -> whole words. Only a piece starting with a quote or
    punctuation continues the previous word (the engine splits a gershayim word such as לחו"ל in two);
    a missing leading space is not enough, the Hebrew model often drops it between words"""
    words: list[Word] = []
    for t in tokens:
        text = str(t.word)
        if words and text and not text[0].isspace() and not text[0].isalnum():
            last = words[-1]
            words[-1] = Word(last.start_s, float(t.end), last.word + text.strip(), last.probability)
        elif text.strip():
            words.append(Word(float(t.start), float(t.end), text.strip(), float(t.probability)))
    return words


def align_words(text: str, tokens: list) -> Optional[list[Word]]:
    """Words of `text` timed by the engine's tokens, matched character by character, or None if the
    tokens do not spell the text. faster-whisper's word tokens do not respect word or segment
    boundaries in Hebrew: a segment's first word comes glued to the previous segment's last token
    ("שהבוראעיצב"); a token that spans words shares its time between them by length"""
    chars = []  # (token, index in its text, its length) per non-space character
    for t in tokens:
        piece = "".join(str(t.word).split())
        chars.extend((t, i, len(piece)) for i in range(len(piece)))
    words = text.split()
    if "".join(words) != "".join(str(t.word).strip().replace(" ", "") for t in tokens):
        return None

    def at(token, i: int, size: int, edge: int) -> float:
        start, end = float(token.start), float(token.end)
        return start + (end - start) * (i + edge) / size

    out, pos = [], 0
    for word in words:
        (first, i, size), (last, j, size_last) = chars[pos], chars[pos + len(word) - 1]
        out.append(Word(at(first, i, size, 0), at(last, j, size_last, 1), word, float(first.probability)))
        pos += len(word)
    return out


def _key(word: str) -> str:
    return word.strip(".,?!:;\"'").lower()


class HypothesisBuffer:
    """LocalAgreement-2: commits the longest common prefix of the last two hypotheses"""

    def __init__(self):
        self.committed: list[Word] = []   # final words still inside the audio buffer
        self.previous: list[Word] = []    # last hypothesis after the committed words
        self.new: list[Word] = []
        self.last_end_s = 0.0

    def insert(self, words: list[Word], offset_s: float):
        new = [Word(w.start_s + offset_s, w.end_s + offset_s, w.word, w.probability) for w in words]
        new = [w for w in new if w.start_s > self.last_end_s - OVERLAP_S]
        if new and self.committed and abs(new[0].start_s - self.last_end_s) < 1.0:
            # the decode may repeat the last final words: drop the longest such n-gram (up to 5)
            for n in range(min(len(self.committed), len(new), 5), 0, -1):
                if [_key(w.word) for w in self.committed[-n:]] == [_key(w.word) for w in new[:n]]:
                    new = new[n:]
                    break
        self.new = new

    def flush(self) -> list[Word]:
        agreed = []
        for a, b in zip(self.new, self.previous):
            if _key(a.word) != _key(b.word):
                break
            agreed.append(a)
        self.previous = self.new[len(agreed):]
        self.new = []
        self.committed.extend(agreed)
        if agreed:
            self.last_end_s = agreed[-1].end_s
        return agreed

    def force(self) -> list[Word]:
        """Commit the whole current hypothesis"""
        words, self.new, self.previous = self.new or self.previous, [], []
        self.committed.extend(words)
        if words:
            self.last_end_s = words[-1].end_s
        return words

    def drop_before(self, time_s: float):
        self.committed = [w for w in self.committed if w.end_s > time_s]

    @property
    def partial(self) -> list[Word]:
        return self.previous


class StreamProcessor:
    """Feed audio with insert_audio(), call process() every chunk, finish() at the end"""

    def __init__(self, asr: Asr, prompt_chars: int = 0, trim_s: float = TRIM_S, max_buffer_s: float = MAX_BUFFER_S):
        self.asr = asr
        self.prompt_chars = prompt_chars
        self.trim_s = trim_s
        self.max_buffer_s = max_buffer_s
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset_s = 0.0               # stream time of self.audio[0]
        self.hypothesis = HypothesisBuffer()
        self.final: list[Word] = []

    @property
    def buffer_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @property
    def end_s(self) -> float:
        return self.offset_s + self.buffer_s

    def insert_audio(self, samples: np.ndarray):
        self.audio = np.concatenate([self.audio, samples.astype(np.float32, copy=False)])

    def _prompt(self) -> str:
        """Final text already cut out of the buffer (the decode sees the rest as audio)"""
        if not self.prompt_chars:
            return ""
        inside = len(self.hypothesis.committed)
        before = self.final[:len(self.final) - inside] if inside else self.final
        return " ".join(w.word for w in before)[-self.prompt_chars:]

    def process(self) -> tuple[list[Word], list[Word]]:
        """Decode the buffer -> (newly final words, partial words)"""
        words = self.asr(self.audio, self._prompt())
        self.hypothesis.insert(words, self.offset_s)
        final = self.hypothesis.flush()
        forced = not final and self.buffer_s > self.max_buffer_s
        if forced:
            final = self.hypothesis.force()
        self.final.extend(final)
        if forced:
            # whatever was not heard as words in this much audio is not going to be: drop it
            self._cut(max(self.hypothesis.last_end_s, self.end_s - SILENCE_KEEP_S))
        else:
            self._trim(bool(words))
        return final, self.hypothesis.partial

    def _trim(self, heard: bool):
        if self.buffer_s <= self.trim_s:
            return
        if self.hypothesis.committed:
            self._cut(self.hypothesis.last_end_s)
        elif not heard and not self.hypothesis.partial:
            self._cut(self.end_s - SILENCE_KEEP_S)

    def _cut(self, time_s: float):
        samples = int((time_s - self.offset_s) * SAMPLE_RATE)
        if samples <= 0:
            return
        self.audio = self.audio[samples:]
        self.offset_s += samples / SAMPLE_RATE
        self.hypothesis.drop_before(self.offset_s)

    def finish(self) -> list[Word]:
        """End of stream: the partial hypothesis becomes final"""
        final = self.process()[0] if len(self.audio) else []
        forced = self.hypothesis.force()
        self.final.extend(forced)
        return final + forced


def segments_of(words: list[Word]) -> list:
    """Final words -> faster-whisper-like segments: a new one after a pause or a sentence end"""
    segments, current = [], []
    for w in words:
        if current and (w.start_s - current[-1].end_s > SEGMENT_GAP_S or current[-1].word.endswith(SENTENCE_END)):
            segments.append(current)
            current = []
        current.append(w)
    if current:
        segments.append(current)
    return [SimpleNamespace(start=seg[0].start_s, end=seg[-1].end_s, text=" ".join(w.word for w in seg),
                            words=[SimpleNamespace(word=w.word, start=w.start_s, end=w.end_s,
                                                   probability=w.probability) for w in seg])
            for seg in segments]


def build_live_document(words: list[Word], duration_s: float, language: Optional[str], model: str,
                        boundary_correction_s: Optional[dict] = None) -> dict:
    doc = build_document(segments_of(words), duration_s, language, model, boundary_correction_s)
    doc["timing"]["source"] = "live"
    return doc


class WhisperAsr:
    """Asr on a loaded ai_trns.transcriber.Transcriber: greedy decode of the whole buffer, VAD on"""

    OPTIONS = dict(
        beam_size=1,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    def __init__(self, transcriber, **options):
        self.transcriber = transcriber
        self.options = {**copy.deepcopy(self.OPTIONS), **options}

    def __call__(self, audio: np.ndarray, prompt: str) -> list[Word]:
        lang = self.transcriber.lang
        segments, _ = self.transcriber.model.transcribe(
            audio, language=None if lang == "auto" else lang, word_timestamps=True,
            initial_prompt=prompt or None, **self.options)
        segments = list(segments)
        tokens = [w for s in segments for w in (s.words or [])]
        text = " ".join(str(s.text).strip() for s in segments)
        aligned = align_words(text, tokens)
        return aligned if aligned is not None else [w for s in segments for w in merge_tokens(s.words or [])]
