"""
Words -> subtitle cues -> SRT, the "house" layout of the Kabbalah Media archive.

Ported from whisper_trainer (timing_alignment.py: CueAggregationPolicy,
wrap_cue_lines, aggregate_words_into_cues, canonicalize_cues, serialize_srt;
inference_bundle.DEFAULT_CUE_POLICY) at commit 29645c8. The algorithm is kept
as is, so SRT from the same words matches the whisper_trainer service; change
it only together with upstream.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Optional

# whisper_trainer bundle v16 cue_policy: line p99 of the archive's professional cues is 31 chars
HOUSE_POLICY: dict[str, Any] = {
    "max_chars": 84,
    "max_duration_s": 7.0,
    "gap_split_s": 2.0,
    "split_after_chars": ".,?!:;",
    "min_split_chars": 22,
    "line_chars": 32,
    "fit_lines": True,
    "pause_split_min_chars": 10,
}

# Published in place of a sung span, always as a cue of its own.
MUSIC_CUE = "♪"


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TimedCue:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class CueAggregationPolicy:
    """Split a stream of timed words into subtitle cues.

    A new cue starts when the inter-word gap exceeds ``gap_split_s``, or when
    adding the word would exceed ``max_chars`` or ``max_duration_s``. Offsets
    are applied to every produced boundary and are meant to be calibrated
    against held-out professional cue times, not hand-picked.

    ``split_after_chars`` ends a cue after a word carrying sentence
    punctuation, which is how a professional partition breaks; ``min_split_chars``
    keeps that from cutting a cue down to a couple of words.

    ``line_chars`` is the display layout: a cue longer than it is published on
    two balanced lines (`wrap_cue_lines`). ``None`` publishes one line.
    ``fit_lines`` also makes those two lines a capacity, so no cue overflows the
    house layout.

    A capacity split (``max_chars`` or ``fit_lines``) cuts before the word that
    overflows. With ``pause_split_min_chars`` it cuts at the cue's longest
    inter-word pause instead, when that pause is longer, the words before it
    reach that many characters, and the words after it fit with the new word.
    """

    max_chars: int = 84
    max_duration_s: float = 7.0
    gap_split_s: float = 2.0
    start_offset_s: float = 0.0
    end_offset_s: float = 0.0
    split_after_chars: str = ".,?!:;"
    min_split_chars: int = 22
    line_chars: Optional[int] = None
    fit_lines: bool = False
    pause_split_min_chars: Optional[int] = None

    @classmethod
    def from_dict(cls, policy: dict, **overrides) -> "CueAggregationPolicy":
        known = {f.name for f in fields(cls)}
        unknown = set(policy) - known
        if unknown:
            raise ValueError(f"unknown cue policy fields: {sorted(unknown)}")
        return cls(**{**policy, **overrides})

    def validate(self) -> None:
        if self.max_chars <= 0 or self.max_duration_s <= 0 or self.gap_split_s < 0:
            raise ValueError(f"Invalid cue aggregation policy: {self}")
        if self.min_split_chars < 0:
            raise ValueError(f"Invalid cue aggregation policy: {self}")
        if self.line_chars is not None and self.line_chars <= 0:
            raise ValueError(f"Invalid cue aggregation policy: {self}")
        if self.fit_lines and self.line_chars is None:
            raise ValueError(f"fit_lines needs line_chars: {self}")
        if self.pause_split_min_chars is not None and self.pause_split_min_chars < 0:
            raise ValueError(f"Invalid cue aggregation policy: {self}")

    def fits(self, text: str) -> bool:
        if len(text) > self.max_chars:
            return False
        if not self.fit_lines:
            return True
        return all(
            len(line) <= self.line_chars
            for line in wrap_cue_lines(text, self.line_chars).split("\n")
        )


def wrap_cue_lines(text: str, line_chars: Optional[int]) -> str:
    """Lay one cue out as the professional files do: at most two balanced lines.

    House breaks follow length balance rather than syntax, so the break is the
    space that minimizes the longer line, with ties going to the shorter first
    line. Only one space becomes a newline, so joining the lines with a space
    restores the cue text exactly.
    """
    if line_chars is None or len(text) <= line_chars:
        return text
    spaces = [index for index, char in enumerate(text) if char == " "]
    if not spaces:
        return text
    split = min(spaces, key=lambda index: (max(index, len(text) - index - 1), index))
    return f"{text[:split]}\n{text[split + 1 :]}"


def aggregate_words_into_cues(
    word_timings: list[WordTiming], policy: CueAggregationPolicy
) -> list[TimedCue]:
    policy.validate()
    cues: list[TimedCue] = []
    current: list[WordTiming] = []

    def emit(words: list[WordTiming]) -> None:
        cues.append(
            TimedCue(
                " ".join(w.word for w in words),
                float(words[0].start_s + policy.start_offset_s),
                float(words[-1].end_s + policy.end_offset_s),
            )
        )

    def flush() -> None:
        if current:
            emit(current)
            current.clear()

    def text_of(words: list[WordTiming]) -> str:
        return " ".join(w.word for w in words)

    def capacity_split(timing: WordTiming) -> int:
        split, longest = len(current), timing.start_s - current[-1].end_s
        if policy.pause_split_min_chars is None:
            return split
        for index in range(1, len(current)):
            pause = current[index].start_s - current[index - 1].end_s
            if (
                pause > longest
                and len(text_of(current[:index])) >= policy.pause_split_min_chars
                and policy.fits(text_of([*current[index:], timing]))
            ):
                split, longest = index, pause
        return split

    for timing in word_timings:
        if timing.word == MUSIC_CUE:
            flush()
            current.append(timing)
            flush()
            continue
        if current:
            if (
                timing.start_s - current[-1].end_s > policy.gap_split_s
                or timing.end_s - current[0].start_s > policy.max_duration_s
            ):
                flush()
            elif not policy.fits(text_of([*current, timing])):
                split = capacity_split(timing)
                emit(current[:split])
                del current[:split]
        current.append(timing)
        if (
            policy.split_after_chars
            and timing.word.rstrip()[-1:] in policy.split_after_chars
            and len(text_of(current)) >= policy.min_split_chars
        ):
            flush()
    flush()
    return cues


def _round_milliseconds(seconds: float) -> int:
    return math.floor(float(seconds) * 1000.0 + 0.5)


def canonicalize_cues(
    cues: list[TimedCue],
    *,
    duration_s: float,
    max_duration_s: float,
    max_chars: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Clamp, de-overlap, and millisecond-quantize cues without dropping text."""
    duration_ms = _round_milliseconds(duration_s)
    if duration_ms <= 0:
        raise ValueError("Cannot serialize subtitles for empty audio")
    max_duration_ms = max(1, _round_milliseconds(max_duration_s))
    result: list[dict[str, Any]] = []
    repairs = {"clamped": 0, "overlap": 0, "duration": 0, "merged_at_end": 0, "oversized": 0}
    for cue in cues:
        text = " ".join(cue.text.split())
        if not text:
            continue
        if len(text) > max_chars:
            # Cue aggregation never splits a word, so a single token longer than
            # the budget must be published intact rather than lost or rejected.
            if " " in text:
                raise ValueError(f"Cue exceeds max_chars={max_chars}: {text!r}")
            repairs["oversized"] += 1
        raw_start = _round_milliseconds(cue.start_s)
        raw_end = _round_milliseconds(cue.end_s)
        start = min(max(raw_start, 0), duration_ms)
        end = min(max(raw_end, 0), duration_ms)
        repairs["clamped"] += int(start != raw_start or end != raw_end)
        if result and start < result[-1]["end_ms"]:
            start = result[-1]["end_ms"]
            repairs["overlap"] += 1
        if end - start > max_duration_ms:
            end = start + max_duration_ms
            repairs["duration"] += 1
        if end <= start:
            if start < duration_ms:
                end = min(duration_ms, start + 1)
            elif result and len(result[-1]["text"]) + 1 + len(text) <= max_chars:
                result[-1]["text"] += " " + text
                repairs["merged_at_end"] += 1
                continue
            else:
                raise ValueError(
                    "No positive-duration subtitle interval remains for recognized text"
                )
        result.append({"text": text, "start_ms": start, "end_ms": end})
    return result, repairs


def _format_srt_time(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def serialize_srt(cues: list[dict[str, Any]], *, line_chars: Optional[int] = None) -> str:
    blocks = []
    previous_end = 0
    for index, cue in enumerate(cues, start=1):
        start, end = int(cue["start_ms"]), int(cue["end_ms"])
        if start < previous_end or end <= start:
            raise ValueError(f"Invalid subtitle ordering at cue {index}: {start}..{end}")
        previous_end = end
        blocks.append(
            f"{index}\n{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
            f"{wrap_cue_lines(cue['text'], line_chars)}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def build_cues(doc: dict, policy: Optional[dict] = None) -> tuple[list[dict[str, Any]], CueAggregationPolicy]:
    """Words document (ai_trns.words) -> canonical cues [{text, start_ms, end_ms}]"""
    correction = doc.get("timing", {}).get("boundary_correction_s") or {"start": 0.0, "end": 0.0}
    cue_policy = CueAggregationPolicy.from_dict(
        policy if policy is not None else HOUSE_POLICY,
        start_offset_s=float(correction["start"]),
        end_offset_s=float(correction["end"]),
    )
    timings = [WordTiming(w["word"], float(w["start_s"]), float(w["end_s"])) for w in doc["words"]]
    if not timings:
        return [], cue_policy
    cues, _repairs = canonicalize_cues(
        aggregate_words_into_cues(timings, cue_policy),
        duration_s=float(doc["duration_s"]),
        max_duration_s=float(cue_policy.max_duration_s),
        max_chars=int(cue_policy.max_chars),
    )
    # Cues are built from the words only, so no text may be lost or changed on the way
    if " ".join(c["text"] for c in cues).split() != [w.word for w in timings]:
        raise RuntimeError("cue text does not match the words")
    return cues, cue_policy


def render_srt(doc: dict, policy: Optional[dict] = None) -> str:
    cues, cue_policy = build_cues(doc, policy)
    return serialize_srt(cues, line_chars=cue_policy.line_chars)
