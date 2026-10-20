"""SRT cues from speaker-labeled word timings (the /diarize/words output),
ignoring the original cue boundaries. Model-free.

Rules, most important first:
- a cue has at most max_lines lines of about max_chars characters each;
- a speaker change always starts a new line (so a 2-line cue holds at most
  two speakers);
- a pause of force_gap_s or more always ends the cue;
- otherwise a cue is grown as far as it fits, then cut at the best point
  among those that keep it at least min_fill full: the longest pause first
  (pauses bucketed by _GAP_LEVELS so tiny timing differences don't outrank
  punctuation), then sentence punctuation, then commas & co, then the
  latest point.
A line that is too long for one line is broken the same way, preferring
an even split.
"""
import re

_CLOSERS = re.escape("\"'״׳”’)]»")
_STRONG = re.compile(rf"[.!?…][{_CLOSERS}]*$")
_WEAK = re.compile(rf"[,;:–—-][{_CLOSERS}]*$")
_GAP_LEVELS = (1.0, 0.5, 0.25)  # seconds, descending


def _punct_level(word: str) -> int:
    if _STRONG.search(word):
        return 2
    if _WEAK.search(word):
        return 1
    return 0


def _gap_level(gap: float) -> int:
    for n, threshold in enumerate(_GAP_LEVELS):
        if gap >= threshold:
            return len(_GAP_LEVELS) - n
    return 0


def _break_score(words: list[dict], k: int) -> tuple[int, int]:
    """Quality of a break between words[k - 1] and words[k]."""
    gap = words[k]["start"] - words[k - 1]["end"]
    return _gap_level(gap), _punct_level(words[k - 1]["word"])


def _chars(words: list[dict]) -> int:
    return sum(len(w["word"]) for w in words) + max(0, len(words) - 1)


def _speaker_runs(words: list[dict]) -> list[list[dict]]:
    runs = []
    for w in words:
        if runs and runs[-1][-1]["speaker"] == w["speaker"]:
            runs[-1].append(w)
        else:
            runs.append([w])
    return runs


def _wrap(words: list[dict], max_chars: int) -> list[list[dict]]:
    """One speaker's words -> lines of <= max_chars (a single overlong word
    gets a line of its own). Two lines are split at the best break, else
    lines are filled greedily."""
    if _chars(words) <= max_chars or len(words) == 1:
        return [words]
    fits = [k for k in range(1, len(words))
            if _chars(words[:k]) <= max_chars and _chars(words[k:]) <= max_chars]
    if fits:
        k = max(fits, key=lambda k: (*_break_score(words, k), -abs(_chars(words[:k]) - _chars(words[k:]))))
        return [words[:k], words[k:]]
    lines = [[words[0]]]
    for w in words[1:]:
        if _chars(lines[-1] + [w]) <= max_chars:
            lines[-1].append(w)
        else:
            lines.append([w])
    return lines


def _layout(words: list[dict], max_chars: int, max_lines: int) -> list[list[dict]] | None:
    """Lines of a cue made of these words, or None if they don't fit."""
    lines = []
    for run in _speaker_runs(words):
        lines.extend(_wrap(run, max_chars))
        if len(lines) > max_lines:
            return None
    return lines


def _cue_text(lines: list[list[dict]], labels: bool, names: dict) -> str:
    out, prev = [], None
    for line in lines:
        text = " ".join(w["word"] for w in line)
        speaker = line[0]["speaker"]
        if labels and speaker and speaker != prev:
            text = f"{names.get(speaker, speaker)}: {text}"
        out.append(text)
        prev = speaker
    return "\n".join(out)


def build_cues(
    words: list[dict],
    max_chars: int = 42,
    max_lines: int = 2,
    force_gap_s: float = 2.0,
    min_fill: float = 0.5,
    labels: bool = True,
    names: dict | None = None,
) -> list[dict]:
    """words: [{"word", "start", "end", "speaker"}] in time order.
    Returns [{"start", "end", "text"}] for diarize.srt.write_srt; with
    labels, a line whose speaker differs from the line above (and every
    cue's first line) is prefixed "<speaker>: " (not counted in max_chars),
    or "<names[speaker]>: " for the speakers names has.
    """
    words = [w for w in words if w["word"].strip()]
    cues = []
    i, n = 0, len(words)
    while i < n:
        j, forced = i + 1, False
        while j < n:
            if words[j]["start"] - words[j - 1]["end"] >= force_gap_s:
                forced = True
                break
            if _layout(words[i:j + 1], max_chars, max_lines) is None:
                break
            j += 1

        if j == n or forced:
            k = j
        else:
            # words[i:j] is the most that fits; pick where to cut within it
            ks = range(i + 1, j + 1)
            full = [k for k in ks if _chars(words[i:k]) >= min_fill * max_chars * max_lines]
            k = max(full or ks, key=lambda k: (*_break_score(words, k), k))

        lines = _layout(words[i:k], max_chars, max_lines)
        start, end = words[i]["start"], words[k - 1]["end"]
        cues.append({"start": start, "end": max(end, start + 0.01), "text": _cue_text(lines, labels, names or {})})
        i = k
    return cues


def pad_cues(
    cues: list[dict],
    lead_s: float = 0.5,
    tail_s: float = 1.0,
    min_duration_s: float = 1.5,
    min_gap_s: float = 0.05,
) -> list[dict]:
    """Post-processing of build_cues' timings: cues borrow time from the
    pauses around them, never overlapping (min_gap_s stays between cues).
    - each cue starts up to lead_s earlier, taking at most half the pause before it;
    - each cue ends up to tail_s later, taking at most half of what is left of
      the pause after it (the rest stays empty);
    - a cue still shorter than min_duration_s is stretched into the free time
      on both sides, half before and half after where there is room.
    """
    out = [dict(c) for c in cues]
    n = len(out)
    for i in range(n):
        prev_end = cues[i - 1]["end"] if i else 0.0
        avail = max(0.0, cues[i]["start"] - prev_end - (min_gap_s if i else 0.0))
        lead = min(lead_s, avail / 2 if i else avail)
        out[i]["start"] = cues[i]["start"] - lead
        if i:
            out[i - 1]["end"] = cues[i - 1]["end"] + min(tail_s, (avail - lead) / 2)
    if n:
        out[-1]["end"] = cues[-1]["end"] + tail_s

    for i in range(n):
        need = min_duration_s - (out[i]["end"] - out[i]["start"])
        if need <= 0:
            continue
        room_before = max(0.0, out[i]["start"] - (out[i - 1]["end"] + min_gap_s if i else 0.0))
        room_after = max(0.0, out[i + 1]["start"] - min_gap_s - out[i]["end"]) if i + 1 < n else need
        after = min(room_after, max(need / 2, need - room_before))
        before = min(room_before, need - after)
        out[i]["start"] -= before
        out[i]["end"] += after
    return out
