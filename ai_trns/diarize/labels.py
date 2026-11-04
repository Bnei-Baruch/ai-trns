"""Diarization document (ai_trns.diarize) -> SRT. CPU only, stdlib only."""
from . import KIND, resegment
from .srt import write_srt

# Display labels in SRT / paragraphs output; the document keeps the raw ids. Three labels, no names of people:
# the Rav (voice sample "Rav"), the reader of the source ("Reader": voice samples of readers, if any), the rest
NAMES = {"Rav": "רב", "Reader": "קריין"}
ANONYMOUS = "תלמיד"

# Layout of an SRT rebuilt from words (resegment)
MAX_CHARS = 42
MAX_LINES = 2
FORCE_GAP_S = 2.0
MIN_FILL = 0.5
LEAD_S = 0.5
TAIL_S = 1.0
MIN_DURATION_S = 1.5
MIN_GAP_S = 0.05


def display_name(speaker: str) -> str:
    """NAMES[speaker] if set, else ANONYMOUS for an anonymous SPEAKER_NN, else the speaker as is"""
    if speaker in NAMES:
        return NAMES[speaker]
    if speaker.startswith("SPEAKER_") and ANONYMOUS:
        return ANONYMOUS
    return speaker


def check(doc: dict) -> None:
    """ValueError unless doc is a diarization document"""
    if doc.get("kind") != KIND or not isinstance(doc.get("words"), list):
        raise ValueError("not a diarization document")
    for i, w in enumerate(doc["words"]):
        if not isinstance(w.get("word"), str) or not isinstance(w.get("cue"), int) \
                or not 0 <= float(w["start"]) <= float(w["end"]):
            raise ValueError(f"bad word {i}")


def speaker_cues(cues: list[dict], doc: dict) -> tuple[list[dict], int]:
    """The input cues, each labeled by its speaker and split where the speaker changes.
    Returns ([{"start", "end", "text", "speaker"}], number of cues split).
    A cue with one speaker keeps its text (line breaks too); the pieces of a split cue are its
    words joined by spaces, timed by them, the first and last piece keeping the cue's edges."""
    by_cue = {}
    for w in doc["words"]:
        by_cue.setdefault(w["cue"], []).append(w)
    out, split = [], 0
    for i, cue in enumerate(cues):
        runs = []
        for w in by_cue.get(i, []):
            if runs and runs[-1][-1]["speaker"] == w["speaker"]:
                runs[-1].append(w)
            else:
                runs.append([w])
        if len(runs) <= 1:
            out.append({**cue, "speaker": runs[0][0]["speaker"] if runs else None})
            continue
        split += 1
        for n, run in enumerate(runs):
            out.append({
                "start": cue["start"] if n == 0 else max(cue["start"], run[0]["start"]),
                "end": cue["end"] if n == len(runs) - 1 else min(cue["end"], run[-1]["end"]),
                "text": " ".join(w["word"] for w in run),
                "speaker": run[0]["speaker"],
            })
    return out, split


def labeled_srt(cues: list[dict], doc: dict) -> tuple[str, dict]:
    """The input SRT with each cue prefixed "<label>: ", split at speaker changes -> (SRT, stats)"""
    pieces, split = speaker_cues(cues, doc)
    srt = write_srt([{**p, "text": f"{display_name(p['speaker'])}: {p['text']}" if p["speaker"] and p["text"]
                      else p["text"]} for p in pieces])
    stats = doc.get("stats") or {}
    return srt, {"speakers": stats.get("speakers", 0), "named": stats.get("named", 0), "split_cues": split}


def words_srt(words: list[dict], max_chars: int = MAX_CHARS, max_lines: int = MAX_LINES,
              labels: bool = True, pad: bool = True) -> str:
    """A new SRT from speaker-labeled words [{"word", "start", "end", "speaker"}], ignoring the
    input cues (resegment.build_cues); pad extends cues into the pauses around them"""
    words = sorted(words, key=lambda w: w["start"])
    cues = resegment.build_cues(
        words, max_chars=max_chars, max_lines=max_lines, force_gap_s=FORCE_GAP_S, min_fill=MIN_FILL,
        labels=labels, names={w["speaker"]: display_name(w["speaker"]) for w in words if w["speaker"]})
    if not cues:
        raise ValueError("No words")
    if pad:
        cues = resegment.pad_cues(cues, lead_s=LEAD_S, tail_s=TAIL_S, min_duration_s=MIN_DURATION_S,
                                  min_gap_s=MIN_GAP_S)
    return write_srt(cues)
