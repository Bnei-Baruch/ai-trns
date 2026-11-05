"""
A part of the media: a job's params["clip"] = {"start": s, "end": e or None} (seconds of the media; no end -- to
its end). The worker cuts that part of the audio and processes only it; the result's times stay the media's
(a word at 3m20s of the lesson is at 200 s), so a kabbalahmedia link's sstart/send and the player agree with it.
Whisper_trainer's service takes whole media URLs only: with it, the whole media is transcribed and the cues of the
part are kept (cut_srt).
"""
import re
from typing import Optional

from .diarize.srt import parse_srt, write_srt

_HMS_RE = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s?)?$")


def parse_time(value) -> float:
    """Seconds from 197, "197.5", "3m17s", "1h2m3s", "03:17", "1:02:03"; ValueError otherwise"""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower()
        if ":" in text:
            parts = text.split(":")
            if len(parts) > 3 or not all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
                raise ValueError(f"bad time {value!r}")
            seconds = 0.0
            for p in parts:
                seconds = seconds * 60 + float(p)
        else:
            m = _HMS_RE.match(text)
            if not text or not m or not any(m.groups()):
                raise ValueError(f"bad time {value!r}")
            h, mi, s = (float(g) if g else 0.0 for g in m.groups())
            seconds = h * 3600 + mi * 60 + s
    if not seconds >= 0 or seconds == float("inf"):
        raise ValueError(f"bad time {value!r}")
    return seconds


def make(start, end) -> Optional[dict]:
    """The params["clip"] of a job from the client's start / end (any parse_time form, empty = not given);
    None for the whole media; ValueError if end is not after start"""
    start = None if start in (None, "") else parse_time(start)
    end = None if end in (None, "") else parse_time(end)
    if not start and end is None:
        return None
    if end is not None and end <= (start or 0.0):
        raise ValueError("clip: end must be after start")
    return {"start": round(start or 0.0, 3), "end": None if end is None else round(end, 3)}


def window(clip: dict, duration: float) -> tuple[float, float]:
    """(start, end) of the clip inside media of this duration; ValueError if it starts after the media ends"""
    start = float(clip["start"])
    end = duration if clip.get("end") is None else min(float(clip["end"]), duration or float(clip["end"]))
    if end <= start:
        raise ValueError(f"clip: starts at {start:.1f} s, the media is {duration:.1f} s long")
    return start, end


def cues_to_clip(cues: list[dict], start: float, end: float) -> list[dict]:
    """Media-time cues -> clip-time cues for the clip's audio, one per cue (diarize keeps cue indices): a cue
    outside the clip gets no text (skipped), one across its edge is clamped to it"""
    out = []
    for c in cues:
        s, e = max(c["start"], start), min(c["end"], end)
        inside = e > s
        out.append({**c, "start": round(s - start, 3) if inside else 0.0,
                    "end": round(e - start, 3) if inside else 0.0, "text": c["text"] if inside else ""})
    return out


def shift_words_doc(doc: dict, start: float) -> dict:
    """A words document (ai_trns.words) of the clip's audio -> media time"""
    if not start:
        return doc
    for s in doc.get("segments", []):
        s["start_s"] = round(s["start_s"] + start, 3)
        s["end_s"] = round(s["end_s"] + start, 3)
    for w in doc.get("words", []):
        w["start_s"] = round(w["start_s"] + start, 3)
        w["end_s"] = round(w["end_s"] + start, 3)
    doc["duration_s"] = round(doc["duration_s"] + start, 3)
    return doc


def shift_diarize_doc(doc: dict, start: float) -> dict:
    """A diarization document (ai_trns.diarize) of the clip's audio -> media time"""
    if not start:
        return doc
    for w in doc.get("words", []):
        w["start"] = round(w["start"] + start, 3)
        w["end"] = round(w["end"] + start, 3)
    doc["duration_s"] = round(doc["duration_s"] + start, 3)
    return doc


def cut_srt(srt: str, start: float, end: Optional[float]) -> str:
    """The cues of an SRT whose middle is inside [start, end) (end None: to the end), times kept"""
    keep = [c for c in parse_srt(srt)
            if (c["start"] + c["end"]) / 2 >= start and (end is None or (c["start"] + c["end"]) / 2 < end)]
    return write_srt(keep)
