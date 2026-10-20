"""Minimal SRT reader/writer.

A cue is {"start": float_sec, "end": float_sec, "text": str}; text keeps its
line breaks. Index lines are optional on input and regenerated on output.
"""
import re

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
_TIMING_RE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")


def _seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _timestamp(t: float) -> str:
    total_ms = max(0, round(t * 1000))
    h, rest = divmod(total_ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt(data: str) -> list[dict]:
    lines = data.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    cues = []
    cur = None
    for lineno, line in enumerate(lines, 1):
        m = _TIMING_RE.match(line)
        if m:
            if cur is not None:
                cues.append(cur)
            start, end = _seconds(*m.groups()[:4]), _seconds(*m.groups()[4:])
            if start > end:
                raise ValueError(f"SRT line {lineno}: start after end")
            cur = {"start": start, "end": end, "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
        elif line.strip() and not line.strip().isdigit():
            raise ValueError(f"SRT line {lineno}: text before the first timing line")
    if cur is not None:
        cues.append(cur)

    result = []
    for i, c in enumerate(cues):
        text_lines = c["lines"]
        # drop trailing blank lines and (for all but the last cue) the next cue's index line
        while text_lines and not text_lines[-1].strip():
            text_lines.pop()
        if i < len(cues) - 1 and text_lines and text_lines[-1].strip().isdigit():
            text_lines.pop()
            while text_lines and not text_lines[-1].strip():
                text_lines.pop()
        result.append({"start": c["start"], "end": c["end"], "text": "\n".join(text_lines).strip()})

    if not result:
        raise ValueError("No SRT cues found")
    return result


def write_srt(cues: list[dict]) -> str:
    blocks = [
        f"{i}\n{_timestamp(c['start'])} --> {_timestamp(c['end'])}\n{c['text']}\n"
        for i, c in enumerate(cues, 1)
    ]
    return "\n".join(blocks)
