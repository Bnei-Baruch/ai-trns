"""Runs a live source through a StreamProcessor and reports what it heard as events"""
from __future__ import annotations

import time
from typing import Callable, Optional

from .source import AudioSource
from .stream import SAMPLE_RATE, StreamProcessor, Word

# Decode once at least this much new audio has arrived
MIN_CHUNK_S = 1.0


def event(kind: str, words: list[Word], **extra) -> dict:
    return {"type": kind, "words": [{"word": w.word, "start_s": round(w.start_s, 3), "end_s": round(w.end_s, 3),
                                     "probability": round(w.probability, 3)} for w in words], **extra}


def run(source: AudioSource, processor: StreamProcessor, on_event: Callable[[dict], None],
        min_chunk_s: float = MIN_CHUNK_S, should_stop: Optional[Callable[[], bool]] = None) -> float:
    """Decode the source until it ends or should_stop(); events: final / partial. Returns stream duration"""
    pending, started = 0, False
    while not (should_stop and should_stop()):
        samples = source.read(wait_s=min_chunk_s)
        if samples is None:
            break
        if len(samples) and not started:
            # a player of the same media starts now, so text and sound go together
            started = True
            on_event({"type": "start", "words": []})
        processor.insert_audio(samples)
        pending += len(samples)
        if pending < min_chunk_s * SAMPLE_RATE:
            continue
        pending = 0
        started = time.monotonic()
        final, partial = processor.process()
        # text lags the audio by about min_chunk_s + decode_s; a slow decode just makes the next chunk bigger
        decode_s = round(time.monotonic() - started, 2)
        if final:
            on_event(event("final", final, decode_s=decode_s))
        on_event(event("partial", partial, decode_s=decode_s, end_s=round(processor.end_s, 3)))
    final = processor.finish()
    if final:
        on_event(event("final", final))
    return processor.end_s
