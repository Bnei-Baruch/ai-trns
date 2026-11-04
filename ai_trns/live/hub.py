"""
Events of one live stream for any number of listeners, as server-sent events. Stdlib only: the API
server keeps a Hub per live job, the demo page one for its stream.

Events: start (audio is coming), final (words that will not change), partial (the current guess
after them, replaces the previous one), end, error. Each has "type" and "words".
"""
from __future__ import annotations

import json
import threading
import time
from typing import Iterator

PING_S = 15.0
TERMINAL = ("end", "error")


class Hub:
    def __init__(self):
        self.events: list[dict] = []
        self.cond = threading.Condition()
        # listeners connected now, and since when there are none
        self.listeners = 0
        self.idle_since = time.time()

    def publish(self, e: dict):
        with self.cond:
            self.events.append(e)
            self.cond.notify_all()

    @property
    def ended(self) -> bool:
        with self.cond:
            return bool(self.events) and self.events[-1]["type"] in TERMINAL

    def since(self, index: int, timeout: float) -> list[dict]:
        with self.cond:
            if len(self.events) <= index:
                self.cond.wait(timeout)
            return self.events[index:]

    def replay(self) -> tuple[list[dict], int]:
        """Everything but the partials so far, and the latest partial unless ended: what a listener
        joining now needs, instead of every partial since the start"""
        with self.cond:
            events = [e for e in self.events if e["type"] != "partial"]
            partials = [e for e in self.events if e["type"] == "partial"]
            if not any(e["type"] in TERMINAL for e in events):
                events += partials[-1:]
            return events, len(self.events)

    def sse(self, ping_s: float = PING_S) -> Iterator[bytes]:
        """The stream as text/event-stream, until end / error; a comment line every ping_s keeps
        proxies from closing a quiet connection"""
        with self.cond:
            self.listeners += 1
        try:
            events, index = self.replay()
            while True:
                for e in events:
                    yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n".encode()
                if not events:
                    yield b": ping\n\n"
                if any(e["type"] in TERMINAL for e in events):
                    return
                events = self.since(index, ping_s)
                index += len(events)
        finally:
            with self.cond:
                self.listeners -= 1
                if not self.listeners:
                    self.idle_since = time.time()

    def idle_s(self) -> float:
        """How long nobody has listened (0 while someone does)"""
        with self.cond:
            return 0.0 if self.listeners else time.time() - self.idle_since
