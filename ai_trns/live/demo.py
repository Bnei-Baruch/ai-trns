"""
Demo page for live transcription: a player and the text as it is recognized. Stdlib HTTP server,
one stream per run, no auth: for trying the engine out, not for users (they get it through the API).

    GET  /            the page (demo.html)
    GET  /config      {"player": url or null, "lang", "state"}
    GET  /audio       the input file, for the player (Range supported)
    POST /start       start the stream (the page calls it on the first Play, so player and text start together)
    GET  /events      SSE: final / partial / end / error; a new client first gets the final words so far
    GET  /result.srt, /result.json   at the end
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from ..subtitles import render
from .engine import MIN_CHUNK_S, run
from .hub import Hub
from .source import AudioSource
from .stream import StreamProcessor

log = logging.getLogger("ai-trns.live")

PAGE = os.path.join(os.path.dirname(__file__), "demo.html")
_range_re = re.compile(r"bytes=(\d*)-(\d*)")


class Session:
    def __init__(self, source: str, make_processor: Callable[[], StreamProcessor],
                 make_document: Callable[[StreamProcessor, float], dict], min_chunk_s: float = MIN_CHUNK_S,
                 out: Optional[str] = None):
        self.source = source
        self.local_file = os.path.isfile(source) and not source.endswith(".sdp")
        self.make_processor = make_processor
        self.make_document = make_document
        self.min_chunk_s = min_chunk_s
        self.out = out
        self.hub = Hub()
        self.stop = threading.Event()
        self.state = "idle"
        self.document: Optional[dict] = None
        self.lock = threading.Lock()

    def start(self) -> bool:
        with self.lock:
            if self.state != "idle":
                return False
            self.state = "live"
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        try:
            processor = self.make_processor()
            # a file is read at its natural pace, in step with the player
            with AudioSource(self.source, realtime=self.local_file) as audio:
                duration = run(audio, processor, self.hub.publish, self.min_chunk_s, self.stop.is_set)
            self.document = self.make_document(processor, duration)
            if self.out:
                with open(self.out + ".json", "w", encoding="utf-8") as f:
                    json.dump(self.document, f, ensure_ascii=False)
                with open(self.out + ".srt", "w", encoding="utf-8") as f:
                    f.write(render(self.document, "srt"))
            self.state = "ended"
            self.hub.publish({"type": "end", "duration_s": round(duration, 3), "words": []})
        except Exception as e:
            log.exception("live stream failed")
            self.state = "error"
            self.hub.publish({"type": "error", "message": str(e), "words": []})

    def player(self) -> Optional[str]:
        if self.local_file:
            return "audio"
        if self.source.lower().startswith(("http://", "https://")):
            return self.source
        return None


def make_handler(session: Session, lang: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, status: int, body: bytes, content_type: str, headers: Optional[dict] = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, data, status: int = 200):
            self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/":
                with open(PAGE, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            elif path == "/config":
                self._json({"player": session.player(), "lang": lang, "state": session.state})
            elif path == "/audio" and session.local_file:
                self._audio()
            elif path == "/events":
                self._events()
            elif path in ("/result.srt", "/result.json") and session.document:
                if path.endswith(".srt"):
                    self._send(200, render(session.document, "srt").encode(), "application/x-subrip; charset=utf-8",
                               {"Content-Disposition": 'attachment; filename="live.srt"'})
                else:
                    self._json(session.document)
            else:
                self._json({"detail": "not found"}, 404)

        do_HEAD = do_GET

        def do_POST(self):
            if self.path == "/start":
                started = session.start()
                self._json({"started": started, "state": session.state})
            else:
                self._json({"detail": "not found"}, 404)

        def _audio(self):
            size = os.path.getsize(session.source)
            start, end = 0, size - 1
            match = _range_re.fullmatch(self.headers.get("Range", "").strip())
            if match and (match.group(1) or match.group(2)):
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                else:
                    start = max(0, size - int(match.group(2)))
            if start > end:
                self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                return
            with open(session.source, "rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            content_type = mimetypes.guess_type(session.source)[0] or "application/octet-stream"
            headers = {"Accept-Ranges": "bytes"}
            if match:
                headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._send(206 if match else 200, body, content_type, headers)

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for chunk in session.hub.sse():
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(session: Session, host: str, port: int, lang: str) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(session, lang))
    server.daemon_threads = True
    return server
