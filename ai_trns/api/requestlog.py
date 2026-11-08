"""
Request log: every request to the API, one JSON line each, in AI_TRNS_LOG_DIR/requests.log. A new file starts at
midnight (the old one becomes requests.log.YYYY-MM-DD), AI_TRNS_LOG_DAYS old files are kept (7: a week), older ones
are removed. Empty AI_TRNS_LOG_DIR: off.

    {"time": "2026-11-08T12:00:00+00:00", "ip": "1.2.3.4", "peer": "10.66.24.1", "method": "POST", "path": "/markup",
     "query": "", "status": 202, "ms": 120, "who": "user@x", "ua": "...", "origin": "https://...", "note": "uid=..."}

ip -- the client as the proxy (Angie) reports it: X-Real-IP (the proxy sets it, a client cannot), else
X-Forwarded-For's first address (a client may have put it there), else the peer;
peer -- the TCP peer; who -- the caller once authenticated (auth.principal), "-" otherwise (workers, public reads,
rejected calls); note -- what an endpoint adds about the call (request.state.note). A streamed response (SSE) is
logged when it ends, its ms is the stream's length.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler

from . import config

FILE = "requests.log"

_log = logging.getLogger("ai-trns.requests")
_log.setLevel(logging.INFO)
_log.propagate = False      # the file only, not the service's own log
_lock = threading.Lock()
_handler = None
_handler_dir = None


def _ready() -> bool:
    """The file handler for config.LOG_DIR (made on first use, so a changed setting takes effect)"""
    global _handler, _handler_dir
    if not config.LOG_DIR:
        return False
    with _lock:
        if _handler_dir != config.LOG_DIR:
            if _handler:
                _log.removeHandler(_handler)
                _handler.close()
            os.makedirs(config.LOG_DIR, exist_ok=True)
            _handler = TimedRotatingFileHandler(os.path.join(config.LOG_DIR, FILE), when="midnight",
                                                backupCount=config.LOG_DAYS, encoding="utf-8")
            _handler.setFormatter(logging.Formatter("%(message)s"))
            _log.addHandler(_handler)
            _handler_dir = config.LOG_DIR
    return True


def _header(headers: dict, name: bytes) -> str:
    return headers.get(name, b"").decode("latin-1")


class RequestLog:
    """ASGI middleware (not BaseHTTPMiddleware: it leaves streamed responses as they are)"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = time.monotonic()
        status = 500

        async def send_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_status)
        finally:
            try:
                self._write(scope, status, start)
            except Exception:
                logging.getLogger("ai-trns").exception("request log: cannot write")

    @staticmethod
    def _write(scope, status: int, start: float):
        if not _ready():
            return
        headers = dict(scope.get("headers") or [])
        peer = (scope.get("client") or ("", 0))[0]
        forwarded = _header(headers, b"x-forwarded-for").split(",")[0].strip()
        state = scope.get("state") or {}
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "ip": _header(headers, b"x-real-ip") or forwarded or peer, "peer": peer,
                 "method": scope.get("method"), "path": scope.get("path"),
                 "query": scope.get("query_string", b"").decode("latin-1"), "status": status,
                 "ms": round((time.monotonic() - start) * 1000), "who": state.get("who", "-"),
                 "ua": _header(headers, b"user-agent"),
                 "origin": _header(headers, b"origin") or _header(headers, b"referer")}
        if state.get("note"):
            entry["note"] = state["note"]
        _log.info(json.dumps(entry, ensure_ascii=False))
