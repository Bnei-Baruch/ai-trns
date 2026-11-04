"""
A live job's text written into a Google Doc by the server, as one Google account
(AI_TRNS_GOOGLE_CREDENTIALS, a JSON file: a service account's key, or an "authorized_user" with a
refresh token). Off without it. Users also write docs of their own from the browser (ai-tools); the
API only keeps the list of those (ai_trns.api.live).

The server's doc is created when a live job is created (the first listener's start): in
AI_TRNS_GOOGLE_FOLDER if set (a service account has no Drive of its own: a folder in a shared drive
it is a member of), shared with anyone who has the link as AI_TRNS_GOOGLE_DOC_ROLE if set. Final words
are appended to its end every WRITE_S; people edit the doc meanwhile, and Google keeps who changed
what. Stdlib + PyJWT only (a service account's token is a JWT signed here).
"""
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import jwt

from . import config

log = logging.getLogger("ai-trns.gdoc")

SCOPE = "https://www.googleapis.com/auth/documents https://www.googleapis.com/auth/drive"
DOCS = "https://docs.googleapis.com/v1/documents"
DRIVE = "https://www.googleapis.com/drive/v3/files"
TOKEN_URI = "https://oauth2.googleapis.com/token"
WRITE_S = 5.0
# A doc that keeps failing this many writes in a row is given up on
MAX_FAILURES = 6

_doc_id_re = re.compile(r"/document/(?:u/\d+/)?d/([A-Za-z0-9_-]{20,})")


class DocError(Exception):
    pass


def doc_id(url: str) -> str:
    match = _doc_id_re.search(url)
    if not match:
        raise DocError("not a Google Docs link (https://docs.google.com/document/d/...)")
    return match.group(1)


class Account:
    """Access tokens for the server's Google account, refreshed before they expire. The JSON file is a
    service account's key ("type": "service_account") or a user's OAuth client and refresh token
    ("type": "authorized_user", as `gcloud auth application-default login` writes it; its scopes must
    include SCOPE)"""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.info = json.load(f)
        self.kind = self.info.get("type", "service_account")
        if self.kind not in ("service_account", "authorized_user"):
            raise ValueError(f"unsupported credentials type {self.kind!r}")
        self.email = self.info.get("client_email")
        self.token, self.expires = "", 0.0
        self.lock = threading.Lock()

    def _grant(self) -> bytes:
        if self.kind == "authorized_user":
            return urllib.parse.urlencode({"grant_type": "refresh_token", "client_id": self.info["client_id"],
                                           "client_secret": self.info["client_secret"],
                                           "refresh_token": self.info["refresh_token"]}).encode()
        now = int(time.time())
        assertion = jwt.encode({"iss": self.email, "scope": SCOPE, "aud": self.info.get("token_uri", TOKEN_URI),
                                "iat": now, "exp": now + 3600}, self.info["private_key"], algorithm="RS256")
        return urllib.parse.urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                                       "assertion": assertion}).encode()

    def access_token(self) -> str:
        with self.lock:
            if time.time() < self.expires - 60:
                return self.token
            reply = _call("POST", self.info.get("token_uri", TOKEN_URI), self._grant(),
                          {"Content-Type": "application/x-www-form-urlencoded"})
            self.token, self.expires = reply["access_token"], time.time() + int(reply.get("expires_in", 3600))
            return self.token


def _call(method: str, url: str, body: Optional[bytes] = None, headers: Optional[dict] = None) -> dict:
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise DocError(f"Google {e.code}: {detail}"[:300]) from None
    except urllib.error.URLError as e:
        raise DocError(f"Google is not reachable: {e.reason}") from None


class Docs:
    def __init__(self, account: Account):
        self.account = account

    def _api(self, method: str, url: str, payload: Optional[dict] = None) -> dict:
        headers = {"Authorization": f"Bearer {self.account.access_token()}"}
        body = None
        if payload is not None:
            body, headers["Content-Type"] = json.dumps(payload).encode(), "application/json"
        return _call(method, url if url.startswith("https://") else f"{DOCS}/{url}", body, headers)

    def create(self, title: str, folder: str = "", role: str = "") -> str:
        """A new doc of the account's (in `folder` if given), shared with anyone who has the link as
        `role` if given ("reader", "commenter", "writer"); its id"""
        meta = {"name": title, "mimeType": "application/vnd.google-apps.document",
                **({"parents": [folder]} if folder else {})}
        doc = self._api("POST", f"{DRIVE}?supportsAllDrives=true&fields=id", meta)["id"]
        if role:
            self._api("POST", f"{DRIVE}/{doc}/permissions?supportsAllDrives=true", {"type": "anyone", "role": role})
        return doc

    def open(self, doc: str) -> tuple[str, bool]:
        """The doc's title and whether it is empty; makes it right-to-left"""
        info = self._api("GET", doc)
        end = info["body"]["content"][-1]["endIndex"]
        if end > 1:
            self._api("POST", f"{doc}:batchUpdate", {"requests": [{"updateParagraphStyle": {
                "range": {"startIndex": 1, "endIndex": end}, "paragraphStyle": {"direction": "RIGHT_TO_LEFT"},
                "fields": "direction"}}]})
        return info.get("title", ""), end <= 2

    def append(self, doc: str, text: str):
        self._api("POST", f"{doc}:batchUpdate",
                  {"requests": [{"insertText": {"text": text, "endOfSegmentLocation": {}}}]})


_account: Optional[Account] = None


def service() -> Optional[Docs]:
    """The server's account, or None: AI_TRNS_GOOGLE_CREDENTIALS not set or unusable"""
    global _account
    if not config.GOOGLE_CREDENTIALS:
        return None
    if _account is None:
        try:
            _account = Account(config.GOOGLE_CREDENTIALS)
        except (OSError, ValueError, KeyError) as e:
            log.warning("AI_TRNS_GOOGLE_CREDENTIALS unusable: %s", e)
            return None
    return Docs(_account)


def doc_url(doc: str) -> str:
    return f"https://docs.google.com/document/d/{doc}/edit"


class Writer:
    """Appends a live job's final words (from its Hub) to a doc until the stream ends or stop()"""

    def __init__(self, docs: Docs, doc: str, hub, on_status, empty: bool):
        self.docs, self.doc, self.hub, self.on_status = docs, doc, hub, on_status
        self.index = 0          # next hub event to read
        self.started = not empty  # a doc with text: the first words need a space before them
        self.stopped = threading.Event()
        threading.Thread(target=self._loop, name=f"gdoc-{doc[:8]}", daemon=True).start()

    def stop(self):
        self.stopped.set()

    def _pending(self) -> tuple[str, int, bool]:
        events = self.hub.events[self.index:]
        words = [w["word"] for e in events if e["type"] == "final" for w in e["words"]]
        ended = any(e["type"] in ("end", "error") for e in events)
        return " ".join(words), self.index + len(events), ended

    def _loop(self):
        failures = 0
        while True:
            text, index, ended = self._pending()
            if text:
                try:
                    self.docs.append(self.doc, (" " if self.started else "") + text)
                    self.started, self.index, failures = True, index, 0
                except DocError as e:
                    failures += 1
                    log.warning("doc %s: %s", self.doc, e)
                    if failures >= MAX_FAILURES:
                        self.on_status("error", str(e))
                        return
            else:
                self.index = index
            if ended or self.stopped.is_set():
                return
            self.stopped.wait(WRITE_S)
