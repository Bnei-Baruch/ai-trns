"""The server's Google Doc of a live job, against a fake Google API, and the list of listeners' docs"""
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.conftest import H, W
from tests.test_auth import oidc  # noqa: F401  (fixture)
from tests.test_live_api import sse

DOC = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
URL = f"https://docs.google.com/document/d/{DOC}/edit"
STREAM = "http://cdn.example.org/live/heb/playlist.m3u8"


def service_account(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()
    path = tmp_path / "sa.json"
    path.write_text(json.dumps({"type": "service_account", "client_email": "ai-trns@proj.iam.gserviceaccount.com",
                                "private_key": key}))
    return path


@pytest.fixture
def google(tmp_path, monkeypatch, api_config):
    from ai_trns.api import gdoc
    monkeypatch.setattr(api_config, "GOOGLE_CREDENTIALS", str(service_account(tmp_path)))
    monkeypatch.setattr(api_config, "GOOGLE_FOLDER", "folder1")
    monkeypatch.setattr(api_config, "GOOGLE_DOC_ROLE", "writer")
    monkeypatch.setattr(gdoc, "_account", None)
    monkeypatch.setattr(gdoc, "WRITE_S", 0.05)
    fake = {"calls": [], "text": "", "fail": False}

    def call(method, url, body=None, headers=None):
        payload = json.loads(body) if body and not url.endswith("/token") else None
        fake["calls"].append((method, url, payload))
        if url.endswith("/token"):
            return {"access_token": "t", "expires_in": 3600}
        if fake["fail"]:
            raise gdoc.DocError("Google 403: The caller does not have permission")
        if url.startswith(gdoc.DRIVE) and "permissions" not in url:
            return {"id": DOC}
        if method == "GET":
            return {"title": "x", "body": {"content": [{"endIndex": 1}, {"endIndex": 2 + len(fake["text"])}]}}
        for r in (payload or {}).get("requests", []):
            if "insertText" in r:
                fake["text"] += r["insertText"]["text"]
        return {}

    monkeypatch.setattr(gdoc, "_call", call)
    return fake


def final(*words):
    return {"type": "final", "words": [{"word": w, "start_s": 0, "end_s": 0, "probability": 1} for w in words]}


def wait_for(check, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline and not check():
        time.sleep(0.05)
    assert check()


def test_doc_links_are_parsed():
    from ai_trns.api.gdoc import DocError, doc_id
    assert doc_id(URL) == DOC
    assert doc_id(f"https://docs.google.com/document/u/0/d/{DOC}/edit?tab=t.0") == DOC
    with pytest.raises(DocError):
        doc_id("https://example.org/doc")


def test_the_server_creates_a_doc_with_the_job_and_writes_into_it(client, google):
    job_id = client.post("/live", headers=H, data={"url": STREAM}).json()["id"]
    wait_for(lambda: job_id in client.app.state.live.writers)
    create = next(c for c in google["calls"] if c[1].startswith("https://www.googleapis.com/drive/v3/files?"))
    assert create[2]["parents"] == ["folder1"] and create[2]["name"].startswith("תמלול חי")
    assert any(c[1].endswith("/permissions?supportsAllDrives=true") and c[2] == {"type": "anyone", "role": "writer"}
               for c in google["calls"])
    # joining the broadcast does not make another doc
    assert client.post("/live", headers=H, data={"url": STREAM}).json()["id"] == job_id
    assert sum(1 for c in google["calls"] if c[1].startswith("https://www.googleapis.com/drive/v3/files?")) == 1

    client.post("/worker/lease", headers=W, json={"worker": "w2", "langs": [], "engines": ["live"]})
    client.post(f"/worker/jobs/{job_id}/events", headers=W, json={"worker": "w2", "events": [final("שלום", "עולם.")]})
    client.post(f"/worker/jobs/{job_id}/events", headers=W, json={"worker": "w2", "events": [final("מה", "נשמע")]})
    wait_for(lambda: google["text"] == "שלום עולם. מה נשמע")
    client.delete(f"/jobs/{job_id}", headers=H)
    docs = [e for e in sse(client.get(f"/live/{job_id}/events", headers=H).text) if e["type"] == "docs"]
    assert docs[-1]["server"] == {"url": URL, "title": docs[-1]["server"]["title"]}


def test_a_doc_that_cannot_be_created_is_reported(client, google):
    google["fail"] = True
    job_id = client.post("/live", headers=H, data={"url": STREAM}).json()["id"]
    wait_for(lambda: "error" in client.app.state.live.server_docs.get(job_id, {}))
    assert job_id not in client.app.state.live.writers


def test_without_credentials_there_is_no_server_doc(client):
    job_id = client.post("/live", headers=H, data={"url": STREAM}).json()["id"]
    time.sleep(0.1)
    assert job_id not in client.app.state.live.server_docs


def test_listeners_docs_are_listed_with_who_writes_them(client, oidc):  # noqa: F811
    from ai_trns.api import live
    from tests.test_auth import token
    u1, u2 = token("u1"), token("u2")
    job_id = client.post("/live", headers=u1, data={"url": STREAM}).json()["id"]
    assert client.put(f"/live/{job_id}/doc", headers=u2, data={"url": URL, "title": "שלי"}).status_code == 200
    assert client.put(f"/live/{job_id}/doc", headers=u2, data={"url": "https://example.org/x"}).status_code == 422
    state = client.app.state.live
    assert list(state.browser_docs[job_id].values())[0]["by"] == "u2@example.org"
    # only its writer takes it off the list
    client.delete(f"/live/{job_id}/doc", headers=u1, params={"url": URL})
    assert URL in state.browser_docs[job_id]
    client.delete(f"/live/{job_id}/doc", headers=u2, params={"url": URL})
    assert URL not in state.browser_docs[job_id]
    # a page that stopped saying it writes is dropped
    client.put(f"/live/{job_id}/doc", headers=u2, data={"url": URL, "title": "שלי"})
    state.browser_docs[job_id][URL]["seen"] -= live.BROWSER_DOC_S + 1
    state.cleanup(client.app.state.store)
    assert URL not in state.browser_docs[job_id]
    events = [e for e in state.hubs[job_id].events if e["type"] == "docs"]
    assert [len(e["browser"]) for e in events] == [1, 0, 1, 0]


def test_authorized_user_credentials_refresh_their_token(tmp_path, monkeypatch):
    from ai_trns.api import gdoc
    path = tmp_path / "user.json"
    path.write_text(json.dumps({"type": "authorized_user", "client_id": "c", "client_secret": "s",
                                "refresh_token": "r"}))
    sent = []
    monkeypatch.setattr(gdoc, "_call", lambda m, u, b=None, h=None: sent.append(b) or {"access_token": "t"})
    assert gdoc.Account(str(path)).access_token() == "t"
    assert b"grant_type=refresh_token" in sent[0] and b"refresh_token=r" in sent[0]
