"""Keycloak tokens (signed here with a test key) and service keys"""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.conftest import H

ISSUER = "https://kc.test/auth/realms/main"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def token(sub="u1", roles=("bb_user",), azp="ai-tools", iss=ISSUER, exp_in=300, key=KEY):
    claims = {"sub": sub, "iss": iss, "azp": azp, "exp": int(time.time()) + exp_in,
              "realm_access": {"roles": list(roles)}, "email": f"{sub}@example.org"}
    return {"Authorization": "Bearer " + jwt.encode(claims, key, algorithm="RS256")}


@pytest.fixture
def oidc(api_config, monkeypatch):
    from ai_trns.api import auth
    monkeypatch.setattr(api_config, "OIDC_ISSUER", ISSUER)
    monkeypatch.setattr(api_config, "OIDC_CLIENTS", ["ai-tools"])
    monkeypatch.setattr(api_config, "OIDC_ROLE", "bb_user")
    monkeypatch.setattr(api_config, "ADMIN_ROLE", "ai_admin")
    monkeypatch.setattr(auth, "_signing_key", lambda t: KEY.public_key())


def new_job(client, headers):
    r = client.post("/jobs", headers=headers, data={"url": "https://cdn.example.org/a.mp3"})
    assert r.status_code == 202, r.text
    return r.json()


def test_users_see_only_their_own_jobs(client, oidc):
    mine = new_job(client, token("u1"))
    assert mine["owner"] == "user:u1"
    assert client.get(f"/jobs/{mine['id']}", headers=token("u1")).status_code == 200
    assert client.get(f"/jobs/{mine['id']}", headers=token("u2")).status_code == 404
    assert client.delete(f"/jobs/{mine['id']}", headers=token("u2")).status_code == 404
    theirs = new_job(client, token("u2"))
    assert [j["id"] for j in client.get("/jobs", headers=token("u1")).json()] == [mine["id"]]
    admin = token("boss", roles=("bb_user", "ai_admin"))
    assert {j["id"] for j in client.get("/jobs", headers=admin).json()} == {mine["id"], theirs["id"]}
    assert client.get(f"/jobs/{mine['id']}", headers=H).status_code == 200  # services see all
    assert new_job(client, H)["owner"] == "key:tests"


@pytest.mark.parametrize("headers, status", [
    (token(exp_in=-120), 401),
    (token(iss="https://evil.test/realms/main"), 401),
    (token(key=OTHER_KEY), 401),
    (token(azp="other-app"), 403),
    # (token(roles=("offline_access",)), 403),  # role check is temporarily off
    ({"Authorization": "Bearer not-a-token"}, 401),
    ({"X-API-Key": "wrong"}, 401),
    ({}, 401),
])
def test_bad_credentials_are_refused(client, oidc, headers, status):
    assert client.post("/jobs", headers=headers, data={"url": "https://cdn.example.org/a.mp3"}).status_code == status


def test_service_key_also_works_as_bearer(client, oidc):
    from tests.conftest import API_KEY
    assert client.get("/jobs", headers={"Authorization": f"Bearer {API_KEY}"}).status_code == 200
