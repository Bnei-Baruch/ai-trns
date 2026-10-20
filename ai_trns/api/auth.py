"""
Client authentication: Keycloak access tokens for users, named API keys for services.

    Authorization: Bearer <Keycloak access token>    -> user  (sub, realm roles)
    X-API-Key: <key>  or  Authorization: Bearer <key> -> service (key name)

A user sees only the jobs they created, unless they have ADMIN_ROLE; a service sees all.
"""
import logging
import secrets
import threading
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Header, HTTPException

from . import config

log = logging.getLogger("ai-trns.auth")


@dataclass(frozen=True)
class Principal:
    owner: str          # stored on jobs: "user:<sub>" or "key:<name>"
    name: str
    sees_all: bool


OPEN = Principal(owner="open", name="open", sees_all=True)

_jwks = None
_jwks_lock = threading.Lock()


def _signing_key(token: str):
    """The issuer's public key for this token; keys are fetched once and refreshed on unknown kid"""
    global _jwks
    with _jwks_lock:
        if _jwks is None:
            _jwks = jwt.PyJWKClient(f"{config.OIDC_ISSUER}/protocol/openid-connect/certs",
                                    cache_keys=True, lifespan=3600, timeout=10)
    return _jwks.get_signing_key_from_jwt(token).key


def _user(token: str) -> Principal:
    try:
        claims = jwt.decode(token, _signing_key(token), algorithms=["RS256", "RS384", "RS512", "ES256", "PS256"],
                            issuer=config.OIDC_ISSUER, options={"verify_aud": False, "require": ["exp", "sub"]},
                            leeway=30)
    except jwt.PyJWKClientError as e:
        log.warning("cannot get Keycloak keys: %s", e)
        raise HTTPException(503, "cannot verify the token now")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"invalid token: {e}")
    if config.OIDC_CLIENTS and claims.get("azp") not in config.OIDC_CLIENTS:
        raise HTTPException(403, f"tokens of client {claims.get('azp')!r} are not accepted")
    roles = set(claims.get("realm_access", {}).get("roles", []))
    if config.OIDC_ROLE and config.OIDC_ROLE not in roles:
        raise HTTPException(403, f"role {config.OIDC_ROLE!r} required")
    name = claims.get("email") or claims.get("preferred_username") or claims["sub"]
    return Principal(owner=f"user:{claims['sub']}", name=name,
                     sees_all=bool(config.ADMIN_ROLE) and config.ADMIN_ROLE in roles)


def _service(key: str) -> Optional[Principal]:
    for name, expected in config.API_KEYS.items():
        if secrets.compare_digest(key, expected):
            return Principal(owner=f"key:{name}", name=name, sees_all=True)
    return None


def principal(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)) -> Principal:
    """FastAPI dependency: who is calling"""
    if not config.OIDC_ISSUER and not config.API_KEYS:
        return OPEN
    bearer = authorization[7:].strip() if authorization and authorization.lower().startswith("bearer ") else None
    for key in (x_api_key, bearer):
        if key and (service := _service(key)):
            return service
    if bearer and config.OIDC_ISSUER and bearer.count(".") == 2:
        return _user(bearer)
    raise HTTPException(401, "a Keycloak token or an API key is required")


def can_see(who: Principal, job: dict) -> bool:
    return who.sees_all or job.get("owner") == who.owner
