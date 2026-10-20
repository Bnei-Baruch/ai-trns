import os


def _list(name: str) -> list:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


JOBS_DIR = os.environ.get("AI_TRNS_JOBS_DIR", "./jobs")
# Default: <jobs dir>/jobs.sqlite
DB_PATH = os.environ.get("AI_TRNS_DB", "")

# Clients authenticate with a Keycloak token (users) or a named API key (services).
# With neither configured the API is open (only for local / closed network).
#
# Keycloak: tokens are checked against the issuer's keys; users must have OIDC_ROLE (realm role)
# and come through one of OIDC_CLIENTS (the token's azp); empty = any
OIDC_ISSUER = os.environ.get("AI_TRNS_OIDC_ISSUER", "").rstrip("/")
OIDC_CLIENTS = _list("AI_TRNS_OIDC_CLIENTS")
OIDC_ROLE = os.environ.get("AI_TRNS_OIDC_ROLE", "")
# Users with this realm role see every job, not only their own
ADMIN_ROLE = os.environ.get("AI_TRNS_ADMIN_ROLE", "")
# Services: "name:key,name:key"; a service sees every job. AI_TRNS_API_KEY is one more key named "default"
API_KEYS = dict(item.split(":", 1) for item in _list("AI_TRNS_API_KEYS"))
if os.environ.get("AI_TRNS_API_KEY"):
    API_KEYS["default"] = os.environ["AI_TRNS_API_KEY"]
# Key GPU workers use for /worker/*; required, workers are never open
WORKER_KEY = os.environ.get("AI_TRNS_WORKER_KEY", "")
CORS_ORIGINS = _list("AI_TRNS_CORS_ORIGINS")

MAX_UPLOAD_MB = int(os.environ.get("AI_TRNS_MAX_UPLOAD_MB", "2048"))
# Hosts allowed for `url` input; empty = URL input disabled
ALLOWED_URL_HOSTS = _list("AI_TRNS_ALLOWED_URL_HOSTS")

# A job is given to workers at most this many times (lost workers, expired leases)
MAX_ATTEMPTS = int(os.environ.get("AI_TRNS_MAX_ATTEMPTS", "3"))
JOB_TTL_HOURS = float(os.environ.get("AI_TRNS_JOB_TTL_HOURS", "72"))
