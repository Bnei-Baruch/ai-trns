import os


def _list(name: str) -> list:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


JOBS_DIR = os.environ.get("AI_TRNS_JOBS_DIR", "./jobs")
# Default: <jobs dir>/jobs.sqlite
DB_PATH = os.environ.get("AI_TRNS_DB", "")

# Empty = no auth (only for local / closed network)
API_KEY = os.environ.get("AI_TRNS_API_KEY", "")
# Key GPU workers use for /worker/*; required, workers are never open
WORKER_KEY = os.environ.get("AI_TRNS_WORKER_KEY", "")
CORS_ORIGINS = _list("AI_TRNS_CORS_ORIGINS")

MAX_UPLOAD_MB = int(os.environ.get("AI_TRNS_MAX_UPLOAD_MB", "2048"))
# Hosts allowed for `url` input; empty = URL input disabled
ALLOWED_URL_HOSTS = _list("AI_TRNS_ALLOWED_URL_HOSTS")

# A job is given to workers at most this many times (lost workers, expired leases)
MAX_ATTEMPTS = int(os.environ.get("AI_TRNS_MAX_ATTEMPTS", "3"))
JOB_TTL_HOURS = float(os.environ.get("AI_TRNS_JOB_TTL_HOURS", "72"))
