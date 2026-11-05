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
# Uploaded audio for jobs of the transcribe engine (transcribe, tadiff, docdiff). Off while workers
# Uploaded files for transcription: off where no worker has a local model for them (whisper_trainer's service,
# AI_TRNS_STT_URL, takes media URLs only; a worker with it runs uploads on its own models/<lang>)
TRANSCRIBE_UPLOADS = os.environ.get("AI_TRNS_TRANSCRIBE_UPLOADS", "0").lower() in ("1", "true", "yes")

# A job is given to workers at most this many times (lost workers, expired leases)
MAX_ATTEMPTS = int(os.environ.get("AI_TRNS_MAX_ATTEMPTS", "3"))
JOB_TTL_HOURS = float(os.environ.get("AI_TRNS_JOB_TTL_HOURS", "72"))

# Lesson sources the server fetches for /diarize/* `uid` (ai_trns.sources): kmedia / CDN timeout per request
# (seconds), and a cache of the extracted texts by file id (default <jobs dir>/sources-cache), its size and age limits
SOURCES_TIMEOUT = float(os.environ.get("AI_TRNS_SOURCES_TIMEOUT", "15"))
SOURCES_CACHE_DIR = os.environ.get("AI_TRNS_SOURCES_CACHE_DIR", "")
SOURCES_CACHE_MB = float(os.environ.get("AI_TRNS_SOURCES_CACHE_MB", "50"))
SOURCES_CACHE_DAYS = float(os.environ.get("AI_TRNS_SOURCES_CACHE_DAYS", "30"))

# Live text into a Google Doc the server creates per live job (ai_trns.api.gdoc): the account's JSON
# (a service account's key or an authorized_user); empty = off. The folder to create docs in (a service
# account needs one in a shared drive); the role anyone with the link gets ("reader", "commenter",
# "writer"; empty: only who the folder is shared with)
GOOGLE_CREDENTIALS = os.environ.get("AI_TRNS_GOOGLE_CREDENTIALS", "")
GOOGLE_FOLDER = os.environ.get("AI_TRNS_GOOGLE_FOLDER", "")
GOOGLE_DOC_ROLE = os.environ.get("AI_TRNS_GOOGLE_DOC_ROLE", "")

# Human review of STT (ai_trns.api.review): sets, lessons and results as JSON files
REVIEW_DIR = os.environ.get("AI_TRNS_REVIEW_DIR", "./review")
# The model reviewers are told they review; default: the model of the newest transcribed lesson
REVIEW_MODEL = os.environ.get("AI_TRNS_REVIEW_MODEL", "")
