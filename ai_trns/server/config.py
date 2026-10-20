import os


def _list(name: str) -> list:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


MODELS_DIR = os.environ.get("AI_TRNS_MODELS_DIR", "./models")
JOBS_DIR = os.environ.get("AI_TRNS_JOBS_DIR", "./jobs")
DB_PATH = os.environ.get("AI_TRNS_DB", os.path.join(JOBS_DIR, "jobs.sqlite"))

DEVICE = os.environ.get("AI_TRNS_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("AI_TRNS_COMPUTE_TYPE", "float16")
# Batch size of the served (batched) profile; 8 fits an 8 GB GPU with room to spare
BATCH_SIZE = int(os.environ.get("AI_TRNS_BATCH_SIZE", "8"))
# How many models to keep in (video) memory at once
MAX_MODELS = int(os.environ.get("AI_TRNS_MAX_MODELS", "1"))
PRELOAD_LANGS = _list("AI_TRNS_PRELOAD_LANGS")

# Empty = no auth (only for local / closed network)
API_KEY = os.environ.get("AI_TRNS_API_KEY", "")
CORS_ORIGINS = _list("AI_TRNS_CORS_ORIGINS")

MAX_UPLOAD_MB = int(os.environ.get("AI_TRNS_MAX_UPLOAD_MB", "2048"))
# Hosts allowed for `url` input; empty = URL input disabled
ALLOWED_URL_HOSTS = _list("AI_TRNS_ALLOWED_URL_HOSTS")

JOB_TTL_HOURS = float(os.environ.get("AI_TRNS_JOB_TTL_HOURS", "72"))
