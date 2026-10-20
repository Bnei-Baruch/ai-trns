import os
import socket


def _list(name: str) -> list:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


API_URL = os.environ.get("AI_TRNS_API_URL", "http://127.0.0.1:8766").rstrip("/")
WORKER_KEY = os.environ.get("AI_TRNS_WORKER_KEY", "")
WORKER_ID = os.environ.get("AI_TRNS_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")

MODELS_DIR = os.environ.get("AI_TRNS_MODELS_DIR", "./models")
# Languages this worker takes jobs for; default: every models/<lang> present ("auto" needs models/en)
LANGS = _list("AI_TRNS_WORKER_LANGS")
DEVICE = os.environ.get("AI_TRNS_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("AI_TRNS_COMPUTE_TYPE", "float16")
# Batch size of the served (batched) profile; 8 fits an 8 GB GPU with room to spare
BATCH_SIZE = int(os.environ.get("AI_TRNS_BATCH_SIZE", "8"))
# The GPU is shared with other services: unload models after this many idle seconds (loading takes ~1 s)
IDLE_UNLOAD_S = float(os.environ.get("AI_TRNS_IDLE_UNLOAD_S", "300"))
# How often to ask for a job while the queue is empty
POLL_S = float(os.environ.get("AI_TRNS_POLL_S", "3"))
WORK_DIR = os.environ.get("AI_TRNS_WORK_DIR", "./work")
