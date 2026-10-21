import os
import socket


def _list(name: str) -> list:
    return [x.strip() for x in os.environ.get(name, "").split(",") if x.strip()]


API_URL = os.environ.get("AI_TRNS_API_URL", "http://127.0.0.1:8766").rstrip("/")
WORKER_KEY = os.environ.get("AI_TRNS_WORKER_KEY", "")
WORKER_ID = os.environ.get("AI_TRNS_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")

MODELS_DIR = os.environ.get("AI_TRNS_MODELS_DIR", "./models")
# Engines this worker runs: transcribe (faster-whisper), diarize (pyannote + MMS, requirements-diarize.txt)
ENGINES = _list("AI_TRNS_WORKER_ENGINES") or ["transcribe"]
# diarize: built-in voice samples, <dir>/<Name>/*.mp3 (or <dir>/<Name>.mp3); the pyannote pipeline and
# a HuggingFace token with its license accepted
SPEAKERS_DIR = os.environ.get("AI_TRNS_SPEAKERS_DIR") or os.path.join(MODELS_DIR, "speakers")
DIARIZE_PIPELINE = os.environ.get("AI_TRNS_DIARIZE_PIPELINE", "pyannote/speaker-diarization-community-1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
# Transcribe through whisper_trainer's job service at this URL (ai_trns.stt_service) instead of a
# model here, e.g. http://10.66.1.53:8765. It takes media URLs only, and is Hebrew only
STT_URL = os.environ.get("AI_TRNS_STT_URL", "").rstrip("/")
STT_TIMEOUT_S = float(os.environ.get("AI_TRNS_STT_TIMEOUT_S", str(3 * 3600)))
# Languages this worker takes jobs for; default: every models/<lang> present ("auto" needs models/en),
# or he with AI_TRNS_STT_URL
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
