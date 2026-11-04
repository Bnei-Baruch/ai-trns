"""
Paragraphing service: the mdb-lora "paragraphing" adapter (Qwen2.5-3B 4-bit + LoRA) over HTTP, for the diarize
worker. Runs on the GPU host that has mdb-lora (`stt`), in its conda env (torch, transformers, peft); this
module itself is stdlib only and imports mdb-lora's own code, so the split is the one the model learned.

    POST /paragraphs  {"text": "..."}  header X-API-Key
      -> {"sentences": [[start, end], ...],   # char spans in the whitespace-collapsed text ("flat")
          "breaks": [n, ...],                 # sentence numbers that start a paragraph (0 never)
          "chunks": int, "seconds": float, "unparsed_chunks": int, "adapter": str}
    GET /health -> {"status": "ok", "loaded": bool}

One request at a time; the model loads on the first request and is unloaded after
PARAGRAPHS_IDLE_UNLOAD_S idle seconds, so the GPU is free between lessons.
Settings: MDB_LORA_DIR (/home/mdb-lora), PARAGRAPHS_ADAPTER (paragraphing: the adapter's directory in
mdb-lora/adapters, its config.yaml names the weights; paragraphing_asr -- the one fine-tuned on ASR text),
PARAGRAPHS_KEY (required), PARAGRAPHS_HOST / PARAGRAPHS_PORT
(0.0.0.0:8766), PARAGRAPHS_IDLE_UNLOAD_S (300), PARAGRAPHS_MAX_NEW_TOKENS (2048), PARAGRAPHS_MAX_CHARS (2000000).
Run: python -m ai_trns.paragraphs.service
"""
import gc
import hmac
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("ai-trns.paragraphs")

MDB_LORA_DIR = os.environ.get("MDB_LORA_DIR", "/home/mdb-lora")
KEY = os.environ.get("PARAGRAPHS_KEY", "")
HOST = os.environ.get("PARAGRAPHS_HOST", "0.0.0.0")
PORT = int(os.environ.get("PARAGRAPHS_PORT", "8766"))
IDLE_UNLOAD_S = float(os.environ.get("PARAGRAPHS_IDLE_UNLOAD_S", "300"))
MAX_NEW_TOKENS = int(os.environ.get("PARAGRAPHS_MAX_NEW_TOKENS", "2048"))
MAX_CHARS = int(os.environ.get("PARAGRAPHS_MAX_CHARS", "2000000"))
ADAPTER = os.environ.get("PARAGRAPHS_ADAPTER", "paragraphing")
ADAPTER_DIR = os.path.join(MDB_LORA_DIR, "adapters", ADAPTER)
# the sentence split and the answer format (segment.py) are paragraphing's for every adapter
SEGMENT_DIR = os.path.join(MDB_LORA_DIR, "adapters", "paragraphing")


def _mdb_lora_on_path():
    for path in (os.path.join(MDB_LORA_DIR, "common"), SEGMENT_DIR):
        if path not in sys.path:
            sys.path.insert(0, path)


class Model:
    """The loaded generator, or None; a lock makes requests take turns"""

    def __init__(self):
        self.lock = threading.Lock()
        self.gen = None
        self.cfg = None
        self.last_used = 0.0

    def _load(self):
        from config import load_config
        from infer import Generator, attach_weights, load_base_model, read_adapters
        config_path = os.path.join(ADAPTER_DIR, "config.yaml")
        cfg = load_config(config_path)
        adapters = read_adapters([config_path], cfg)
        if not adapters[0].has_weights:
            raise RuntimeError(f"no LoRA weights in {adapters[0].weights_path}")
        t0 = time.time()
        model, tokenizer = load_base_model(cfg)
        model = attach_weights(model, adapters)
        model.eval()
        gen = Generator(model, tokenizer, adapters, MAX_NEW_TOKENS)
        gen.set_adapter(adapters[0].name)
        gen.set_lora(True)
        self.gen, self.cfg = gen, cfg
        log.info("model loaded in %.1f s (adapter %s, %s)", time.time() - t0, ADAPTER, adapters[0].weights_path)

    def split(self, text: str) -> dict:
        _mdb_lora_on_path()
        from segment import build_chunks, flatten, parse_answer
        with self.lock:
            if self.gen is None:
                self._load()
            t0 = time.time()
            flat = flatten(text)
            sentences, chunks = build_chunks(flat, self.gen.tokenizer, self.cfg["task"]["chunk_max_tokens"])
            breaks, unparsed = set(), 0
            for ids, chunk in chunks:
                ids_found = parse_answer(self.gen.generate(chunk))
                if ids_found is None:
                    unparsed += 1
                    continue
                # a chunk's first sentence is never in its answer; numbers are global
                breaks.update(i for i in ids_found if i in ids and i != ids.start)
            self.last_used = time.time()
            seconds = round(time.time() - t0, 1)
            log.info("%d chars, %d sentences, %d chunks -> %d breaks in %.1f s", len(flat), len(sentences),
                     len(chunks), len(breaks), seconds)
            return {"sentences": sentences, "breaks": sorted(breaks), "chunks": len(chunks), "adapter": ADAPTER,
                    "unparsed_chunks": unparsed, "seconds": seconds}

    def unload_if_idle(self):
        with self.lock:
            if self.gen is not None and time.time() - self.last_used > IDLE_UNLOAD_S:
                self.gen = None
                gc.collect()
                if "torch" in sys.modules:
                    sys.modules["torch"].cuda.empty_cache()
                log.info("model unloaded after %.0f s idle", IDLE_UNLOAD_S)


MODEL = Model()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: dict):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok", "loaded": MODEL.gen is not None, "adapter": ADAPTER})
        self._send(404, {"detail": "not found"})

    def do_POST(self):
        if self.path != "/paragraphs":
            return self._send(404, {"detail": "not found"})
        if not hmac.compare_digest(self.headers.get("X-API-Key", ""), KEY):
            return self._send(401, {"detail": "bad key"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_CHARS * 4:
                return self._send(413, {"detail": "text too long"})
            text = json.loads(self.rfile.read(length))["text"]
            if not isinstance(text, str) or len(text) > MAX_CHARS:
                return self._send(400, {"detail": "text must be a string up to PARAGRAPHS_MAX_CHARS"})
        except (ValueError, KeyError, TypeError):
            return self._send(400, {"detail": 'expected JSON {"text": "..."}'})
        try:
            self._send(200, MODEL.split(text))
        except Exception as e:
            log.exception("paragraphs failed")
            self._send(500, {"detail": f"{type(e).__name__}: {e}"})


def main():
    if not KEY:
        raise SystemExit("PARAGRAPHS_KEY is not set")

    def unloader():
        while True:
            time.sleep(30)
            MODEL.unload_if_idle()

    threading.Thread(target=unloader, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("paragraphs service on %s:%d, mdb-lora %s, adapter %s", HOST, PORT, MDB_LORA_DIR, ADAPTER)
    server.serve_forever()


if __name__ == "__main__":
    main()
