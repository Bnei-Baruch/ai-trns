# ai-trns

Speech-to-text service: Hebrew (and other languages) audio → words with timecodes → SRT / txt.

```
clients ──HTTP──▶ API server (no GPU)                 GPU worker (one per GPU)
                   jobs (SQLite), keys, TTL   ◀─pull─  lease a job, get the audio,
                   words → srt/txt (CPU)               transcribe → words.json
```

- **API server** (`python -m ai_trns.api`) takes jobs, keeps results, renders formats. It never touches a GPU.
- **GPU worker** (`python -m ai_trns.worker`) holds the model, asks the API for work when idle and sends back a
  words document. Workers pull, so a worker needs only outbound HTTP to the API; more GPUs = more workers.
  A worker that goes silent loses its lease and the job goes back to the queue.
- **Words document** (`ai_trns/words.py`) is the boundary between them: transcript, segments, words with the
  engine's timecodes, audio duration and the model's calibrated boundary correction. SRT and txt are rendered
  from it on the API side, so a format can be rendered again later without the GPU.
- **SRT layout** (`ai_trns/cues.py`) is ported 1:1 from whisper_trainer (`timing_alignment.py` @29645c8): cues
  of ≤ 84 chars on two balanced lines of ≤ 32, ≤ 7 s, split at pauses > 2 s, sentence punctuation and the longest
  pause. `fix_rtl` adds RTL marks to Hebrew on top of it.

```
ai_trns/
  protocol.py      API ↔ worker protocol (stdlib only)
  words.py         words document (from faster-whisper segments)
  cues.py          words → cues → SRT (whisper_trainer cue policy)
  subtitles.py     words → txt / srt / json, RTL marks
  audio.py         ffprobe / ffmpeg (-protocol_whitelist)
  transcriber.py   faster-whisper model + decoding profiles → words document
  api/             FastAPI app, job store with leases
  worker/          worker loop, stdlib HTTP client
```

## Models

`models/<lang>/` in CTranslate2 format; `auto` uses `models/en` (must be multilingual). Hebrew is
[`orendar/whisper-large-v3-turbo-hebrew-kabbalah-ct2`](https://huggingface.co/orendar/whisper-large-v3-turbo-hebrew-kabbalah-ct2).
An optional `ai_trns_model.json` next to the model sets its decoding profile and timecode correction:

```json
{"name": "whisper-large-v3-turbo-hebrew-kabbalah-ct2", "profile": "served",
 "boundary_correction_s": {"start": -0.10020327299935161, "end": 0.21972081900943863}}
```

Profiles: `served` (batched VAD windows, greedy, the model card's settings), `stable` (beam 8, no VAD),
`fast` (beam 3, VAD). Without the file a model uses `stable` and no correction.

Measured with the Hebrew model, fp16, batch 8: RTX 3070 ~125× real time (105 min in ~50 s, peak 3.4 GB VRAM),
RTX 2080 ~95×.

## API

### Authorization

- **Users** send a Keycloak access token: `Authorization: Bearer <token>` (as ai-tools does with `kc.token`).
  The API checks the signature against the issuer's keys, expiry, the client (`azp` in `AI_TRNS_OIDC_CLIENTS`)
  and the realm role `AI_TRNS_OIDC_ROLE`. A user sees only their own jobs (others' are 404), except with
  `AI_TRNS_ADMIN_ROLE`.
- **Services** send a named key: `X-API-Key: <key>` or `Authorization: Bearer <key>`. Keys are listed as
  `AI_TRNS_API_KEYS=name:key,...`, so one service's key can be revoked alone. Services see all jobs.
- **Workers** use their own key (`AI_TRNS_WORKER_KEY`) on `/worker/*` only.

Each job records its `owner` (`user:<sub>` or `key:<name>`). With no issuer and no keys configured the API is open.

| Method | Path | |
|---|---|---|
| `POST` | `/jobs` | multipart: `file` **or** `url`, `lang` (`he`, `en`, `ru`, `auto`), `format` (comma-separated `srt`, `txt`, `json`), `profile`, `fix_rtl` → `202` + job |
| `GET` | `/jobs?limit=50` | the caller's jobs, newest first |
| `GET` | `/jobs/{id}` | `status` (`queued`, `running`, `done`, `error`, `cancelled`), `stage`, `progress` 0..1, `queue_position`, `worker`, `attempts`, `error` |
| `GET` | `/jobs/{id}/result?format=srt&fix_rtl=` | result; any format, also one not asked for, is rendered from the job's words |
| `DELETE` | `/jobs/{id}` | cancel (a running job stops at the worker's next heartbeat) |
| `POST` | `/render?format=srt&fix_rtl=` | words document (JSON body) → SRT / txt, no job, no GPU |
| `GET` | `/health` | queue length, workers seen in the last 5 min and their jobs (no auth) |

```bash
curl -H "X-API-Key: $KEY" -F file=@lecture.mp3 -F lang=he -F format=srt,txt http://host:8766/jobs
curl -H "X-API-Key: $KEY" http://host:8766/jobs/<id>
curl -H "X-API-Key: $KEY" -o lecture.srt "http://host:8766/jobs/<id>/result?format=srt"
```

Worker endpoints (`/worker/*`, worker key + protocol version headers) are described in `ai_trns/protocol.py`.

## Configuration

API (`ai-trns-api.env`):

| Variable | Default | |
|---|---|---|
| `AI_TRNS_HOST` / `AI_TRNS_PORT` | `127.0.0.1` / `8000` | |
| `AI_TRNS_OIDC_ISSUER` | empty | Keycloak realm URL, e.g. `https://accounts.kab.info/auth/realms/main`; empty = no user tokens |
| `AI_TRNS_OIDC_CLIENTS` | empty | accepted `azp` (Keycloak clients), e.g. `ai-tools`; empty = any |
| `AI_TRNS_OIDC_ROLE` | empty | realm role users must have, e.g. `bb_user`; empty = any user |
| `AI_TRNS_ADMIN_ROLE` | empty | realm role that sees every job |
| `AI_TRNS_API_KEYS` | empty | service keys `name:key,...`; `AI_TRNS_API_KEY` adds one named `default` |
| `AI_TRNS_WORKER_KEY` | empty | required for workers to connect |
| `AI_TRNS_JOBS_DIR` | `./jobs` | `jobs/<id>/` per job + `jobs.sqlite` |
| `AI_TRNS_ALLOWED_URL_HOSTS` | empty | hosts allowed for `url`; empty = URL input disabled |
| `AI_TRNS_MAX_UPLOAD_MB` | `2048` | |
| `AI_TRNS_MAX_ATTEMPTS` | `3` | times a job is given to workers before it fails |
| `AI_TRNS_JOB_TTL_HOURS` | `72` | finished jobs and their files are deleted after this |
| `AI_TRNS_CORS_ORIGINS` | empty | comma-separated |
| `AI_TRNS_SOURCES_TIMEOUT` | `15` | seconds per kmedia / CDN request when `/diarize/*` gets a lesson `uid` (`ai_trns.sources`) |
| `AI_TRNS_SOURCES_CACHE_DIR` | `<jobs dir>/sources-cache` | extracted source texts by CDN file id |
| `AI_TRNS_SOURCES_CACHE_MB` / `_DAYS` | `50` / `30` | cache limits: total size, days since last use |

Worker (`ai-trns-worker.env`):

| Variable | Default | |
|---|---|---|
| `AI_TRNS_API_URL` | `http://127.0.0.1:8766` | |
| `AI_TRNS_WORKER_KEY` | — | same as the API's |
| `AI_TRNS_MODELS_DIR` | `./models` | |
| `AI_TRNS_WORKER_LANGS` | models present | languages this worker takes |
| `AI_TRNS_DEVICE` / `AI_TRNS_COMPUTE_TYPE` | `cuda` / `float16` | |
| `AI_TRNS_BATCH_SIZE` | `8` | batch of the `served` profile |
| `AI_TRNS_IDLE_UNLOAD_S` | `300` | the GPU is shared: models are unloaded after this idle time |
| `AI_TRNS_WORK_DIR` | `./work` | temporary audio of the running job |
| `AI_TRNS_WORKER_ID` | `<host>-<pid>` | |

## Install

```bash
python3 -m venv /home/ai-trns/env
/home/ai-trns/env/bin/pip install -r requirements-api.txt      # API host
/home/ai-trns/env/bin/pip install -r requirements-worker.txt   # GPU host
# ctranslate2 needs CUDA 12 cuBLAS/cuDNN; if the host has another toolkit:
/home/ai-trns/env/bin/pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

systemd units, run scripts and env examples are in `scripts/` (`ai-trns-api.service`, `ai-trns-worker.service`,
run as user `ai-trns` from `/home/ai-trns`). `run_worker.sh` puts the pip CUDA libraries on `LD_LIBRARY_PATH`.
faster-whisper 1.2.1 breaks with PyAV ≥ 15; audio is always decoded with ffmpeg, so PyAV is never used.

Tests: `pytest tests` (needs ffmpeg; the worker tests run a real API server and a fake model).
