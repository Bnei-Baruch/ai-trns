# Audio Transcription with Whisper

faster-whisper + ffmpeg: transcription to text and SRT subtitles (with advanced splitting and RTL fixes for Hebrew).

Two ways to run, on the same core (`ai_trns/`):

- **CLI** (`whisper_fullfile_cli.py`) — one file per run, launched by [exec-api](https://github.com/Bnei-Baruch/exec-api). The current production setup.
- **REST server** (`python -m ai_trns.server`) — the model stays in memory, jobs go through a queue.

```
ai_trns/
  transcriber.py   Transcriber: model + profiles, one Whisper pass for all formats, progress/cancel
  subtitles.py     text / SRT rendering (standard, advanced ToSubs, RTL fixes)
  audio.py         ffprobe / ffmpeg → 16 kHz mono wav (with -protocol_whitelist)
  server/          FastAPI app, SQLite job store, GPU worker
whisper_fullfile_cli.py
```

Models in CTranslate2 format: `models/<lang>/` (`en`, `he`, `ru`; `auto` uses `models/en`, so it must be multilingual).

## Installation

```bash
pip install -r requirements.txt          # CLI
pip install -r requirements-server.txt   # REST server
```

## CLI (exec-api)

```bash
python whisper_fullfile_cli.py --input audio.mp3 --srt --lang he --fix-rtl
python whisper_fullfile_cli.py --input https://host/file.mp4 --srt --txt --lang ru
```

| Option | Default | |
|---|---|---|
| `--input` | — | file path or URL |
| `--txt`, `--srt` | — | output formats; both together use one Whisper pass |
| `--lang` | `he` | `en`, `he`, `ru`, `auto` |
| `--impl` | `2` | SRT: `1` standard (split by duration), `2` advanced (pauses + punctuation) |
| `--fix-rtl` | off | RTL markers for Hebrew (with `--impl 2`) |
| `--profile` | `stable` | `stable` or `fast` |
| `--out-dir` | `.` | where `output.txt` / `output.srt` / `output.wav` go |
| `--report-file` | `report_python.log` | progress file; for exec-api it must be `report_<service ID>.log` |
| `--models-dir` | `./models` | |
| `--nginx-dir` | `/usr/share/nginx/html/ai/` | copy results there; `""` to disable |
| `--keep-wav` | off | keep `output.wav` |
| `--rtf` | `40` | only for the "expected time" estimate |

The report file holds the last status line: `[FFmpeg] [###…] NN%`, `[Whisper] [###…] NN% - N sec`
(ai-tools reads the `NN%`), then `[✓] DONE` or `[✗] ERROR: …`. Exit code: 0 ok, 1 error, 143 stopped (SIGTERM).

exec-api service entry (arguments come only from `Args` in the config, `start` takes none):

```json
{"ID": "python", "Name": "python3",
 "Args": ["whisper_fullfile_cli.py", "--input", "/path/to/file.mp3", "--srt", "--lang", "he", "--fix-rtl"]}
```

## REST server

```bash
AI_TRNS_API_KEY=... AI_TRNS_HOST=0.0.0.0 python -m ai_trns.server
```

One process, one worker thread: jobs run one after another on the GPU. Job state is in SQLite;
after a restart queued jobs continue, a job that was running is marked `error`.

| Method | Path | |
|---|---|---|
| `POST` | `/jobs` | multipart: `file` **or** `url`, `lang`, `format` (`srt`, `txt` or `srt,txt`), `impl`, `profile`, `fix_rtl` → `202` + job |
| `GET` | `/jobs/{id}` | `status` (`queued`, `running`, `done`, `error`, `cancelled`), `stage`, `progress` 0..1, `queue_position`, `error` |
| `GET` | `/jobs/{id}/result?format=srt` | result file (default: first format of the job) |
| `DELETE` | `/jobs/{id}` | cancel (queued at once, running between segments) |
| `GET` | `/health` | worker state, queue length, loaded models (no auth) |

Auth: `X-API-Key: <key>` or `Authorization: Bearer <key>`.

```bash
curl -H "X-API-Key: $KEY" -F file=@lecture.mp3 -F lang=he -F format=srt,txt -F fix_rtl=true http://localhost:8000/jobs
curl -H "X-API-Key: $KEY" http://localhost:8000/jobs/<id>
curl -H "X-API-Key: $KEY" -o lecture.srt "http://localhost:8000/jobs/<id>/result?format=srt"
```

Environment:

| Variable | Default | |
|---|---|---|
| `AI_TRNS_HOST` / `AI_TRNS_PORT` | `127.0.0.1` / `8000` | |
| `AI_TRNS_API_KEY` | empty | empty = **no auth** (a warning is logged) |
| `AI_TRNS_MODELS_DIR` | `./models` | |
| `AI_TRNS_JOBS_DIR` | `./jobs` | `jobs/<id>/` per job + `jobs.sqlite` |
| `AI_TRNS_DB` | `$AI_TRNS_JOBS_DIR/jobs.sqlite` | |
| `AI_TRNS_DEVICE` / `AI_TRNS_COMPUTE_TYPE` | `cuda` / `float16` | |
| `AI_TRNS_MAX_MODELS` | `1` | models kept in memory, LRU |
| `AI_TRNS_PRELOAD_LANGS` | empty | e.g. `he` — load at startup |
| `AI_TRNS_ALLOWED_URL_HOSTS` | empty | hosts allowed for `url`; empty = URL input disabled |
| `AI_TRNS_MAX_UPLOAD_MB` | `2048` | |
| `AI_TRNS_JOB_TTL_HOURS` | `72` | finished jobs and their files are deleted after this |
| `AI_TRNS_CORS_ORIGINS` | empty | comma-separated; empty = no CORS headers |

## SRT implementations

- `--impl 1` / `impl=1` — split each Whisper segment by duration (≤10 s), words spread evenly.
- `--impl 2` / `impl=2` — word timestamps + `ToSubs` (from mdb-ai): split by pauses, punctuation, line length (≤60 chars, two lines), dashes in Hebrew.
- `fix_rtl` — RTL embedding and LRM/RLM marks around punctuation, numbers and Latin words (Hebrew only).

## Quality profiles

- `stable` (default) — `beam_size=8`, no VAD, temperature fallback
- `fast` — `beam_size=3`, VAD filter
