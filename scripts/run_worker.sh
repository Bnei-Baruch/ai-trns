#!/usr/bin/env bash
# GPU worker; settings from the environment (ai-trns-worker.env via systemd)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${AI_TRNS_PYTHON:-/home/ai-trns/env/bin/python}
# ctranslate2 needs CUDA 12 cuBLAS/cuDNN: take them from pip wheels if installed, whatever the host has
NV=$("$PY" -c "import glob, site; print(':'.join(p for sp in site.getsitepackages() for p in glob.glob(sp + '/nvidia/*/lib')))")
export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$PY" -m ai_trns.worker
