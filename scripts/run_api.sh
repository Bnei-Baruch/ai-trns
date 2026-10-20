#!/usr/bin/env bash
# API server; settings from the environment (ai-trns-api.env via systemd)
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${AI_TRNS_PYTHON:-/home/ai-trns/env/bin/python}" -m ai_trns.api
