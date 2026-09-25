#!/usr/bin/env bash
# App web local del detector. Solo escucha en 127.0.0.1: nadie más en la red puede entrar.
set -euo pipefail
cd "$(dirname "$0")"
# Rutas locales: ajusta DETECTOR_IA_HOME y HF_HOME a tu máquina (ver README).
export DETECTOR_IA_HOME="${DETECTOR_IA_HOME:-$HOME/detector-ia-datos}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
exec .venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
