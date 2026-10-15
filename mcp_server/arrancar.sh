#!/usr/bin/env bash
# Lanzador del servidor MCP con rutas absolutas, para registrarlo en todas las sesiones
# de Claude Code (ámbito de usuario) y no solo dentro de la carpeta del proyecto.
set -euo pipefail
PROYECTO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROYECTO"
export DETECTOR_URL="${DETECTOR_URL:-http://127.0.0.1:8000}"
exec "$PROYECTO/.venv/bin/python" -m mcp_server.server
