#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ] || [ ! -d frontend/node_modules ]; then
  echo 'Install dependencies using the README before starting.' >&2
  exit 1
fi
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 &
backend_pid=$!
cleanup() { kill "$backend_pid" "${frontend_pid:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
(cd frontend && exec node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort) &
frontend_pid=$!
wait -n "$backend_pid" "$frontend_pid"
