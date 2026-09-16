#!/usr/bin/env bash
set -euo pipefail

if [[ "${HUNYFORGE_DEMO:-0}" == "1" ]]; then
  exec python3.10 -m uvicorn backend.main:app --host 0.0.0.0 --port 8081
fi

if [[ ! -f "${HUNYUAN_MODEL_PATH}/hunyuan3d-dit-v2-1/model.fp16.ckpt" ]]; then
  echo "Hunyuan weights not found at ${HUNYUAN_MODEL_PATH}" >&2
  exit 2
fi

mkdir -p /opt/hunyuan/gradio_cache /data/runtime-logs

api_fifo="$(mktemp -u /tmp/hunyforge-api.XXXXXX)"
worker_fifo="$(mktemp -u /tmp/hunyforge-worker.XXXXXX)"
mkfifo "$api_fifo" "$worker_fifo"

tee -a /data/runtime-logs/api.log < "$api_fifo" &
api_tee=$!
tee -a /data/runtime-logs/worker.log < "$worker_fifo" &
worker_tee=$!

echo "Starting HunyForge API on :8081 and Hunyuan3D worker on :8082 (models stay unloaded until first request)..."

(cd /app && exec python3.10 -m uvicorn backend.main:app --host 0.0.0.0 --port 8081 > "$api_fifo" 2>&1) &
api_pid=$!
(cd /opt/hunyuan && exec python3.10 -m uvicorn backend.hunyuan_worker:app --host 127.0.0.1 --port 8082 > "$worker_fifo" 2>&1) &
worker_pid=$!

shutdown() {
  trap - TERM INT EXIT
  for pid in "$api_pid" "$worker_pid" "$api_tee" "$worker_tee"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap 'shutdown' TERM INT EXIT

set +e
wait -n "$api_pid" "$worker_pid"
status=$?
set -e

shutdown
wait "$api_pid" "$worker_pid" 2>/dev/null || true
exit "$status"
