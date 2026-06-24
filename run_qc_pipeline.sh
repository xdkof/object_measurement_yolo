#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIDEO_PATH="${1:-videos/steel_plates_cutting.mp4}"
RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/belt_stream}"
PYTHON_BIN="${PYTHON_BIN:-}"
MEDIAMTX_BIN="${MEDIAMTX_BIN:-}"
MEDIAMTX_CONFIG="${MEDIAMTX_CONFIG:-$SCRIPT_DIR/mediamtx_qc.yml}"
STREAM_PID=""
MEDIAMTX_PID=""
CLEANUP_DONE=0

rtsp_host="localhost"
rtsp_port="8554"
if [[ "$RTSP_URL" =~ ^rtsp://([^/:]+):([0-9]+) ]]; then
  rtsp_host="${BASH_REMATCH[1]}"
  rtsp_port="${BASH_REMATCH[2]}"
elif [[ "$RTSP_URL" =~ ^rtsp://([^/:/]+) ]]; then
  rtsp_host="${BASH_REMATCH[1]}"
  rtsp_port="554"
fi

cleanup() {
  if [[ "$CLEANUP_DONE" -eq 1 ]]; then
    return
  fi
  CLEANUP_DONE=1

  echo
  echo "[QC] Stopping background stream services..."
  if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
    kill "$STREAM_PID" 2>/dev/null || true
    wait "$STREAM_PID" 2>/dev/null || true
  fi
  if [[ -n "$MEDIAMTX_PID" ]] && kill -0 "$MEDIAMTX_PID" 2>/dev/null; then
    kill "$MEDIAMTX_PID" 2>/dev/null || true
    wait "$MEDIAMTX_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

port_open() {
  nc -z "$rtsp_host" "$rtsp_port" >/dev/null 2>&1
}

wait_for_port() {
  local label="$1"
  local seconds="${2:-30}"

  for _ in $(seq 1 "$seconds"); do
    if port_open; then
      return 0
    fi
    sleep 1
  done

  echo "[QC] Timed out waiting for $label at $rtsp_host:$rtsp_port" >&2
  return 1
}

wait_for_rtsp_stream() {
  local seconds="${1:-45}"

  for _ in $(seq 1 "$seconds"); do
    if ffprobe \
      -v error \
      -rtsp_transport tcp \
      -select_streams v:0 \
      -show_entries stream=codec_name \
      -of csv=p=0 \
      "$RTSP_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "[QC] Timed out waiting for stream at $RTSP_URL" >&2
  return 1
}

resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "$PYTHON_BIN"
  elif [[ -x "$SCRIPT_DIR/yolo_env/bin/python" ]]; then
    echo "$SCRIPT_DIR/yolo_env/bin/python"
  elif [[ -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    echo "$SCRIPT_DIR/venv/bin/python"
  else
    echo "python3"
  fi
}

resolve_mediamtx() {
  if [[ -n "$MEDIAMTX_BIN" ]]; then
    echo "$MEDIAMTX_BIN"
  elif command -v mediamtx >/dev/null 2>&1; then
    command -v mediamtx
  elif [[ -x "$SCRIPT_DIR/mediamtx" ]]; then
    echo "$SCRIPT_DIR/mediamtx"
  elif [[ -x "$SCRIPT_DIR/mediamtx/mediamtx" ]]; then
    echo "$SCRIPT_DIR/mediamtx/mediamtx"
  else
    local candidate
    while IFS= read -r candidate; do
      if [[ -x "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done < <(find "$HOME/Downloads" -maxdepth 3 -type f -name mediamtx 2>/dev/null || true)
  fi
}

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[QC] Video not found: $VIDEO_PATH" >&2
  exit 1
fi

if [[ ! -x "$SCRIPT_DIR/stream_to_mediamtx_high_quality.sh" ]]; then
  chmod +x "$SCRIPT_DIR/stream_to_mediamtx_high_quality.sh"
fi

if [[ ! -f "$MEDIAMTX_CONFIG" ]]; then
  echo "[QC] MediaMTX config not found: $MEDIAMTX_CONFIG" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python)"

if port_open; then
  echo "[QC] MediaMTX already appears to be running on $rtsp_host:$rtsp_port"
  echo "[QC] If publishing fails with \"path 'belt_stream' is not configured\", stop that MediaMTX and run this script again."
else
  MEDIAMTX_BIN="$(resolve_mediamtx)"
  if [[ -z "$MEDIAMTX_BIN" || ! -x "$MEDIAMTX_BIN" ]]; then
    echo "[QC] Could not find MediaMTX." >&2
    echo "[QC] Start MediaMTX manually, or run this with:" >&2
    echo "[QC]   MEDIAMTX_BIN=/full/path/to/mediamtx ./run_qc_pipeline.sh" >&2
    exit 1
  fi

  echo "[QC] Starting MediaMTX: $MEDIAMTX_BIN $MEDIAMTX_CONFIG"
  "$MEDIAMTX_BIN" "$MEDIAMTX_CONFIG" &
  MEDIAMTX_PID="$!"
  wait_for_port "MediaMTX"
fi

echo "[QC] Starting RTSP publisher from $VIDEO_PATH"
RTSP_URL="$RTSP_URL" "$SCRIPT_DIR/stream_to_mediamtx_high_quality.sh" "$VIDEO_PATH" &
STREAM_PID="$!"

wait_for_rtsp_stream

echo "[QC] Starting production monitor"
echo "[QC] CSV output will be written under outputs/production_runs/"
RTSP_URL="$RTSP_URL" "$PYTHON_BIN" "$SCRIPT_DIR/production.py"
