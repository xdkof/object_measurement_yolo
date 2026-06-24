#!/usr/bin/env bash
set -euo pipefail

VIDEO_PATH="${1:-videos/steel_plates_cutting.mp4}"
RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/belt_stream}"
STREAM_MODE="${STREAM_MODE:-encode}"
STREAM_FPS="${STREAM_FPS:-30}"
CRF="${CRF:-14}"
MAXRATE="${MAXRATE:-20M}"
BUFSIZE="${BUFSIZE:-40M}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Video not found: $VIDEO_PATH" >&2
  exit 1
fi

echo "Publishing $VIDEO_PATH to $RTSP_URL"
echo "Mode: $STREAM_MODE"

if [[ "$STREAM_MODE" == "copy" ]]; then
  echo "Warning: copy mode preserves the source bitstream and may keep decode artifacts."
  ffmpeg \
    -hide_banner \
    -re \
    -stream_loop -1 \
    -i "$VIDEO_PATH" \
    -an \
    -map 0:v:0 \
    -c:v copy \
    -f rtsp \
    -rtsp_transport tcp \
    "$RTSP_URL"
else
  ffmpeg \
    -hide_banner \
    -re \
    -stream_loop -1 \
    -i "$VIDEO_PATH" \
    -an \
    -map 0:v:0 \
    -vf "fps=${STREAM_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
    -c:v libx264 \
    -preset veryfast \
    -tune zerolatency \
    -crf "$CRF" \
    -maxrate "$MAXRATE" \
    -bufsize "$BUFSIZE" \
    -x264-params "keyint=${STREAM_FPS}:min-keyint=${STREAM_FPS}:scenecut=0:repeat-headers=1" \
    -f rtsp \
    -rtsp_transport tcp \
    "$RTSP_URL"
fi
