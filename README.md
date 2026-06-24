# Object Measurement YOLO

Real-time steel-sheet measurement and quality control from an RTSP stream.
The production monitor tracks each sheet, compares its final measured length
against the 1200 mm target, and records one accepted or rejected result per
finished sheet.

## Requirements

- Python 3.9 or newer
- FFmpeg and `ffprobe`
- MediaMTX
- A YOLO model at `best.pt`
- A source video, by default `videos/steel_plates_cutting.mp4`

## Setup

```bash
python3 -m venv yolo_env
source yolo_env/bin/activate
python -m pip install -r requirements.txt
```

Place the MediaMTX executable in the project directory, install it on `PATH`,
or provide its full path with `MEDIAMTX_BIN`.

## Run

Start the RTSP server, video publisher, and production monitor together:

```bash
./run_qc_pipeline.sh
```

Use a different source video:

```bash
./run_qc_pipeline.sh /full/path/to/video.mp4
```

If MediaMTX is elsewhere:

```bash
MEDIAMTX_BIN=/full/path/to/mediamtx ./run_qc_pipeline.sh
```

Press `q` in the monitor window or `Ctrl+C` in the terminal to stop.

Each run creates a new CSV under `outputs/production_runs/` named
`qr_test_YYYYMMDD_HHMMSS.csv`. A row is written only after a tracked sheet
reaches its final shape and leaves the measurement area.

## Production Settings

The main values are defined near the top of `production.py`:

- Target length: `1200.0 mm`
- Acceptance tolerance: `15.0 mm`
- Default stream: `rtsp://localhost:8554/belt_stream`
