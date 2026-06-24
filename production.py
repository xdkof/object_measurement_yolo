import csv
import json
import cv2
import os
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from urllib.parse import urlparse

BELT_START_Y_PIXEL = 150  
BELT_END_Y_PIXEL = 680    

VISIBLE_BELT_LENGTH_MM = 1945.0  
TARGET_LENGTH_MM = 1200.0  
TOLERANCE_MM = 15.0        
MIN_VALID_LENGTH_MM = 1000.0
DEFAULT_RTSP_URL = "rtsp://localhost:8554/belt_stream"
WINDOW_NAME = "QC Operator Monitor"
RTSP_RETRY_DELAY_SECONDS = 2
FINAL_VERDICT_DISPLAY_SECONDS = 1.0
FINAL_LENGTH_STABLE_FRAMES = 6
FINAL_LENGTH_STABLE_TOLERANCE_MM = 8.0
FINAL_LENGTH_PASSED_PEAK_MM = 35.0
VALIDATION_EDGE_MARGIN_PIXELS = 0
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "production_runs"
CSV_HEADERS = [
    "time",
    "id",
    "sheet_no",
    "status",
    "final_length_mm",
    "delta_mm",
    "target_length_mm",
    "tolerance_mm",
]


def rtsp_target(rtsp_url):
    parsed = urlparse(rtsp_url)
    return parsed.hostname or "localhost", parsed.port or 554


def parse_rate(rate_text):
    if not rate_text or rate_text == "0/0":
        return 0.0

    try:
        numerator, denominator = rate_text.split("/", 1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return 0.0
        return float(numerator) / denominator_value
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_stream_info(rtsp_url):
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        rtsp_url,
    ]

    try:
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = payload.get("streams") or []
    if not streams:
        return None

    stream = streams[0]
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate"))

    if width <= 0 or height <= 0:
        return None

    return width, height, fps


def rtsp_stream_ready(rtsp_url):
    ffprobe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "csv=p=0",
        rtsp_url,
    ]

    try:
        result = subprocess.run(
            ffprobe_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        host, port = rtsp_target(rtsp_url)
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False


def wait_for_rtsp_stream(rtsp_url):
    host, port = rtsp_target(rtsp_url)
    while not rtsp_stream_ready(rtsp_url):
        print(f"[STREAM] Waiting for RTSP stream at {host}:{port} ...")
        time.sleep(RTSP_RETRY_DELAY_SECONDS)


def print_stream_info(width, height, fps):
    if width and height:
        fps_text = f"{fps:.2f}" if fps else "unknown"
        print(f"[STREAM] Input resolution: {width}x{height} @ {fps_text} fps")
    else:
        print("[STREAM] Waiting for stream metadata...")


def unique_csv_output_path(started_at):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_stamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"qr_test_{file_stamp}.csv"

    suffix = 2
    while output_path.exists():
        output_path = OUTPUT_DIR / f"qr_test_{file_stamp}_{suffix}.csv"
        suffix += 1

    return output_path


def create_results_csv(output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()


def append_result_csv_row(output_path, row):
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writerow(row)


def box_is_ready_for_validation(box, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = box
    margin = VALIDATION_EDGE_MARGIN_PIXELS

    return (
        x2 > margin
        and y2 > margin
        and x1 < frame_width - margin
        and y1 < frame_height - margin
    )


class FFmpegFrameReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.proc = None
        self.thread = None
        self.running = False
        self.frame_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None

    def start(self):
        info = probe_stream_info(self.rtsp_url)
        if info is None:
            return False

        self.width, self.height, self.fps = info
        frame_size = self.width * self.height * 3
        if frame_size <= 0:
            return False

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.rtsp_url,
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.running = True
        self.frame_event.clear()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def _reader_loop(self):
        assert self.proc is not None
        assert self.proc.stdout is not None

        frame_size = self.width * self.height * 3
        while self.running and self.proc.poll() is None:
            data = bytearray()
            while len(data) < frame_size and self.running and self.proc.poll() is None:
                chunk = self.proc.stdout.read(frame_size - len(data))
                if not chunk:
                    break
                data.extend(chunk)

            if len(data) < frame_size:
                break

            frame = np.frombuffer(data, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            ).copy()
            with self.lock:
                self.latest_frame = frame
            self.frame_event.set()

        self.running = False
        self.frame_event.set()

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def wait_for_frame(self, timeout=5):
        return self.frame_event.wait(timeout)

    def is_open(self):
        return self.running and self.proc is not None and self.proc.poll() is None

    def stop(self):
        self.running = False

        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

        if self.thread is not None:
            try:
                self.thread.join(timeout=2)
            except KeyboardInterrupt:
                pass
        self.thread = None


def main():
    model = YOLO('best.pt')

    rtsp_url = os.environ.get("RTSP_URL", DEFAULT_RTSP_URL)
    run_started_at = datetime.now()
    output_path = unique_csv_output_path(run_started_at)
    create_results_csv(output_path)
    print(f"[EXPORT] Logging final verdicts to {output_path}")
    reader = None

    wait_for_rtsp_stream(rtsp_url)
    reader = FFmpegFrameReader(rtsp_url)

    while not reader.start():
        print("[STREAM] Failed to start frame reader. Retrying...")
        time.sleep(RTSP_RETRY_DELAY_SECONDS)
        wait_for_rtsp_stream(rtsp_url)

    max_recorded_lengths = {}
    completed_ids = set()
    assigned_production_ids = {}
    last_seen_boxes = {}
    stable_length_frames = {}
    final_verdict_overlays = {}
    production_sheet_counter = 1
    
    belt_pixel_length = BELT_END_Y_PIXEL - BELT_START_Y_PIXEL

    print("--- Production QC Stream Active (RTSP Mode) ---")

    def finalize_sheet(tid, visible_box=None):
        nonlocal production_sheet_counter

        if tid in completed_ids:
            return

        final_length = max_recorded_lengths[tid]
        if final_length < MIN_VALID_LENGTH_MM:
            return

        completed_ids.add(tid)
        official_id = production_sheet_counter
        assigned_production_ids[tid] = official_id
        production_sheet_counter += 1

        status = "ACCEPTED" if abs(final_length - TARGET_LENGTH_MM) <= TOLERANCE_MM else "REJECTED"
        delta_mm = final_length - TARGET_LENGTH_MM
        finalized_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_result_csv_row(
            output_path,
            {
                "time": finalized_at,
                "id": int(tid),
                "sheet_no": official_id,
                "status": status,
                "final_length_mm": f"{final_length:.1f}",
                "delta_mm": f"{delta_mm:.1f}",
                "target_length_mm": f"{TARGET_LENGTH_MM:.1f}",
                "tolerance_mm": f"{TOLERANCE_MM:.1f}",
            },
        )
        print(f"[VERDICT] Sheet #{official_id} | Status: {status} | Final Length: {final_length:.1f} mm")

        if visible_box is not None:
            final_verdict_overlays[tid] = {
                "box": visible_box,
                "official_id": official_id,
                "status": status,
                "final_length": final_length,
                "expires_at": time.monotonic() + FINAL_VERDICT_DISPLAY_SECONDS,
            }

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print_stream_info(reader.width, reader.height, reader.fps)
        if not reader.wait_for_frame(timeout=5):
            print("[STREAM] Waiting for first frame...")

        while True:
            frame = reader.read()
            if frame is None:
                if not reader.is_open():
                    print("Warning: Stream dropped. Attempting to reconnect...")
                    reader.stop()
                    time.sleep(RTSP_RETRY_DELAY_SECONDS)
                    wait_for_rtsp_stream(rtsp_url)
                    reader = FFmpegFrameReader(rtsp_url)
                    while not reader.start():
                        print("[STREAM] Failed to start frame reader. Retrying...")
                        time.sleep(RTSP_RETRY_DELAY_SECONDS)
                        wait_for_rtsp_stream(rtsp_url)
                    print_stream_info(reader.width, reader.height, reader.fps)
                    reader.wait_for_frame(timeout=5)
                else:
                    time.sleep(0.01)
                continue

            results = model.track(frame, persist=True, verbose=False)[0]
            current_frame_ids = []

            if results.boxes and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()
                current_frame_ids = [int(track_id) for track_id in track_ids]

                for box, track_id in zip(boxes, track_ids):
                    track_id = int(track_id)
                    if track_id in completed_ids:
                        continue

                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    visible_box = (x1, y1, x2, y2)

                    sheet_pixel_length = y2 - y1
                    current_length_mm = (sheet_pixel_length / belt_pixel_length) * VISIBLE_BELT_LENGTH_MM
                    previous_max_length = max_recorded_lengths.get(track_id)

                    if previous_max_length is None:
                        max_recorded_lengths[track_id] = current_length_mm
                        stable_length_frames[track_id] = 0
                    elif current_length_mm > previous_max_length + FINAL_LENGTH_STABLE_TOLERANCE_MM:
                        max_recorded_lengths[track_id] = current_length_mm
                        stable_length_frames[track_id] = 0
                    else:
                        if current_length_mm > previous_max_length:
                            max_recorded_lengths[track_id] = current_length_mm

                        stable_length_frames[track_id] = stable_length_frames.get(track_id, 0) + 1

                    final_length = max_recorded_lengths[track_id]
                    stable_ready = stable_length_frames.get(track_id, 0) >= FINAL_LENGTH_STABLE_FRAMES
                    passed_peak_ready = current_length_mm <= final_length - FINAL_LENGTH_PASSED_PEAK_MM

                    last_seen_boxes[track_id] = visible_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

                    if (
                        final_length >= MIN_VALID_LENGTH_MM
                        and box_is_ready_for_validation(visible_box, frame.shape)
                        and (stable_ready or passed_peak_ready)
                    ):
                        finalize_sheet(track_id, visible_box)

            for tid in list(max_recorded_lengths.keys()):
                if tid not in current_frame_ids and tid not in completed_ids:
                    stable_length_frames[tid] = 0

            now = time.monotonic()
            for tid, overlay in list(final_verdict_overlays.items()):
                if now >= overlay["expires_at"]:
                    del final_verdict_overlays[tid]
                    continue

                x1, y1, x2, y2 = overlay["box"]
                status = overlay["status"]
                color = (0, 255, 0) if status == "ACCEPTED" else (255, 200, 0)
                label = (
                    f"Sheet #{overlay['official_id']} | {status} | "
                    f"Final Length: {overlay['final_length']:.1f} mm"
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                label_x = max(5, min(x1, frame.shape[1] - text_width - 10))
                label_y = y1 - 10
                if label_y - text_height - baseline < 0:
                    label_y = min(frame.shape[0] - baseline - 5, y2 + text_height + 10)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (label_x, label_y), font, font_scale, color, thickness
                )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\n[QC] Stop requested.")
    finally:
        if reader is not None:
            try:
                reader.stop()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(f"[EXPORT] CSV results saved at {output_path}")

if __name__ == '__main__':
    main()
