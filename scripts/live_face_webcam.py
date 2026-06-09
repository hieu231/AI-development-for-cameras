import argparse
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import SessionLocal
from src.face_recognition.face_engine import get_face_engine
from src.face_recognition.repository import FaceProfileRepository


def find_face_matches(repo, embeddings, threshold=0.45):
    matches = []
    if repo is None:
        return [None for _ in embeddings]

    for embedding in embeddings:
        match_result = repo.find_best_match(
            embedding,
            threshold=threshold,
            active_only=True,
        )
        if not match_result:
            matches.append(None)
            continue

        profile, similarity = match_result
        matches.append(
            {
                "profile_id": profile.id,
                "employee_id": profile.employee_id,
                "employee_name": profile.employee_name,
                "similarity": similarity,
            }
        )
    return matches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live face recognition on a local webcam."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to open. Default: 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Requested capture width. Default: 640",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested capture height. Default: 480",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the preview horizontally.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "gui", "mjpeg", "console"),
        default="auto",
        help="Preview mode. 'auto' tries GUI first, then falls back to MJPEG. Default: auto",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for MJPEG preview mode. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for MJPEG preview mode. Default: 8765",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames. 0 means run until interrupted. Useful for testing.",
    )
    return parser.parse_args()


def draw_overlay(frame, num_faces, fps, status_text, status_color):
    cv2.putText(
        frame,
        status_text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Faces: {num_faces}",
        (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (16, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 200, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Press Q or ESC to quit",
        (16, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def detect_face_boxes(engine, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(rgb_frame)
    boxes, probabilities = engine.detector.detector.detect(pil_frame, landmarks=False)

    if boxes is None:
        return []

    if probabilities is None:
        probabilities = [None] * len(boxes)

    faces = []
    for box, probability in zip(boxes, probabilities):
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        faces.append({
            "box": (x1, y1, x2, y2),
            "confidence": float(probability) if probability is not None else 0.0,
        })
    return faces


def draw_face_boxes(frame, detected_faces):
    for detected_face in detected_faces:
        x1, y1, x2, y2 = detected_face["box"]
        confidence = detected_face["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            f"face {confidence:.2f}",
            (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )


class MJPEGStreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes = None

    def update(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return
        with self.lock:
            self.jpeg_bytes = encoded.tobytes()

    def get(self):
        with self.lock:
            return self.jpeg_bytes


def supports_highgui():
    try:
        cv2.namedWindow("__highgui_probe__")
        cv2.destroyWindow("__highgui_probe__")
        return True
    except cv2.error:
        return False


def create_mjpeg_handler(stream_state):
    class MJPEGHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = (
                    "<html><head><title>Live Face Recognition</title></head>"
                    "<body style='margin:0;background:#111;color:#eee;font-family:sans-serif;'>"
                    "<div style='padding:12px;'>OpenCV GUI is unavailable, using browser preview.</div>"
                    "<img src='/stream.mjpg' style='display:block;width:100%;max-width:960px;margin:0 auto;' />"
                    "</body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream.mjpg":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    jpeg_bytes = stream_state.get()
                    if jpeg_bytes is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, format, *args):
            return

    return MJPEGHandler


def start_mjpeg_server(host, port, stream_state):
    server = ThreadingHTTPServer((host, port), create_mjpeg_handler(stream_state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    args = parse_args()

    print("Initializing face recognition engine...")
    engine = get_face_engine(force_reload=True)
    if engine is None:
        raise RuntimeError(
            "Face recognition engine is unavailable. Check ENABLE_FACE_RECOGNITION and model setup."
        )

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Live Face Recognition"
    previous_time = time.perf_counter()
    stream_state = MJPEGStreamState()
    server = None
    highgui_available = supports_highgui()

    mode = args.mode
    if mode == "auto":
        mode = "gui" if highgui_available else "mjpeg"

    if mode == "gui" and not highgui_available:
        print("OpenCV GUI is not available in this environment. Falling back to MJPEG preview.")
        mode = "mjpeg"

    if mode == "mjpeg":
        server = start_mjpeg_server(args.host, args.port, stream_state)
        print(f"MJPEG preview available at http://{args.host}:{args.port}")
    elif mode == "console":
        print("Console mode active. No preview window will be shown.")

    try:
        frame_count = 0
        last_console_log = 0.0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            result = engine.process_image(frame)
            detected_faces = detect_face_boxes(engine, frame) if result.get("success") else []
            num_faces = result.get("num_faces", 0) if result.get("success") else 0
            status_text = "Face engine OK" if result.get("success") else f"Error: {result.get('error')}"
            status_color = (0, 200, 0) if result.get("success") else (0, 0, 255)

            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - previous_time, 1e-6)
            previous_time = current_time

            draw_face_boxes(frame, detected_faces)
            draw_overlay(frame, num_faces, fps, status_text, status_color)
            if mode == "gui":
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
            elif mode == "mjpeg":
                stream_state.update(frame)
            else:
                if current_time - last_console_log >= 1.0:
                    print(f"faces={num_faces} fps={fps:.1f} status={status_text}")
                    last_console_log = current_time

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    finally:
        cap.release()
        if server is not None:
            server.shutdown()
            server.server_close()
        if mode == "gui" and highgui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
