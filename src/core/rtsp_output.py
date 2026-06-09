"""
RTSP Output Worker — push annotated frames to a local RTSP server (e.g. MediaMTX)
via FFmpeg subprocess.

Each worker reads frames from FrameBufferManager, pipes raw video to an FFmpeg
process that encodes to H.264 and pushes to an RTSP endpoint.

Targeted for NVIDIA Jetson (h264_nvenc / nvv4l2h264enc), falls back to libx264.
"""

import logging
import os
import subprocess
import shutil
import time
import threading
from uuid import UUID
from typing import Dict, Optional

import numpy as np
import cv2

from src.core.frame_buffer import frame_buffer_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RTSP_OUTPUT_ENABLED = os.getenv("RTSP_OUTPUT_ENABLED", "false").lower() == "true"
RTSP_OUTPUT_URL = os.getenv("RTSP_OUTPUT_URL", "rtsp://localhost:8554")
RTSP_ENCODER = os.getenv("RTSP_ENCODER", "auto")  # auto | h264_nvenc | libx264
RTSP_DEFAULT_FPS = int(os.getenv("RTSP_DEFAULT_FPS", "24"))

ALLOWED_FPS = {5, 10, 24, 45, 60}


def _detect_encoder() -> str:
    """Pick the best available H.264 encoder."""
    if RTSP_ENCODER != "auto":
        return RTSP_ENCODER

    # Try NVENC first (Jetson / desktop NVIDIA)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("FFmpeg not found on PATH — falling back to libx264")
        return "libx264"

    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        encoders_output = result.stdout + result.stderr
        if "h264_nvenc" in encoders_output:
            logger.info("RTSP encoder: using h264_nvenc (NVENC)")
            return "h264_nvenc"
    except Exception:
        pass

    logger.info("RTSP encoder: using libx264 (fallback)")
    return "libx264"


class RTSPOutputWorker:
    """Background thread that streams annotated frames for one camera via FFmpeg → RTSP."""

    def __init__(self, camera_id: UUID, fps: int = RTSP_DEFAULT_FPS):
        self.camera_id = camera_id
        self.fps = fps if fps in ALLOWED_FPS else 24
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning(f"RTSP worker for camera {self.camera_id} already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"rtsp-out-{str(self.camera_id)[:8]}")
        self._thread.start()
        logger.info(f"RTSP output started for camera {self.camera_id} at {self.fps} FPS")

    def stop(self) -> None:
        self._stop_event.set()
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info(f"RTSP output stopped for camera {self.camera_id}")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        encoder = _detect_encoder()
        stream_url = f"{RTSP_OUTPUT_URL}/{self.camera_id}"
        interval = 1.0 / self.fps

        while not self._stop_event.is_set():
            try:
                # Wait until first frame is available to know dimensions
                frame_data = None
                while not self._stop_event.is_set():
                    frame_data = frame_buffer_manager.get(self.camera_id)
                    if frame_data is not None:
                        break
                    time.sleep(0.2)

                if self._stop_event.is_set():
                    break

                first_frame, _ = frame_data
                h, w = first_frame.shape[:2]

                # Build FFmpeg command
                cmd = self._build_ffmpeg_cmd(encoder, w, h, stream_url)
                logger.info(f"RTSP [{str(self.camera_id)[:8]}] launching FFmpeg: {' '.join(cmd)}")

                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                # Feed frames
                while not self._stop_event.is_set():
                    t0 = time.time()
                    result = frame_buffer_manager.get(self.camera_id)
                    if result is not None:
                        bgr_frame, _ = result
                        # Resize if resolution changed
                        fh, fw = bgr_frame.shape[:2]
                        if (fh, fw) != (h, w):
                            bgr_frame = cv2.resize(bgr_frame, (w, h))
                        try:
                            self._process.stdin.write(bgr_frame.tobytes())
                        except BrokenPipeError:
                            logger.warning(f"RTSP [{str(self.camera_id)[:8]}] FFmpeg pipe broken, restarting…")
                            break

                    # Pace
                    elapsed = time.time() - t0
                    sleep_time = interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except Exception as exc:
                logger.error(f"RTSP [{str(self.camera_id)[:8]}] error: {exc}", exc_info=True)
                time.sleep(2)
            finally:
                if self._process:
                    try:
                        self._process.stdin.close()
                    except Exception:
                        pass
                    try:
                        self._process.terminate()
                        self._process.wait(timeout=3)
                    except Exception:
                        pass
                    self._process = None

    # ------------------------------------------------------------------
    def _build_ffmpeg_cmd(self, encoder: str, w: int, h: int, stream_url: str) -> list:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self.fps),
            "-i", "-",
        ]
        if encoder == "h264_nvenc":
            cmd += [
                "-c:v", "h264_nvenc",
                "-preset", "ll",          # low latency
                "-delay", "0",
                "-g", str(self.fps * 2),   # GOP = 2 seconds
                "-bf", "0",
            ]
        else:
            cmd += [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-g", str(self.fps * 2),
                "-bf", "0",
            ]
        cmd += [
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            stream_url,
        ]
        return cmd


# ---------------------------------------------------------------------------
# RTSPOutputManager — manages workers per camera
# ---------------------------------------------------------------------------
class RTSPOutputManager:
    """Start/stop RTSP output workers for multiple cameras."""

    def __init__(self):
        self._workers: Dict[UUID, RTSPOutputWorker] = {}

    def start(self, camera_id: UUID, fps: int = RTSP_DEFAULT_FPS) -> bool:
        if camera_id in self._workers:
            logger.warning(f"RTSP output for camera {camera_id} already running")
            return False
        worker = RTSPOutputWorker(camera_id, fps)
        worker.start()
        self._workers[camera_id] = worker
        return True

    def stop(self, camera_id: UUID) -> bool:
        worker = self._workers.pop(camera_id, None)
        if worker is None:
            return False
        worker.stop()
        return True

    def stop_all(self) -> None:
        for cam_id in list(self._workers.keys()):
            self.stop(cam_id)

    def list_cameras(self) -> list:
        return [str(cid) for cid in self._workers.keys()]


# Module-level singleton
rtsp_output_manager = RTSPOutputManager()
