import os

# OPENCV_FFMPEG_CAPTURE_OPTIONS is set in src/core/opencv_config.py which is
# imported by capture_backends.py BEFORE its `import cv2`.  We re-import here
# for explicitness (the module is a no-op on second import).
import src.core.opencv_config  # noqa: F401

import cv2
import time
import logging
import numpy as np
from queue import Queue, Full, Empty
from threading import Thread, Event, Lock
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models.camera import Camera
from src.core.camera_single_thread import SingleThreadProcessor
from src.core.capture_backends import (
    _is_vss_rate_limit_error,
    build_refreshable_vss_source_url,
    create_vss_capture,
    uses_vss_backend,
)
from src.core.frame_buffer import frame_buffer_manager
from src.core.rtsp_output import rtsp_output_manager, RTSP_OUTPUT_ENABLED
from src.utils.image_utils import preprocess_frame

logger = logging.getLogger(__name__)


def _get_camera_frame_queue_size() -> int:
    """Return per-camera frame queue size with a low-latency default."""
    raw_value = os.getenv("CAMERA_FRAME_QUEUE_SIZE", "2").strip()
    try:
        size = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid CAMERA_FRAME_QUEUE_SIZE='%s', falling back to 2",
            raw_value,
        )
        size = 2
    return max(1, min(size, 30))


def _is_non_retryable_capture_error(exc: Exception) -> bool:
    return _is_vss_rate_limit_error(exc)


def _is_non_retryable_capture_error_str(error_msg: str | None) -> bool:
    """String-based check for non-retryable errors (used when only the error message is available)."""
    if not error_msg:
        return False
    msg = error_msg.lower()
    return (
        "giới hạn đăng nhập" in msg
        or "login too frequently" in msg
        or "too many requests" in msg
    )


class CameraThread:
    """Thread để capture frames từ camera stream sources."""

    def __init__(
        self,
        camera_id: UUID,
        rtsp_url: str,
        frame_queue: Queue,
        stop_event: Event,
        preprocess_resolution: str = None,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.capture_source_url: Optional[str] = None
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.connected_event = (
            Event()
        )  # Signal when successfully connected and capturing
        self.thread = None
        self.cap = None
        self.startup_error: Optional[str] = None
        self.preprocess_resolution = preprocess_resolution
        raw_buffer_fps = int(os.getenv("RAW_FRAME_BUFFER_FPS", "0"))
        if raw_buffer_fps > 0:
            raw_buffer_fps = max(1, min(raw_buffer_fps, 60))
            self._raw_frame_interval = 1.0 / raw_buffer_fps
        else:
            # 0 = no throttle, push every captured frame
            self._raw_frame_interval = 0.0
        self._last_raw_push = 0.0
        self._drop_oldest_on_full = (
            os.getenv("DROP_OLDEST_ON_FULL", "true").lower() == "true"
        )

    def start(self):
        """Khởi động thread capture"""
        # daemon=True so a capture thread blocked in cap.read() (up to the
        # 5 s FFmpeg socket timeout) never prevents the whole process from
        # exiting on SIGTERM. The old daemon=False combined with the 1 s
        # join() in stop() left zombie threads on shutdown holding CUDA and
        # FFmpeg handles across reloads.
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.camera_id}: Capture thread started")

    def _get_stream_backend(self):
        """Return the OpenCV backend to use for this stream.

        Note: we deliberately do NOT mutate OPENCV_FFMPEG_CAPTURE_OPTIONS here.
        OpenCV reads that env var exactly once on first VideoCapture creation
        and ignores later changes. It is set at module-import time (see the
        top of this file) with TCP transport + 5s socket timeout so dead
        streams are detected quickly instead of blocking for 30 s.
        """
        return cv2.CAP_FFMPEG

    def _create_capture(self):
        """Create a capture backend for the configured stream URL."""
        if uses_vss_backend(self.rtsp_url):
            if self.capture_source_url is None:
                self.capture_source_url = self._resolve_capture_source_url()
            return create_vss_capture(self.capture_source_url)

        backend = self._get_stream_backend()
        return cv2.VideoCapture(self.rtsp_url, backend)

    def _resolve_capture_source_url(self) -> str:
        source_url = self.rtsp_url
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == self.camera_id).first()
            if camera is None or (camera.protocol or "").upper() != "VSS":
                return source_url

            refreshable_source_url = build_refreshable_vss_source_url(
                source_url,
                base_url=camera.vss_base_url,
                username=camera.vss_username,
                password_md5=camera.vss_password,
                device_id=camera.vss_device_id,
                channel=camera.vss_channel,
            )
            if refreshable_source_url != source_url:
                logger.info(
                    "Camera %s: enriched HTTP-FLV URL with stored VSS credentials for token refresh",
                    self.camera_id,
                )
            return refreshable_source_url
        except Exception as exc:
            logger.warning(
                "Camera %s: failed to enrich VSS source URL for token refresh: %s",
                self.camera_id,
                exc,
            )
            return source_url
        finally:
            db.close()

    def _capture_frames(self):
        """Capture frames từ stream URL và đưa vào queue."""
        retry_count = 0
        consecutive_failures = 0
        # Keep reconnect threshold short so a dead stream is noticed in a few
        # seconds, not minutes. With the 5 s FFmpeg socket timeout configured
        # at module import time, 5 failures ≈ 25 s worst-case before reconnect,
        # instead of the old 30 × 30 s = 15-minute hang that produced:
        #     "Too many failures (30), reconnecting..."
        # every 15 minutes in the logs.
        max_consecutive_failures = int(
            os.getenv("CAMERA_MAX_CONSECUTIVE_FAILURES", "5")
        )

        while not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    # Cleanup old capture if exists
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except:
                            pass
                        self.cap = None

                    logger.info(
                        f"Camera {self.camera_id}: Connecting to stream {self.rtsp_url[:50]}..."
                    )

                    self.cap = self._create_capture()

                    # Set buffer size to reduce latency and avoid decoder issues
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # Allow time for stream to initialize (increased for RTSP stability)
                    time.sleep(2.0)

                    if not self.cap.isOpened():
                        retry_count += 1
                        backoff = min(2 * (2 ** (retry_count - 1)), 60)
                        logger.warning(
                            f"Camera {self.camera_id}: Connection attempt {retry_count} failed, retrying in {backoff}s..."
                        )
                        self.startup_error = f"Connection attempt {retry_count} failed, retrying in {backoff}s"
                        time.sleep(backoff)
                        continue

                    retry_count = 0
                    consecutive_failures = 0
                    self.startup_error = None
                    logger.info(f"Camera {self.camera_id}: Connected successfully")

                    # Detect and store source FPS for WebRTC to match
                    src_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if not src_fps or src_fps < 1 or src_fps > 120:
                        src_fps = 25.0  # safe fallback
                    frame_buffer_manager.update_source_fps(self.camera_id, src_fps)
                    logger.info(
                        f"Camera {self.camera_id}: Source FPS detected = {src_fps}"
                    )

                    signal_on_open = getattr(
                        self.cap, "signals_connection_on_open", None
                    )
                    if callable(signal_on_open) and signal_on_open():
                        logger.info(
                            f"Camera {self.camera_id}: Stream transport connected, awaiting first frame"
                        )

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(
                            f"Camera {self.camera_id}: Too many failures ({consecutive_failures}), reconnecting..."
                        )
                        if self.cap is not None:
                            try:
                                self.cap.release()
                            except:
                                pass
                        self.cap = None
                        self.connected_event.clear()  # Clear connection status on failure
                        consecutive_failures = 0
                        time.sleep(1)
                    else:
                        time.sleep(0.01)
                    continue

                # Reset failure counter on success
                consecutive_failures = 0

                # Signal that we're successfully capturing frames (only once)
                if not self.connected_event.is_set():
                    self.connected_event.set()
                    self.startup_error = None
                    logger.info(
                        f"Camera {self.camera_id}: Successfully capturing frames"
                    )

                # Push raw frame to buffer at capture rate (≥24 FPS)
                # This keeps WebRTC streams smooth even when AI processing is slow
                now = time.time()
                if (
                    self._raw_frame_interval <= 0
                    or now - self._last_raw_push >= self._raw_frame_interval
                ):
                    raw = frame
                    if self.preprocess_resolution:
                        raw = preprocess_frame(raw, self.preprocess_resolution)
                    frame_buffer_manager.update_raw(self.camera_id, raw.copy(), now)
                    self._last_raw_push = now

                try:
                    self.frame_queue.put(frame, block=False)
                except Full:
                    if self._drop_oldest_on_full:
                        try:
                            _ = self.frame_queue.get_nowait()
                            self.frame_queue.put(frame, block=False)
                        except (Empty, Full):
                            pass

            except Exception as e:
                is_non_retryable = _is_non_retryable_capture_error(e)
                if not self.connected_event.is_set():
                    self.startup_error = str(e)
                logger.error(
                    f"Camera {self.camera_id}: Capture error - {e}",
                    exc_info=not is_non_retryable,
                )
                # Force cleanup on error
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except:
                        pass
                    self.cap = None
                if is_non_retryable:
                    logger.error(
                        f"Camera {self.camera_id}: Non-retryable capture error, stopping startup"
                    )
                    break
                time.sleep(1)

        # Final cleanup
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(
                    f"Camera {self.camera_id}: Error during cap cleanup: {e}"
                )
        self.cap = None
        logger.info(f"Camera {self.camera_id}: Capture thread stopped and cleaned up")

    def stop(self):
        """Dừng thread capture"""
        logger.info(f"Camera {self.camera_id}: Stopping capture thread...")
        self.stop_event.set()

        # Force release VideoCapture in background to avoid blocking
        if self.cap is not None:
            cap_to_release = self.cap
            self.cap = None  # Set to None immediately

            def release_in_background():
                try:
                    logger.info(
                        f"Camera {self.camera_id}: Force releasing VideoCapture..."
                    )
                    cap_to_release.release()
                    logger.info(f"Camera {self.camera_id}: VideoCapture released")
                except Exception as e:
                    logger.warning(
                        f"Camera {self.camera_id}: Error during force release: {e}"
                    )

            import threading

            release_thread = threading.Thread(target=release_in_background, daemon=True)
            release_thread.start()

        # Wait for thread to finish with configurable timeout.
        if self.thread is not None and self.thread.is_alive():
            logger.info(
                f"Camera {self.camera_id}: Waiting for capture thread to finish..."
            )
            join_timeout = max(
                0.1,
                float(os.getenv("CAMERA_CAPTURE_STOP_TIMEOUT_SEC", "5")),
            )
            self.thread.join(timeout=join_timeout)
            if self.thread.is_alive():
                logger.warning(
                    f"Camera {self.camera_id}: Capture thread still alive after {join_timeout:.1f}s"
                )
            else:
                self.thread = None
        if self.thread is not None and not self.thread.is_alive():
            self.thread = None

        logger.info(f"Camera {self.camera_id}: Capture thread stop completed")


class ProcessorThread:
    """Thread để xử lý AI inference từ frames"""

    def __init__(
        self,
        camera_id: UUID,
        frame_queue: Queue,
        stop_event: Event,
        show_display: bool = True,
        preprocess_resolution: str = None,
    ):
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.max_frame_in_queue = 30
        self.frame_skip = 1
        self.stop_event = stop_event
        self.thread = None
        self.processor = SingleThreadProcessor(camera_id)
        self.show_display = show_display
        self.window_name = f"Camera {str(camera_id)[:8]}"
        self._window_created = False
        self.preprocess_resolution = preprocess_resolution
        self.process_latest_only = (
            os.getenv("PROCESS_LATEST_FRAME_ONLY", "true").lower() == "true"
        )
        self.latest_frame_lock = Lock()
        self.latest_annotated_frame = None
        # Periodic CUDA cache flush to bound memory fragmentation on long-running
        # Jetson deployments. 0 disables the call (useful for CPU-only tests).
        self._cuda_empty_cache_every = int(
            os.getenv("CUDA_EMPTY_CACHE_EVERY_FRAMES", "600")
        )
        self._frames_since_cache_flush = 0

        if self.preprocess_resolution:
            logger.info(
                f"Camera {self.camera_id}: Preprocessing resolution set to {self.preprocess_resolution}"
            )

    def start(self):
        """Khởi động thread xử lý"""
        # daemon=True — see comment on CameraThread.start() for rationale.
        self.thread = Thread(target=self._process_frames, daemon=True)
        self.thread.start()
        logger.info(
            f"Camera {self.camera_id}: Processor thread started with {len(self.processor.models)} models"
        )

    def _close_display_window(self) -> None:
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except Exception:
            pass
        finally:
            self._window_created = False

    def _process_frames(self):
        """Xử lý frames từ queue"""
        while not self.stop_event.is_set():
            try:
                # Lấy frame từ queue với timeout ngắn để check stop_event thường xuyên
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except:
                    # Queue empty or timeout, continue loop to check stop_event
                    continue

                if self.process_latest_only:
                    try:
                        while True:
                            frame = self.frame_queue.get_nowait()
                    except Empty:
                        pass

                # Preprocess frame if resolution is specified
                if self.preprocess_resolution:
                    frame = preprocess_frame(frame, self.preprocess_resolution)

                # Implement FPS limiting — default 1 FPS to avoid flooding.
                # Set env AI_FPS_LIMIT=0 to run unlimited.
                ai_fps_limit = int(os.getenv("AI_FPS_LIMIT", "0"))
                if ai_fps_limit > 0:
                    start_time = time.time()
                    # Chạy inference
                    results = self.processor.process_frame(frame)

                    # Calculate sleep time to maintain FPS
                    process_time = time.time() - start_time
                    delay = (1.0 / ai_fps_limit) - process_time
                    if delay > 0:
                        time.sleep(delay)
                else:
                    # Chạy inference (tốc độ tối đa)
                    results = self.processor.process_frame(frame)

                # Build the stream/display frame by redrawing every model's
                # detections onto a fresh copy of the clean input frame.
                # We do NOT chain per-model annotated frames anymore: each
                # model now runs inference on its own private clean copy
                # (see SingleThreadProcessor.process_frame), so their
                # annotated outputs cannot be composited pixel-wise and
                # chaining was exactly what caused multi-model inference
                # to break.
                combined_frame = self.processor.compose_annotations(frame, results)
                stream_detections = []
                for model_key, result in results.items():
                    metadata = result.metadata or {}
                    detection_groups = (
                        metadata.get("detections", []) or [],
                        metadata.get("people_statuses", []) or [],
                    )
                    for detections in detection_groups:
                        for detection in detections:
                            bbox = detection.get("bbox")
                            if not bbox or len(bbox) < 4:
                                continue
                            stream_detections.append(
                                {
                                    "bbox": [int(v) for v in bbox[:4]],
                                    "label": detection.get("display_name")
                                    or detection.get("label")
                                    or detection.get("class_name")
                                    or model_key,
                                    "confidence": detection.get("confidence"),
                                    "track_id": detection.get("track_id"),
                                    "model_name": metadata.get("model") or model_key,
                                }
                            )

                # Store latest annotated frame for streaming (thread-safe)
                with self.latest_frame_lock:
                    self.latest_annotated_frame = combined_frame

                # Also publish to centralised frame buffer for WebRTC / RTSP output
                # Pass original frame as base_frame so overlay diff is always correct
                frame_buffer_manager.update(
                    self.camera_id,
                    combined_frame,
                    time.time(),
                    base_frame=frame,
                    detections=stream_detections,
                )

                # Periodically release cached CUDA blocks to keep long-running
                # Jetson deployments from fragmenting GPU memory. Cheap when
                # nothing is to free; bounds the steady-state VRAM footprint.
                if self._cuda_empty_cache_every > 0:
                    self._frames_since_cache_flush += 1
                    if self._frames_since_cache_flush >= self._cuda_empty_cache_every:
                        self._frames_since_cache_flush = 0
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                # Display frame với detections (cv2.imshow)
                # Note: cv2.imshow CAN cause issues in threads, especially with Qt backend
                # Only use when explicitly enabled and NOT in server mode
                if self.show_display:
                    try:
                        # Use the composite produced above — it already has
                        # every model's detections drawn on a clean base.
                        display_frame = combined_frame if combined_frame is not None else frame

                        # Thêm info text lên frame
                        info_text = f"Camera: {str(self.camera_id)[:8]} | Models: {len(results)}"
                        cv2.putText(
                            display_frame,
                            info_text,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )

                        # Show frame
                        cv2.imshow(self.window_name, display_frame)
                        cv2.waitKey(1)  # Important: process window events
                        self._window_created = True
                    except Exception as e:
                        # cv2.imshow can fail in threads or headless environments
                        # Disable display and continue processing
                        logger.warning(
                            f"Camera {self.camera_id}: Display error, disabling cv2.imshow: {e}"
                        )
                        self.show_display = False
                        self._close_display_window()

                # Results now contain DetectionResult objects
                # Events are automatically saved if detection_result.event == True

            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(
                        f"Camera {self.camera_id}: Processor error - {e}", exc_info=True
                    )
                time.sleep(0.1)

        # Cleanup: destroy window when stopped (safe cleanup)
        self._close_display_window()

        logger.info(f"Camera {self.camera_id}: Processor thread stopped")

    def reload_models(self):
        """Reload models (khi có thay đổi enable/disable)"""
        logger.info(f"Camera {self.camera_id}: Reloading models...")
        self.processor.load_models()
        logger.info(
            f"Camera {self.camera_id}: Reloaded {len(self.processor.models)} models"
        )

    def stop(self):
        """Dừng thread xử lý"""
        logger.info(f"Camera {self.camera_id}: Stopping processor thread...")
        self.stop_event.set()

        # Destroy window immediately (safe even if window doesn't exist)
        self._close_display_window()

        # Wait for thread to finish with configurable timeout.
        if self.thread is not None and self.thread.is_alive():
            logger.info(
                f"Camera {self.camera_id}: Waiting for processor thread to finish..."
            )
            join_timeout = max(
                0.1,
                float(os.getenv("CAMERA_PROCESSOR_STOP_TIMEOUT_SEC", "5")),
            )
            self.thread.join(timeout=join_timeout)
            if self.thread.is_alive():
                logger.warning(
                    f"Camera {self.camera_id}: Processor thread still alive after {join_timeout:.1f}s"
                )
            else:
                self.thread = None
        if self.thread is not None and not self.thread.is_alive():
            self.thread = None

        # Cleanup processor and models
        try:
            logger.info(f"Camera {self.camera_id}: Cleaning up processor...")
            self.processor.cleanup()
        except Exception as e:
            logger.warning(
                f"Camera {self.camera_id}: Error during processor cleanup: {e}"
            )

        logger.info(f"Camera {self.camera_id}: Processor thread stop completed")


class CameraManager:
    """Quản lý một camera với 2 threads"""

    def __init__(
        self,
        camera_id: UUID,
        rtsp_url: str,
        show_display: bool = True,
        preprocess_resolution: str = None,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        queue_size = _get_camera_frame_queue_size()
        self.frame_queue = Queue(maxsize=queue_size)
        self.stop_event = Event()  # Tạo mới Event, đảm bảo clear state
        self.stop_event.clear()  # Ensure it's cleared

        logger.info(f"Camera {self.camera_id}: Frame queue size set to {queue_size}")

        self.capture_thread = CameraThread(
            camera_id,
            rtsp_url,
            self.frame_queue,
            self.stop_event,
            preprocess_resolution,
        )
        self.processor_thread = ProcessorThread(
            camera_id,
            self.frame_queue,
            self.stop_event,
            show_display,
            preprocess_resolution,
        )

    def start(self):
        """Khởi động cả 2 threads"""
        # Ensure stop_event is cleared before starting
        self.stop_event.clear()

        # Pre-seed frame buffer so streams are always "online" from the moment camera starts
        # (even before first frame is processed / when there are no events)
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Connecting...",
            (200, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        frame_buffer_manager.update(self.camera_id, placeholder, time.time())

        self.capture_thread.start()
        self.processor_thread.start()

        # Auto-start RTSP output worker if enabled
        if RTSP_OUTPUT_ENABLED:
            try:
                rtsp_output_manager.start(self.camera_id)
                logger.info(f"Camera {self.camera_id}: RTSP output worker started")
            except Exception as e:
                logger.warning(
                    f"Camera {self.camera_id}: Failed to start RTSP output: {e}"
                )

        logger.info(f"Camera {self.camera_id}: Manager started")

    def stop(self) -> bool:
        """Dừng cả 2 threads"""
        logger.info(f"Camera {self.camera_id}: Stopping manager...")

        # Set stop event first
        self.stop_event.set()

        # Clear queue first to unblock any waiting threads
        logger.info(f"Camera {self.camera_id}: Clearing queue...")
        try:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            logger.info(f"Camera {self.camera_id}: Queue cleared")
        except Exception as e:
            logger.warning(f"Camera {self.camera_id}: Error clearing queue: {e}")

        # Stop processor first (consumer)
        logger.info(f"Camera {self.camera_id}: Stopping processor...")
        self.processor_thread.stop()

        # Then stop capture (producer)
        logger.info(f"Camera {self.camera_id}: Stopping capture...")
        self.capture_thread.stop()

        # Best-effort grace wait so caller can treat stop as synchronous.
        stop_grace_sec = max(0.2, float(os.getenv("CAMERA_STOP_GRACE_SEC", "5")))
        deadline = time.time() + stop_grace_sec
        while time.time() < deadline:
            processor_alive = bool(
                self.processor_thread.thread is not None
                and self.processor_thread.thread.is_alive()
            )
            capture_alive = bool(
                self.capture_thread.thread is not None
                and self.capture_thread.thread.is_alive()
            )
            if not processor_alive and not capture_alive:
                break
            time.sleep(0.05)

        # Force destroy any remaining windows for this camera
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass

        # Stop RTSP output worker if running
        try:
            rtsp_output_manager.stop(self.camera_id)
        except Exception:
            pass

        # Remove frame buffer entry
        frame_buffer_manager.remove(self.camera_id)

        # Close any active WebRTC connections for this camera
        try:
            from src.core.webrtc_manager import webrtc_manager
            webrtc_manager.close_camera_connections_sync(self.camera_id)
        except Exception as e:
            logger.warning(
                f"Camera {self.camera_id}: Error closing WebRTC connections: {e}"
            )

        processor_alive = bool(
            self.processor_thread.thread is not None
            and self.processor_thread.thread.is_alive()
        )
        capture_alive = bool(
            self.capture_thread.thread is not None
            and self.capture_thread.thread.is_alive()
        )
        stopped_cleanly = not processor_alive and not capture_alive
        if not stopped_cleanly:
            logger.warning(
                "Camera %s: manager stop timeout (processor_alive=%s, capture_alive=%s)",
                self.camera_id,
                processor_alive,
                capture_alive,
            )
        logger.info(f"Camera {self.camera_id}: Manager stopped and cleaned up")
        return stopped_cleanly

    def reload_models(self):
        """Reload models"""
        self.processor_thread.reload_models()


class ThreadManager:
    """Quản lý tất cả cameras và threads"""

    def __init__(self):
        self.cameras: Dict[UUID, CameraManager] = {}
        self.last_start_errors: Dict[UUID, str] = {}
        logger.info("ThreadManager initialized")

    def add_camera(
        self,
        camera_id: UUID,
        rtsp_url: str,
        show_display: bool = True,
        preprocess_resolution: str = None,
    ) -> bool:
        """Thêm và start camera"""
        if camera_id in self.cameras:
            logger.warning(f"Camera {camera_id} already exists")
            return False

        try:
            self.last_start_errors.pop(camera_id, None)
            manager = CameraManager(
                camera_id, rtsp_url, show_display, preprocess_resolution
            )
            manager.start()
            self.cameras[camera_id] = manager

            # Wait for camera to successfully connect
            # VSS cameras need extra time for login retries + token resolution
            if uses_vss_backend(rtsp_url):
                connect_timeout = int(
                    os.getenv("VSS_CAMERA_CONNECT_TIMEOUT_SEC", "120")
                )
            else:
                connect_timeout = int(os.getenv("CAMERA_CONNECT_TIMEOUT_SEC", "10"))
            logger.info(
                f"Camera {camera_id}: Waiting for successful connection (timeout={connect_timeout}s)..."
            )
            connected = False
            deadline = time.time() + connect_timeout

            while time.time() < deadline:
                if manager.capture_thread.connected_event.wait(timeout=0.1):
                    connected = True
                    break

                startup_error = manager.capture_thread.startup_error
                capture_thread = manager.capture_thread.thread
                capture_thread_exited = (
                    capture_thread is not None and not capture_thread.is_alive()
                )

                if startup_error:
                    self.last_start_errors[camera_id] = startup_error
                    logger.error(
                        f"Camera {camera_id}: Startup failed early: {startup_error}"
                    )
                    break

                if capture_thread_exited:
                    error_detail = "Capture thread exited before receiving frames"
                    self.last_start_errors[camera_id] = error_detail
                    logger.error(f"Camera {camera_id}: {error_detail}")
                    break

            if connected:
                self.last_start_errors.pop(camera_id, None)
                # Update camera status to True in database only if successfully connected
                db = SessionLocal()
                try:
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if camera:
                        camera.status = True
                        db.commit()
                        logger.info(
                            f"Camera {camera_id} status set to True (successfully connected)"
                        )
                except Exception as db_error:
                    logger.error(
                        f"Failed to update camera {camera_id} status: {db_error}"
                    )
                    db.rollback()
                finally:
                    db.close()

                logger.info(f"Camera {camera_id} added and started successfully")
                return True
            else:
                # Check if failure is non-retryable (e.g. VSS rate limit)
                startup_error = manager.capture_thread.startup_error
                capture_thread = manager.capture_thread.thread
                capture_dead = (
                    capture_thread is not None and not capture_thread.is_alive()
                )

                if startup_error and _is_non_retryable_capture_error_str(startup_error):
                    # Truly non-retryable — remove camera
                    self.last_start_errors[camera_id] = startup_error
                    del self.cameras[camera_id]
                    manager.stop()
                    logger.error(
                        f"Camera {camera_id}: Non-retryable error, removed: {startup_error}"
                    )
                    return False

                # Retryable failure — keep camera alive so capture thread retries in background
                if startup_error:
                    self.last_start_errors[camera_id] = startup_error
                else:
                    self.last_start_errors.setdefault(
                        camera_id, "Camera not connected yet, retrying in background"
                    )
                logger.warning(
                    f"Camera {camera_id}: Not connected yet, keeping alive for background retry"
                )

                # Keep status=True in database so camera auto-starts on server restart
                db = SessionLocal()
                try:
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if camera and not camera.status:
                        camera.status = True
                        db.commit()
                except Exception as db_error:
                    logger.error(
                        f"Failed to update camera {camera_id} status: {db_error}"
                    )
                    db.rollback()
                finally:
                    db.close()

                return True

        except Exception as e:
            logger.error(f"Failed to add camera {camera_id}: {e}")
            # Clean up if camera was added to dict
            if camera_id in self.cameras:
                del self.cameras[camera_id]
            return False

    def get_last_start_error(self, camera_id: UUID) -> Optional[str]:
        """Return the most recent startup error for a camera, if any."""
        return self.last_start_errors.get(camera_id)

    def remove_camera(self, camera_id: UUID) -> bool:
        """Remove và stop camera"""
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not found")
            return False

        try:
            manager = self.cameras[camera_id]
            self.last_start_errors.pop(camera_id, None)

            # Then stop manager synchronously.
            stopped_cleanly = manager.stop()

            # Update camera status to False in database
            db = SessionLocal()
            try:
                camera = db.query(Camera).filter(Camera.id == camera_id).first()
                if camera:
                    camera.status = False
                    db.commit()
                    logger.info(f"Camera {camera_id} status set to False")
            except Exception as db_error:
                logger.error(f"Failed to update camera {camera_id} status: {db_error}")
                db.rollback()
            finally:
                db.close()

            if not stopped_cleanly:
                logger.error("Camera %s stop timeout", camera_id)
                # Keep manager registered so callers can retry stop and avoid
                # losing control of still-running worker threads.
                return False

            del self.cameras[camera_id]
            logger.info(f"Camera {camera_id} removed and stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to remove camera {camera_id}: {e}", exc_info=True)

            # Still try to update status even if removal failed
            try:
                db = SessionLocal()
                camera = db.query(Camera).filter(Camera.id == camera_id).first()
                if camera:
                    camera.status = False
                    db.commit()
                    logger.info(f"Camera {camera_id} status set to False (after error)")
                db.close()
            except:
                pass

            return False

    def start_camera(self, camera_id: UUID, show_display: bool = True) -> bool:
        """Start camera (nếu đã tồn tại)"""
        # Nếu camera đang chạy, stop trước khi start lại
        if camera_id in self.cameras:
            logger.info(f"Camera {camera_id} already running, stopping first...")
            success = self.remove_camera(camera_id)
            if not success:
                logger.error(f"Failed to stop existing camera {camera_id}")
                return False
            # Thêm delay dài hơn để đảm bảo cleanup hoàn tất
            logger.info(f"Waiting for camera {camera_id} to fully stop...")
            time.sleep(2.0)

        # Load camera info từ database
        db = SessionLocal()
        try:
            camera = db.query(Camera).filter(Camera.id == camera_id).first()
            if not camera:
                logger.error(f"Camera {camera_id} not found in database")
                return False

            logger.info(
                f"Starting camera {camera_id} with URL: {camera.rtsp_url}, preprocess: {camera.preprocess_resolution}"
            )
            return self.add_camera(
                camera_id, camera.rtsp_url, show_display, camera.preprocess_resolution
            )
        except Exception as e:
            logger.error(f"Error starting camera {camera_id}: {e}", exc_info=True)
            return False
        finally:
            db.close()

    def stop_camera(self, camera_id: UUID) -> bool:
        """Stop camera"""
        return self.remove_camera(camera_id)

    def reload_models(self, camera_id: UUID) -> bool:
        """Reload models cho camera"""
        if camera_id not in self.cameras:
            logger.warning(f"Camera {camera_id} not running")
            return False

        try:
            self.cameras[camera_id].reload_models()
            return True
        except Exception as e:
            logger.error(f"Failed to reload models for camera {camera_id}: {e}")
            return False

    def get_status(self) -> Dict:
        """Lấy trạng thái tất cả cameras"""
        return {
            "total_cameras": len(self.cameras),
            "cameras": [str(cam_id) for cam_id in self.cameras.keys()],
        }

    def load_all_active_cameras(
        self,
        show_display: bool = False,
        stagger_seconds: Optional[float] = None,
        retry_per_camera: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Graceful bring-up of every Camera with status=True.

        This is what the FastAPI lifespan hook calls on every service
        restart. Any camera that was online before the restart MUST come
        back on its own without a manual PUT /start.

        Guarantees:

        - **Per-camera isolation**: a bad RTSP URL, missing TensorRT
          engine, or broken model on one camera does NOT stop the loop
          from bringing the rest up. Each failure is caught, logged,
          and recorded in the returned summary.
        - **Stagger**: we wait ``stagger_seconds`` between consecutive
          camera starts. With 20 cameras, starting them all
          simultaneously caused a GPU / ffprobe / TensorRT thundering
          herd that could OOM CUDA or stall the capture thread pool.
          Default is tuned via env (``CAMERA_STARTUP_STAGGER_SEC``,
          default 1.5 s).
        - **Retry**: RTSP endpoints are frequently briefly unreachable
          during the first few seconds after an edge-device reboot
          (camera firmware still warming up, switch STP, etc.). Each
          camera gets ``retry_per_camera`` extra attempts with
          exponential backoff before it is recorded as failed. Default
          is tuned via env (``CAMERA_STARTUP_RETRIES``, default 2).
        - **Non-blocking caller**: this function can take tens of
          seconds on a 20-camera edge. Callers should run it in a
          background thread so /health is not gated on RTSP handshakes.

        Returns:
            dict with keys: ``total``, ``started``, ``failed``,
            ``failures`` (list of ``(camera_id, reason)``), and
            ``elapsed_seconds``.
        """
        if stagger_seconds is None:
            stagger_seconds = float(
                os.getenv("CAMERA_STARTUP_STAGGER_SEC", "1.5")
            )
        if retry_per_camera is None:
            retry_per_camera = int(os.getenv("CAMERA_STARTUP_RETRIES", "2"))

        start_time = time.time()

        db = SessionLocal()
        try:
            # Exclude soft-deleted cameras so a stale status=True flag on a
            # removed row never spins up an orphan worker on boot.
            cameras = (
                db.query(Camera)
                .filter(Camera.status == True)  # noqa: E712
                .filter(Camera.is_deleted == False)  # noqa: E712
                .all()
            )
        finally:
            db.close()

        total = len(cameras)
        if total == 0:
            logger.info(
                "Graceful startup: no active cameras to load "
                "(display=%s, stagger=%.1fs)",
                show_display,
                stagger_seconds,
            )
            return {
                "total": 0,
                "started": 0,
                "failed": 0,
                "failures": [],
                "elapsed_seconds": 0.0,
            }

        logger.info(
            "Graceful startup: bringing up %d active cameras "
            "(stagger=%.1fs, retries_per_camera=%d)",
            total,
            stagger_seconds,
            retry_per_camera,
        )

        started: List[str] = []
        failures: List[Tuple[str, str]] = []

        for idx, camera in enumerate(cameras):
            cid = str(camera.id)
            name = getattr(camera, "name", None) or cid[:8]
            attempts = retry_per_camera + 1
            succeeded = False
            last_error: Optional[str] = None

            for attempt in range(attempts):
                try:
                    ok = self.add_camera(
                        camera.id,
                        camera.rtsp_url,
                        show_display=show_display,
                        preprocess_resolution=camera.preprocess_resolution,
                    )
                    if ok:
                        succeeded = True
                        break
                    # add_camera already logged the cause and set
                    # last_start_errors[camera_id]; surface that to the
                    # retry decision.
                    last_error = (
                        self.last_start_errors.get(camera.id)
                        or "add_camera returned False"
                    )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    logger.error(
                        "Graceful startup: camera %s raised on attempt %d/%d: %s",
                        name,
                        attempt + 1,
                        attempts,
                        last_error,
                        exc_info=True,
                    )

                if attempt < attempts - 1:
                    # Exponential backoff: 2s, 4s, 8s, ...
                    backoff = 2.0 * (2 ** attempt)
                    logger.warning(
                        "Graceful startup [%d/%d] %s: attempt %d/%d failed (%s), "
                        "retrying in %.1fs",
                        idx + 1,
                        total,
                        name,
                        attempt + 1,
                        attempts,
                        last_error,
                        backoff,
                    )
                    # If add_camera half-registered the camera in self.cameras,
                    # clear it so the retry can re-register cleanly.
                    if camera.id in self.cameras:
                        try:
                            self.remove_camera(camera.id)
                        except Exception:
                            pass
                    time.sleep(backoff)

            if succeeded:
                started.append(cid)
                logger.info(
                    "Graceful startup [%d/%d]: %s (%s) up",
                    idx + 1,
                    total,
                    name,
                    cid[:8],
                )
            else:
                failures.append((cid, last_error or "unknown"))
                logger.error(
                    "Graceful startup [%d/%d]: %s (%s) gave up after %d attempts: %s",
                    idx + 1,
                    total,
                    name,
                    cid[:8],
                    attempts,
                    last_error,
                )

            # Stagger between cameras (skip after the last one).
            if idx < total - 1 and stagger_seconds > 0:
                time.sleep(stagger_seconds)

        elapsed = time.time() - start_time
        summary = {
            "total": total,
            "started": len(started),
            "failed": len(failures),
            "failures": failures,
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info(
            "Graceful startup complete: %d/%d cameras up, %d failed in %.1fs",
            summary["started"],
            summary["total"],
            summary["failed"],
            elapsed,
        )
        return summary

    def stop_all(self):
        """Dừng tất cả cameras"""
        logger.info("Stopping all cameras...")
        camera_ids = list(self.cameras.keys())
        for camera_id in camera_ids:
            self.remove_camera(camera_id)
        logger.info("All cameras stopped")

    def synchronize(self) -> Dict[str, Any]:
        """
        Synchronize camera states between database and running system.

        This method ensures:
        1. Cameras with status=True in database are running
        2. Cameras with status=False in database are stopped
        3. Database status is updated to match actual running state
        4. Models are reloaded for running cameras to ensure latest configuration

        Returns:
            Dict with synchronization results
        """
        logger.info("=" * 60)
        logger.info("SYSTEM SYNCHRONIZATION STARTED")
        logger.info("=" * 60)

        db = SessionLocal()
        sync_results = {
            "started": [],
            "stopped": [],
            "reloaded": [],
            "errors": [],
            "already_synced": [],
        }

        try:
            # Get all cameras from database
            all_cameras = db.query(Camera).all()
            logger.info(f"Found {len(all_cameras)} cameras in database")

            # Get currently running cameras
            running_camera_ids = set(self.cameras.keys())
            logger.info(f"Currently running cameras: {len(running_camera_ids)}")

            # Step 1: Start cameras that should be running (status=True) but aren't
            for camera in all_cameras:
                if camera.status and camera.id not in running_camera_ids:
                    logger.info(
                        f"Starting camera {camera.id} ({camera.name}) - status=True but not running"
                    )
                    try:
                        success = self.add_camera(
                            camera.id,
                            camera.rtsp_url,
                            show_display=False,
                            preprocess_resolution=camera.preprocess_resolution,
                        )
                        if success:
                            sync_results["started"].append(
                                {"camera_id": str(camera.id), "name": camera.name}
                            )
                            logger.info(f"✓ Started camera {camera.id} ({camera.name})")
                        else:
                            # Camera kept alive for background retry — don't flip status
                            sync_results["errors"].append(
                                {
                                    "camera_id": str(camera.id),
                                    "name": camera.name,
                                    "error": "Not connected yet, retrying in background",
                                }
                            )
                            logger.warning(
                                f"⟳ Camera {camera.id} ({camera.name}) not connected yet, retrying in background"
                            )
                    except Exception as e:
                        logger.error(
                            f"✗ Error starting camera {camera.id} ({camera.name}): {e}",
                            exc_info=True,
                        )
                        sync_results["errors"].append(
                            {
                                "camera_id": str(camera.id),
                                "name": camera.name,
                                "error": str(e),
                            }
                        )

            # Step 2: Stop cameras that shouldn't be running (status=False) but are
            for camera in all_cameras:
                if not camera.status and camera.id in running_camera_ids:
                    logger.info(
                        f"Stopping camera {camera.id} ({camera.name}) - status=False but running"
                    )
                    try:
                        success = self.remove_camera(camera.id)
                        if success:
                            sync_results["stopped"].append(
                                {"camera_id": str(camera.id), "name": camera.name}
                            )
                            logger.info(f"✓ Stopped camera {camera.id} ({camera.name})")
                        else:
                            sync_results["errors"].append(
                                {
                                    "camera_id": str(camera.id),
                                    "name": camera.name,
                                    "error": "Failed to stop camera",
                                }
                            )
                            logger.error(
                                f"✗ Failed to stop camera {camera.id} ({camera.name})"
                            )
                    except Exception as e:
                        logger.error(
                            f"✗ Error stopping camera {camera.id} ({camera.name}): {e}",
                            exc_info=True,
                        )
                        sync_results["errors"].append(
                            {
                                "camera_id": str(camera.id),
                                "name": camera.name,
                                "error": str(e),
                            }
                        )

            # Step 3: Reload models for cameras that are running (to ensure they have latest config)
            for camera_id in list(self.cameras.keys()):
                try:
                    logger.info(f"Reloading models for camera {camera_id}")
                    self.cameras[camera_id].reload_models()
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if camera:
                        sync_results["reloaded"].append(
                            {"camera_id": str(camera_id), "name": camera.name}
                        )
                        logger.info(
                            f"✓ Reloaded models for camera {camera_id} ({camera.name})"
                        )
                except Exception as e:
                    logger.error(
                        f"✗ Error reloading models for camera {camera_id}: {e}",
                        exc_info=True,
                    )
                    camera = db.query(Camera).filter(Camera.id == camera_id).first()
                    if camera:
                        sync_results["errors"].append(
                            {
                                "camera_id": str(camera_id),
                                "name": camera.name,
                                "error": f"Model reload failed: {str(e)}",
                            }
                        )

            # Step 4: Update database status to match actual running state
            running_camera_ids_after = set(self.cameras.keys())
            for camera in all_cameras:
                should_be_running = camera.id in running_camera_ids_after
                if camera.status != should_be_running:
                    logger.info(
                        f"Updating camera {camera.id} ({camera.name}) status: {camera.status} -> {should_be_running}"
                    )
                    camera.status = should_be_running
                    db.commit()

            # Count cameras that were already in sync (before any changes)
            for camera in all_cameras:
                was_in_sync = (camera.status and camera.id in running_camera_ids) or (
                    not camera.status and camera.id not in running_camera_ids
                )
                if was_in_sync:
                    sync_results["already_synced"].append(
                        {"camera_id": str(camera.id), "name": camera.name}
                    )

            logger.info("=" * 60)
            logger.info("SYSTEM SYNCHRONIZATION COMPLETED")
            logger.info(f"Started: {len(sync_results['started'])}")
            logger.info(f"Stopped: {len(sync_results['stopped'])}")
            logger.info(f"Reloaded: {len(sync_results['reloaded'])}")
            logger.info(f"Already synced: {len(sync_results['already_synced'])}")
            logger.info(f"Errors: {len(sync_results['errors'])}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Critical error during synchronization: {e}", exc_info=True)
            sync_results["errors"].append(
                {"error": f"Critical synchronization error: {str(e)}"}
            )
            db.rollback()
        finally:
            db.close()

        return sync_results


# Global instance
thread_manager = ThreadManager()
