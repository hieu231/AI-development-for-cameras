"""
Frame Buffer Manager - centralised per-camera latest-frame store.

Provides the single source of truth for annotated frames consumed by
MJPEG, WebRTC, and RTSP output pipelines.  Only the *most recent* frame
is kept per camera (no queue) to minimise latency.
"""

import time
import os
import logging
import threading
from uuid import UUID
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from src.utils.image_utils import make_cv2_safe_text

logger = logging.getLogger(__name__)
STREAM_OVERLAY_MAX_AGE_MS = max(0, int(os.getenv("STREAM_OVERLAY_MAX_AGE_MS", "120")))
STREAM_TRACKING_MAX_AGE_MS = max(0, int(os.getenv("STREAM_TRACKING_MAX_AGE_MS", "250")))
STREAM_TRACK_MAX_DETECTIONS = max(1, int(os.getenv("STREAM_TRACK_MAX_DETECTIONS", "20")))
STREAM_TRACK_SEARCH_EXPAND = max(0.1, float(os.getenv("STREAM_TRACK_SEARCH_EXPAND", "0.6")))
STREAM_TRACK_MIN_SCORE = max(0.0, min(float(os.getenv("STREAM_TRACK_MIN_SCORE", "0.45")), 1.0))



class _CameraBuffer:
    """Internal per-camera frame holder with its own lock."""

    __slots__ = (
        "frame", "timestamp",
        "raw_frame", "raw_timestamp",
        "source_fps",
        "_overlay_mask", "_overlay_pixels", "_overlay_ann_ts", "_overlay_base_frame",
        "_stream_detections",
        "lock",
    )

    def __init__(self) -> None:
        self.frame: Optional[np.ndarray] = None
        self.timestamp: float = 0.0
        self.raw_frame: Optional[np.ndarray] = None
        self.raw_timestamp: float = 0.0
        self.source_fps: float = 0.0
        # Pre-computed overlay cache (recomputed only when annotated frame changes)
        self._overlay_mask: Optional[np.ndarray] = None   # boolean mask
        self._overlay_pixels: Optional[np.ndarray] = None  # pixel values at mask positions
        self._overlay_ann_ts: float = 0.0                  # timestamp of cached annotated frame
        self._overlay_base_frame: Optional[np.ndarray] = None  # base frame matched with overlay
        self._stream_detections: list[dict] = []
        self.lock = threading.Lock()

    def update(self, frame: np.ndarray, timestamp: float,
               base_frame: 'Optional[np.ndarray]' = None,
               detections: 'Optional[list[dict]]' = None) -> None:
        with self.lock:
            self.frame = frame
            self.timestamp = timestamp
            self._stream_detections = list(detections or [])
            # Recompute overlay cache when annotated frame changes
            self._recompute_overlay(base_frame)

    def update_raw(self, frame: np.ndarray, timestamp: float) -> None:
        """Store the raw (unannotated) RTSP frame."""
        with self.lock:
            self.raw_frame = frame
            self.raw_timestamp = timestamp

    def _recompute_overlay(self, base_frame: 'Optional[np.ndarray]' = None) -> None:
        """Pre-compute the diff mask between base and annotated frames.

        Called inside the lock whenever the annotated frame is updated.
        *base_frame* is the original unannotated frame that the AI model
        processed — using it (instead of the latest raw_frame) guarantees
        the diff is always between matching frames.
        """
        src = base_frame if base_frame is not None else self.raw_frame
        if src is None or self.frame is None:
            self._overlay_mask = None
            self._overlay_pixels = None
            self._overlay_base_frame = None
            return
        if src.shape != self.frame.shape:
            self._overlay_mask = None
            self._overlay_pixels = None
            self._overlay_base_frame = None
            return
        diff = cv2.absdiff(src, self.frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        self._overlay_mask = gray > 10
        self._overlay_pixels = self.frame[self._overlay_mask].copy()
        self._overlay_base_frame = src.copy()
        self._overlay_ann_ts = self.timestamp

    def get(self) -> Optional[Tuple[np.ndarray, float]]:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy(), self.timestamp

    def get_raw(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return ``(raw_frame_copy, timestamp)`` or ``None``."""
        with self.lock:
            if self.raw_frame is None:
                return None
            return self.raw_frame.copy(), self.raw_timestamp

    def get_overlay_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """Return a stable display frame for streaming.

        Takes the latest raw frame and composites the most recent bbox/ROI
        overlay mask on top.  To avoid obviously stale boxes, the overlay is
        dropped once it gets older than STREAM_OVERLAY_MAX_AGE_MS.
        """
        with self.lock:
            if self.raw_frame is not None:
                result = self.raw_frame.copy()
                overlay_age_ms = (self.raw_timestamp - self._overlay_ann_ts) * 1000.0
                tracked = self._render_tracked_detections(result, overlay_age_ms)
                if tracked is not None:
                    return tracked, self.raw_timestamp
                if (
                    self._overlay_mask is not None
                    and self._overlay_pixels is not None
                    and self.frame is not None
                    and result.shape == self.frame.shape
                    and overlay_age_ms <= STREAM_OVERLAY_MAX_AGE_MS
                ):
                    result[self._overlay_mask] = self._overlay_pixels
                return result, self.raw_timestamp

            if self.frame is not None:
                return self.frame.copy(), self.timestamp
            return None

    def _render_tracked_detections(
        self,
        target_frame: np.ndarray,
        overlay_age_ms: float,
    ) -> Optional[np.ndarray]:
        if (
            not self._stream_detections
            or self._overlay_base_frame is None
            or overlay_age_ms > STREAM_TRACKING_MAX_AGE_MS
            or self._overlay_base_frame.shape != target_frame.shape
        ):
            return None

        base_gray = cv2.cvtColor(self._overlay_base_frame, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        drawn = 0

        for detection in self._stream_detections[:STREAM_TRACK_MAX_DETECTIONS]:
            bbox = detection.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            tracked_bbox = self._track_bbox(base_gray, target_gray, bbox)
            self._draw_detection(target_frame, detection, tracked_bbox)
            drawn += 1

        if drawn == 0:
            return None
        return target_frame

    def _track_bbox(
        self,
        base_gray: np.ndarray,
        target_gray: np.ndarray,
        bbox: list,
    ) -> tuple[int, int, int, int]:
        height, width = target_gray.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 8 or box_h < 8:
            return x1, y1, x2, y2

        template = base_gray[y1:y2, x1:x2]
        if template.size == 0:
            return x1, y1, x2, y2

        pad_x = max(6, int(box_w * STREAM_TRACK_SEARCH_EXPAND))
        pad_y = max(6, int(box_h * STREAM_TRACK_SEARCH_EXPAND))
        sx1 = max(0, x1 - pad_x)
        sy1 = max(0, y1 - pad_y)
        sx2 = min(width, x2 + pad_x)
        sy2 = min(height, y2 + pad_y)
        search = target_gray[sy1:sy2, sx1:sx2]

        if search.shape[0] < box_h or search.shape[1] < box_w:
            return x1, y1, x2, y2

        match = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, max_score, _, max_loc = cv2.minMaxLoc(match)
        if max_score < STREAM_TRACK_MIN_SCORE:
            return x1, y1, x2, y2

        nx1 = sx1 + max_loc[0]
        ny1 = sy1 + max_loc[1]
        nx2 = nx1 + box_w
        ny2 = ny1 + box_h
        return nx1, ny1, nx2, ny2

    def _draw_detection(
        self,
        frame: np.ndarray,
        detection: dict,
        bbox: tuple[int, int, int, int],
    ) -> None:
        x1, y1, x2, y2 = bbox
        label = str(
            detection.get("label")
            or detection.get("class_name")
            or detection.get("model_name")
            or "object"
        )
        confidence = detection.get("confidence")
        track_id = detection.get("track_id")

        if track_id is not None:
            label = f"{label} ID:{track_id}"
        if confidence is not None:
            try:
                label = f"{label} {float(confidence):.2f}"
            except (TypeError, ValueError):
                pass
        label = make_cv2_safe_text(label, fallback="object")

        color = self._label_color(label)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top = max(0, y1 - th - 10)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 8, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _label_color(self, label: str) -> tuple[int, int, int]:
        seed = abs(hash(label))
        return (
            80 + (seed % 140),
            80 + ((seed // 17) % 140),
            80 + ((seed // 31) % 140),
        )

    def is_online(self, timeout: float = 3.0) -> bool:
        with self.lock:
            if self.frame is None:
                return False
            return (time.time() - self.timestamp) <= timeout


class FrameBufferManager:
    """
    Singleton managing latest annotated frames for every active camera.

    Usage::

        from src.core.frame_buffer import frame_buffer_manager

        # Writer side (ProcessorThread)
        frame_buffer_manager.update(cam_id, annotated_frame, time.time())

        # Reader side (WebRTC / MJPEG / RTSP output)
        result = frame_buffer_manager.get(cam_id)
        if result:
            frame, ts = result
    """

    _instance: Optional["FrameBufferManager"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "FrameBufferManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._buffers = {}
                    inst._map_lock = threading.Lock()
                    cls._instance = inst
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, camera_id: UUID, frame: np.ndarray, timestamp: float,
               base_frame: 'Optional[np.ndarray]' = None,
               detections: 'Optional[list[dict]]' = None) -> None:
        """Overwrite the latest frame for *camera_id*.

        *base_frame*: optional original unannotated frame used to compute
        the bounding-box overlay mask.  When provided, the diff is always
        between the exact frame the AI model processed and its annotated
        output, avoiding timing mismatches with the raw buffer.
        """
        buf = self._get_or_create(camera_id)
        buf.update(frame, timestamp, base_frame, detections)

    def update_raw(self, camera_id: UUID, frame: np.ndarray, timestamp: float) -> None:
        """Overwrite the latest *raw* (unannotated) frame for *camera_id*."""
        buf = self._get_or_create(camera_id)
        buf.update_raw(frame, timestamp)

    def get(self, camera_id: UUID) -> Optional[Tuple[np.ndarray, float]]:
        """Return ``(frame_copy, timestamp)`` or ``None``."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return None
        return buf.get()

    def get_raw(self, camera_id: UUID) -> Optional[Tuple[np.ndarray, float]]:
        """Return ``(raw_frame_copy, timestamp)`` or ``None``."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return None
        return buf.get_raw()

    def get_overlay_frame(self, camera_id: UUID) -> Optional[Tuple[np.ndarray, float]]:
        """Return raw frame with pre-computed bounding-box overlay, or ``None``."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return None
        return buf.get_overlay_frame()

    def update_source_fps(self, camera_id: UUID, fps: float) -> None:
        """Store the source RTSP FPS for *camera_id*."""
        buf = self._get_or_create(camera_id)
        with buf.lock:
            buf.source_fps = fps

    def get_source_fps(self, camera_id: UUID) -> float:
        """Return stored source FPS (0.0 if unknown)."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return 0.0
        with buf.lock:
            return buf.source_fps

    def get_frame(self, camera_id: UUID) -> Optional[np.ndarray]:
        """Convenience: return only the frame, or ``None``."""
        result = self.get(camera_id)
        return result[0] if result else None

    def is_online(self, camera_id: UUID, timeout: float = 3.0) -> bool:
        """``True`` if *camera_id* has been updated within *timeout* seconds."""
        buf = self._buffers.get(camera_id)
        if buf is None:
            return False
        return buf.is_online(timeout)

    def remove(self, camera_id: UUID) -> None:
        """Remove buffer entry when camera is stopped."""
        with self._map_lock:
            self._buffers.pop(camera_id, None)
        logger.info(f"FrameBuffer: removed buffer for camera {camera_id}")

    def list_cameras(self) -> list:
        """Return list of camera IDs currently in the buffer."""
        with self._map_lock:
            return list(self._buffers.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, camera_id: UUID) -> _CameraBuffer:
        buf = self._buffers.get(camera_id)
        if buf is not None:
            return buf
        with self._map_lock:
            # Double-check after acquiring lock
            buf = self._buffers.get(camera_id)
            if buf is None:
                buf = _CameraBuffer()
                self._buffers[camera_id] = buf
                logger.info(f"FrameBuffer: created buffer for camera {camera_id}")
            return buf


# Module-level singleton
frame_buffer_manager = FrameBufferManager()
