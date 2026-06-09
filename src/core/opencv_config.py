"""
src/core/opencv_config.py
Configure OpenCV FFmpeg backend BEFORE any ``import cv2`` in the process.

OpenCV reads ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` exactly once — when the FFmpeg
backend is first initialised (i.e. on the first ``cv2.VideoCapture``).  Any
changes after that are silently ignored.  This module must therefore be
imported before **any** module that does ``import cv2``.

The previous location (top of ``thread_manager.py``) was too late because
``capture_backends.py`` — which also ``import cv2`` at module level — is
imported by ``src.api.camera`` before ``thread_manager``.

Environment knobs (all optional):
    RTSP_SOCKET_TIMEOUT_US  – FFmpeg ``stimeout`` in µs  (default 5 000 000 = 5 s)
    RTSP_MAX_DELAY_US       – FFmpeg ``max_delay`` in µs (default   500 000 = 0.5 s)
"""

import os

_RTSP_SOCKET_TIMEOUT_US = os.environ.setdefault("RTSP_SOCKET_TIMEOUT_US", "5000000")
_RTSP_MAX_DELAY_US = os.environ.setdefault("RTSP_MAX_DELAY_US", "500000")

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    (
        f"rtsp_transport;tcp"
        f"|stimeout;{_RTSP_SOCKET_TIMEOUT_US}"
        f"|max_delay;{_RTSP_MAX_DELAY_US}"
        f"|reorder_queue_size;0"
        f"|fflags;nobuffer"
        f"|flags;low_delay"
    ),
)
