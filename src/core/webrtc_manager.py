"""
WebRTC Manager - handles WebRTC peer connections for low-latency video streaming.

Uses aiortc to create PeerConnections that push buffered camera frames as video tracks.
By default, WebRTC uses the latest raw frame plus a short-lived overlay so the
stream stays smooth like the source feed while stale boxes are dropped quickly.
"""

import asyncio
import logging
import os
import time
import uuid as _uuid
from fractions import Fraction
from typing import Dict, Optional

import numpy as np

try:
    import av
    from aiortc import (
        MediaStreamTrack,
        RTCPeerConnection,
        RTCConfiguration,
        RTCIceServer,
        RTCSessionDescription,
    )
    AIORTC_AVAILABLE = True
except ImportError:
    AIORTC_AVAILABLE = False

from src.core.frame_buffer import frame_buffer_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
WEBRTC_ENABLED = os.getenv("WEBRTC_ENABLED", "true").lower() == "true"
WEBRTC_STUN_SERVER = os.getenv("WEBRTC_STUN_SERVER", "stun:stun.l.google.com:19302")
WEBRTC_TURN_SERVER = os.getenv("WEBRTC_TURN_SERVER", "")
WEBRTC_TURN_USERNAME = os.getenv("WEBRTC_TURN_USERNAME", "")
WEBRTC_TURN_PASSWORD = os.getenv("WEBRTC_TURN_PASSWORD", "")
WEBRTC_DEFAULT_FPS = int(os.getenv("WEBRTC_DEFAULT_FPS", "30"))
WEBRTC_FRAME_SOURCE = os.getenv("WEBRTC_FRAME_SOURCE", "overlay").strip().lower()

# Allowed FPS range (reasonable defaults for streaming)
ALLOWED_FPS_RANGE = (1, 60)


def is_fps_allowed(fps: int) -> bool:
    return ALLOWED_FPS_RANGE[0] <= fps <= ALLOWED_FPS_RANGE[1]


def _build_rtc_config() -> "RTCConfiguration":
    """Build ICE server list from environment variables."""
    ice_servers = []
    if WEBRTC_STUN_SERVER:
        ice_servers.append(RTCIceServer(urls=[WEBRTC_STUN_SERVER]))
    if WEBRTC_TURN_SERVER:
        ice_servers.append(
            RTCIceServer(
                urls=[WEBRTC_TURN_SERVER],
                username=WEBRTC_TURN_USERNAME,
                credential=WEBRTC_TURN_PASSWORD,
            )
        )
    return RTCConfiguration(iceServers=ice_servers)


# ---------------------------------------------------------------------------
# Custom VideoStreamTrack that reads from FrameBufferManager
# ---------------------------------------------------------------------------
if AIORTC_AVAILABLE:
    class FrameVideoTrack(MediaStreamTrack):
        """Serve buffered frames for a single camera."""

        kind = "video"

        def __init__(self, camera_id, fps: int = 0):
            super().__init__()
            self.camera_id = camera_id
            # fps=0 means auto-detect from RTSP source
            self._requested_fps = fps
            self.fps = fps if fps and is_fps_allowed(fps) else WEBRTC_DEFAULT_FPS
            self._interval = 1.0 / self.fps
            self._next_tick: Optional[float] = None
            self._frame_count = 0
            self._fps_resolved = False
            self._placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            self._last_frame = self._placeholder
            self._frame_source = self._resolve_frame_source()
            logger.info(
                "WebRTC track for %s using frame source '%s'",
                str(self.camera_id)[:8],
                self._frame_source,
            )

        def _resolve_frame_source(self) -> str:
            if WEBRTC_FRAME_SOURCE in {"annotated", "overlay", "raw"}:
                return WEBRTC_FRAME_SOURCE
            logger.warning(
                "Unknown WEBRTC_FRAME_SOURCE='%s', falling back to 'annotated'",
                WEBRTC_FRAME_SOURCE,
            )
            return "annotated"

        def _resolve_fps(self) -> None:
            """Try to match the RTSP source FPS (called once)."""
            src_fps = frame_buffer_manager.get_source_fps(self.camera_id)
            if src_fps > 0 and is_fps_allowed(int(src_fps)):
                self.fps = int(src_fps)
                self._interval = 1.0 / self.fps
                logger.info(
                    "WebRTC track for %s: matched source FPS %s",
                    str(self.camera_id)[:8],
                    self.fps,
                )
            self._fps_resolved = True

        def _get_next_buffered_frame(self):
            if self._frame_source == "raw":
                getters = (
                    frame_buffer_manager.get_raw,
                    frame_buffer_manager.get,
                    frame_buffer_manager.get_overlay_frame,
                )
            elif self._frame_source == "overlay":
                getters = (
                    frame_buffer_manager.get_overlay_frame,
                    frame_buffer_manager.get,
                    frame_buffer_manager.get_raw,
                )
            else:
                # Default: bbox must match the AI frame being displayed.
                getters = (
                    frame_buffer_manager.get,
                    frame_buffer_manager.get_raw,
                    frame_buffer_manager.get_overlay_frame,
                )

            for getter in getters:
                result = getter(self.camera_id)
                if result is not None:
                    return result
            return None

        async def recv(self):
            """Return the next video frame at the source FPS pace."""
            if self._next_tick is None:
                self._next_tick = time.perf_counter()
                if not self._fps_resolved and not self._requested_fps:
                    self._resolve_fps()

            wait = self._next_tick - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)

            self._next_tick += self._interval
            if time.perf_counter() - self._next_tick > (self._interval * 2):
                self._next_tick = time.perf_counter() + self._interval

            self._frame_count += 1
            result = self._get_next_buffered_frame()

            if result is not None:
                bgr_frame, _ts = result
                self._last_frame = bgr_frame
            else:
                bgr_frame = self._last_frame

            try:
                video_frame = av.VideoFrame.from_ndarray(bgr_frame, format="bgr24")
            except Exception:
                rgb_frame = bgr_frame[:, :, ::-1].copy()
                video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")

            video_frame.pts = self._frame_count
            video_frame.time_base = Fraction(1, self.fps)
            return video_frame


# ---------------------------------------------------------------------------
# WebRTCManager
# ---------------------------------------------------------------------------
class WebRTCManager:
    """Manage active WebRTC peer connections."""

    def __init__(self):
        self._connections: Dict[str, Dict] = {}
        logger.info("WebRTCManager initialized (aiortc available: %s)", AIORTC_AVAILABLE)

    async def create_connection(
        self,
        camera_id,
        sdp_offer: str,
        sdp_type: str = "offer",
        fps: int = WEBRTC_DEFAULT_FPS,
    ) -> Dict:
        """
        Accept a client SDP offer, create a PeerConnection with a video track,
        and return the SDP answer plus connection metadata.
        """
        if not AIORTC_AVAILABLE:
            raise RuntimeError("aiortc is not installed. Install with: pip install aiortc")

        if not WEBRTC_ENABLED:
            raise RuntimeError("WebRTC is disabled (set WEBRTC_ENABLED=true)")

        conn_id = str(_uuid.uuid4())
        pc = RTCPeerConnection(configuration=_build_rtc_config())

        video_track = FrameVideoTrack(camera_id, fps=fps)
        pc.addTrack(video_track)

        @pc.on("connectionstatechange")
        async def on_state_change():
            state = pc.connectionState
            logger.info("WebRTC [%s] state: %s", conn_id[:8], state)
            if state in ("failed", "closed"):
                await self._cleanup_connection(conn_id)

        offer = RTCSessionDescription(sdp=sdp_offer, type=sdp_type)
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self._connections[conn_id] = {
            "pc": pc,
            "camera_id": str(camera_id),
            "fps": video_track.fps,
            "created_at": time.time(),
        }

        logger.info(
            "WebRTC [%s] created for camera %s at %s FPS",
            conn_id[:8],
            camera_id,
            video_track.fps,
        )

        return {
            "connection_id": conn_id,
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "camera_id": str(camera_id),
            "fps": video_track.fps,
        }

    async def close_connection(self, conn_id: str) -> bool:
        """Close a specific peer connection."""
        return await self._cleanup_connection(conn_id)

    def get_connections(self) -> list:
        """Return metadata for all active connections."""
        result = []
        for conn_id, info in self._connections.items():
            result.append({
                "connection_id": conn_id,
                "camera_id": info["camera_id"],
                "fps": info["fps"],
                "created_at": info["created_at"],
                "state": info["pc"].connectionState if info["pc"] else "unknown",
            })
        return result

    async def close_camera_connections(self, camera_id) -> int:
        """Close all connections for a specific camera. Returns count closed."""
        camera_id_str = str(camera_id)
        to_close = [
            conn_id
            for conn_id, info in self._connections.items()
            if info["camera_id"] == camera_id_str
        ]
        for conn_id in to_close:
            await self._cleanup_connection(conn_id)
        if to_close:
            logger.info(
                "Closed %d WebRTC connection(s) for camera %s",
                len(to_close),
                camera_id_str[:8],
            )
        return len(to_close)

    def close_camera_connections_sync(self, camera_id) -> None:
        """Synchronous wrapper for close_camera_connections (safe from non-async code)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # Already in an async context — schedule and don't block
            asyncio.ensure_future(self.close_camera_connections(camera_id))
        except RuntimeError:
            # No running loop — create one
            asyncio.run(self.close_camera_connections(camera_id))

    async def close_all(self):
        """Close every active connection (used during shutdown)."""
        ids = list(self._connections.keys())
        for conn_id in ids:
            await self._cleanup_connection(conn_id)

    async def _cleanup_connection(self, conn_id: str) -> bool:
        info = self._connections.pop(conn_id, None)
        if info is None:
            return False
        try:
            await info["pc"].close()
        except Exception as exc:
            logger.warning("WebRTC [%s] close error: %s", conn_id[:8], exc)
        logger.info("WebRTC [%s] closed", conn_id[:8])
        return True


webrtc_manager = WebRTCManager()
