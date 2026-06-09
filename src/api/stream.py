"""
Video Streaming API - MJPEG streaming and frame buffer status
"""
import cv2
import logging
import subprocess
import threading
import time
import os
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session
from uuid import UUID

from src.database import get_db
from src.models.camera import Camera
from src.core.thread_manager import thread_manager
from src.core.frame_buffer import frame_buffer_manager
from src.core.webrtc_manager import WEBRTC_ENABLED, AIORTC_AVAILABLE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/stream", tags=["stream"])

MJPEG_DEFAULT_FPS = int(os.getenv("MJPEG_DEFAULT_FPS", "30"))
MJPEG_MAX_FPS = int(os.getenv("MJPEG_MAX_FPS", "60"))
MJPEG_DEFAULT_QUALITY = int(os.getenv("MJPEG_JPEG_QUALITY", "85"))

# Shared FFmpeg binary path cache (used by fMP4 stream and ws_video_proxy)
_WS_VIDEO_FFMPEG_PATH: str | None = None


def _clamp_mjpeg_fps(fps: int) -> int:
    return max(1, min(fps, MJPEG_MAX_FPS))


def _encode_snapshot_jpeg(frame: np.ndarray, quality: int) -> bytes:
    jpeg_quality = max(50, min(int(quality), 95))
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG frame")
    return buffer.tobytes()


def _build_snapshot_placeholder() -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        "Waiting for stream...",
        (175, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )
    return frame


def _build_stream_links(camera_id, base_url: str) -> dict:
    """Build stream links dict for a camera."""
    cam_str = str(camera_id)
    return {
        "camera_id": cam_str,
        "is_running": camera_id in thread_manager.cameras,
        "mjpeg": f"{base_url}/api/stream/camera/{cam_str}/mjpeg",
        "video": f"{base_url}/api/stream/camera/{cam_str}/video",
        "webrtc": {
            "offer_url": f"{base_url}/api/webrtc/offer",
            "play_url": f"{base_url}/api/webrtc/play/{cam_str}",
            "enabled": WEBRTC_ENABLED and AIORTC_AVAILABLE,
        },
        "websocket_events": f"ws://{base_url.split('://')[-1]}/ws/events/{cam_str}",
    }


@router.get("/links")
async def get_all_stream_links(request: Request, db: Session = Depends(get_db)):
    """
    Trả về stream links cho TẤT CẢ camera trong hệ thống.
    Frontend dùng endpoint này để biết URL kết nối stream.
    """
    base_url = str(request.base_url).rstrip("/")
    cameras = db.query(Camera).all()

    return {
        "total": len(cameras),
        "cameras": [_build_stream_links(cam.id, base_url) for cam in cameras],
    }


@router.get("/camera/{camera_id}/links")
async def get_camera_stream_links(camera_id: UUID, request: Request):
    """
    Trả về stream links cho 1 camera cụ thể.
    Bao gồm: MJPEG URL, WebRTC offer URL, WebSocket events URL.
    """
    base_url = str(request.base_url).rstrip("/")
    return _build_stream_links(camera_id, base_url)


def generate_mjpeg_stream(
    camera_id: UUID,
    target_fps: int = MJPEG_DEFAULT_FPS,
    jpeg_quality: int = MJPEG_DEFAULT_QUALITY,
):
    """Generate MJPEG stream from annotated frames via frame buffer"""
    fps = _clamp_mjpeg_fps(target_fps)
    src_fps = frame_buffer_manager.get_source_fps(camera_id)
    if 1 <= src_fps <= 120:
        fps = min(fps, max(1, int(src_fps)))
    interval = 1.0 / fps
    quality = max(50, min(jpeg_quality, 95))
    next_tick = time.perf_counter()
    last_ts = -1.0
    last_payload: bytes | None = None

    while True:
        try:
            # Stop streaming if camera was removed
            if camera_id not in thread_manager.cameras:
                logger.info(f"MJPEG stream for camera {camera_id}: camera stopped, ending stream")
                return

            # Read high-rate frame (raw + latest overlay) first for smoother stream
            result = frame_buffer_manager.get_overlay_frame(camera_id)
            if result is None:
                # Fallback to fully-annotated frame if overlay frame not ready
                result = frame_buffer_manager.get(camera_id)

            if result is not None:
                frame, frame_ts = result
            else:
                # Buffer empty (brief race at startup) — stream stays online with placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_ts = 0.0
                cv2.putText(
                    frame,
                    "Connecting...",
                    (220, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

            # Reuse last encoded JPEG when frame timestamp is unchanged.
            if last_payload is None or frame_ts != last_ts:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ret:
                    frame_bytes = buffer.tobytes()
                    last_payload = (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                    last_ts = frame_ts
                elif last_payload is None:
                    continue

            if last_payload is not None:
                yield last_payload

            # Monotonic pacing avoids jitter when system clock changes.
            next_tick += interval
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -(interval * 2):
                next_tick = time.perf_counter()

        except GeneratorExit:
            logger.info(f"MJPEG stream for camera {camera_id}: client disconnected")
            return
        except Exception as e:
            logger.error(f"Error generating MJPEG stream for camera {camera_id}: {e}")
            time.sleep(0.5)  # Wait before retrying


@router.get("/camera/{camera_id}/mjpeg")
async def stream_camera_mjpeg(
    camera_id: UUID,
    fps: int = MJPEG_DEFAULT_FPS,
    quality: int = MJPEG_DEFAULT_QUALITY,
):
    """
    Stream annotated camera feed as MJPEG (Motion JPEG)
    
    This endpoint streams the latest annotated frames with bounding boxes
    in real-time. Use this URL in an <img> tag or video player that supports MJPEG.
    
    Example:
        <img src="http://192.168.1.15:8668/api/stream/camera/{camera_id}/mjpeg" />
    """
    # Check if camera is running
    if camera_id not in thread_manager.cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} is not running. Start the camera first."
        )

    return StreamingResponse(
        generate_mjpeg_stream(
            camera_id,
            target_fps=fps,
            jpeg_quality=quality,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# fMP4 Video Stream — H.264 encoded media video via FFmpeg
# ---------------------------------------------------------------------------

FMP4_DEFAULT_FPS = int(os.getenv("FMP4_DEFAULT_FPS", "15"))
FMP4_DEFAULT_CRF = int(os.getenv("FMP4_DEFAULT_CRF", "23"))


def _get_ffmpeg_path_cached() -> str:
    """Return cached FFmpeg binary path."""
    global _WS_VIDEO_FFMPEG_PATH
    if _WS_VIDEO_FFMPEG_PATH is None:
        import shutil as _shutil
        path = os.getenv("FFMPEG_PATH") or _shutil.which("ffmpeg")
        if not path:
            raise RuntimeError("ffmpeg not found. Set FFMPEG_PATH env var.")
        _WS_VIDEO_FFMPEG_PATH = path
    return _WS_VIDEO_FFMPEG_PATH


def generate_fmp4_video_stream(
    camera_id: UUID,
    target_fps: int = FMP4_DEFAULT_FPS,
    crf: int = FMP4_DEFAULT_CRF,
):
    """Generate an fMP4 (H.264) video stream from annotated frames via FFmpeg.

    Pipes raw BGR frames from FrameBufferManager into FFmpeg which encodes
    to H.264 and outputs fragmented MP4 chunks on stdout.  The result is a
    standards-compliant video stream playable in any browser ``<video>`` tag.
    """
    fps = max(1, min(target_fps, 60))
    crf = max(0, min(crf, 51))
    interval = 1.0 / fps

    # Wait for first frame to learn dimensions
    first_frame = None
    for _ in range(50):  # ~10 seconds max
        result = frame_buffer_manager.get_overlay_frame(camera_id)
        if result is None:
            result = frame_buffer_manager.get(camera_id)
        if result is not None:
            first_frame, _ = result
            break
        time.sleep(0.2)

    if first_frame is None:
        logger.warning("fMP4 stream for camera %s: no frames available", camera_id)
        return

    h, w = first_frame.shape[:2]
    ffmpeg_path = _get_ffmpeg_path_cached()

    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", str(crf),
        "-g", str(fps * 2),   # key-frame every 2 seconds
        "-bf", "0",
        "-f", "mp4",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "pipe:1",
    ]

    logger.info(
        "fMP4 stream for camera %s: starting FFmpeg (%dx%d @ %d fps, crf %d)",
        camera_id, w, h, fps, crf,
    )

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 7,
    )

    # Background thread: feed annotated frames → FFmpeg stdin
    feed_stop = threading.Event()

    def _feed_frames():
        next_tick = time.perf_counter()
        try:
            while not feed_stop.is_set():
                # Stop if camera removed
                if camera_id not in thread_manager.cameras:
                    break

                result = frame_buffer_manager.get_overlay_frame(camera_id)
                if result is None:
                    result = frame_buffer_manager.get(camera_id)

                if result is not None:
                    bgr_frame, _ = result
                    fh, fw = bgr_frame.shape[:2]
                    if (fh, fw) != (h, w):
                        bgr_frame = cv2.resize(bgr_frame, (w, h))
                    try:
                        proc.stdin.write(bgr_frame.tobytes())
                    except (BrokenPipeError, OSError):
                        break

                # Pace at target FPS
                next_tick += interval
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -(interval * 2):
                    next_tick = time.perf_counter()
        except Exception as e:
            logger.warning("fMP4 feed thread error for camera %s: %s", camera_id, e)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    feeder = threading.Thread(target=_feed_frames, daemon=True,
                              name=f"fmp4-feed-{str(camera_id)[:8]}")
    feeder.start()

    # Yield fMP4 chunks from FFmpeg stdout
    try:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            yield chunk
    except GeneratorExit:
        logger.info("fMP4 stream for camera %s: client disconnected", camera_id)
    except Exception as e:
        logger.error("fMP4 stream read error for camera %s: %s", camera_id, e)
    finally:
        feed_stop.set()
        try:
            proc.kill()
        except Exception:
            pass
        feeder.join(timeout=3)
        logger.info("fMP4 stream for camera %s ended", camera_id)


@router.get("/camera/{camera_id}/video")
async def stream_camera_video(
    camera_id: UUID,
    fps: int = FMP4_DEFAULT_FPS,
    crf: int = FMP4_DEFAULT_CRF,
):
    """
    Stream annotated camera feed as H.264 encoded video (fMP4).

    This endpoint streams real encoded video with bounding boxes.
    Use in a standard ``<video>`` tag or any media player (VLC, ffplay, etc.).

    Example::

        <video src="http://192.168.1.15:8668/api/stream/camera/{camera_id}/video" autoplay muted></video>

    Query params:
        fps:  target frame rate (default 15, max 60)
        crf:  H.264 quality (0=lossless, 23=default, 51=worst)
    """
    if camera_id not in thread_manager.cameras:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {camera_id} is not running. Start the camera first.",
        )

    return StreamingResponse(
        generate_fmp4_video_stream(camera_id, target_fps=fps, crf=crf),
        media_type="video/mp4",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Connection": "keep-alive",
        },
    )


@router.get("/camera/{camera_id}/snapshot")
async def snapshot_camera_frame(camera_id: UUID, quality: int = MJPEG_DEFAULT_QUALITY):
    """Return a single JPEG snapshot for dashboards that poll static previews."""
    frame_result = frame_buffer_manager.get_overlay_frame(camera_id)
    if frame_result is None:
        frame_result = frame_buffer_manager.get(camera_id)

    if frame_result is None:
        frame = _build_snapshot_placeholder()
    else:
        frame, _ts = frame_result

    try:
        jpeg = _encode_snapshot_jpeg(frame, quality)
    except Exception as e:
        logger.error("Snapshot encode failed for camera %s: %s", camera_id, e)
        raise HTTPException(status_code=500, detail="Failed to encode snapshot")

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@router.get("/camera/{camera_id}/status")
async def get_camera_stream_status(camera_id: UUID):
    """
    Get streaming status for a camera.

    Returns:
    - online: True when camera is running (stream always available, even with no events)
    - frames_fresh: True when receiving frames within last 3s
    - last_frame_timestamp: timestamp of most recent frame
    """
    camera_running = camera_id in thread_manager.cameras
    frames_fresh = frame_buffer_manager.is_online(camera_id, timeout=3.0)
    result = frame_buffer_manager.get(camera_id)

    if result is None:
        return {
            "camera_id": str(camera_id),
            "online": camera_running,
            "frames_fresh": False,
            "last_frame_timestamp": None,
            "age_seconds": None,
            "message": "Stream available, waiting for first frame" if camera_running else "Camera not running",
        }

    _, ts = result
    age = round(time.time() - ts, 2)
    return {
        "camera_id": str(camera_id),
        "online": camera_running,
        "frames_fresh": frames_fresh,
        "last_frame_timestamp": ts,
        "age_seconds": age,
        "message": "Stream online" if camera_running else "Camera stopped",
    }


# ---------------------------------------------------------------------------
# WebSocket Video Proxy — forward H.265 NAL via fMP4 to browser
# ---------------------------------------------------------------------------

import asyncio
import shutil
import struct
from fastapi import WebSocket, WebSocketDisconnect

from src.core.capture_backends import (
    build_vss_ws_params,
    _build_howen_play_command,
    _prepare_howen_video_payload,
    uses_vss_backend,
    HOWEN_WS_CONNECT_TIMEOUT,
    HOWEN_WS_READ_TIMEOUT_SEC,
    HOWEN_WS_MSG_HEADER_LEN,
    HOWEN_WS_FRAME_HEADER_LEN,
    HOWEN_WS_VIDEO_FRAME_TYPES,
    HOWEN_WS_CODEC,
)


@router.websocket("/camera/{camera_id}/ws-video")
async def ws_video_proxy(websocket: WebSocket, camera_id: UUID, codec: str = ""):
    """WebSocket video proxy: VSS Howen WS → ffmpeg remux (fMP4) → browser.

    Browser connects here, receives fMP4 chunks, plays via MSE SourceBuffer.
    No re-encoding — ffmpeg only changes container (H.265 NAL → fMP4).

    Query params:
        codec: force codec (hevc/h264). Default: auto-detect from HOWEN_WS_CODEC env.
    """
    await websocket.accept()

    # Resolve camera RTSP URL from DB
    db = None
    try:
        from src.database import SessionLocal
        db = SessionLocal()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if camera is None:
            await websocket.close(code=4004, reason="Camera not found")
            return
        rtsp_url = camera.rtsp_url
        if not rtsp_url or not uses_vss_backend(rtsp_url):
            await websocket.close(code=4022, reason="Not a VSS camera")
            return
    except Exception as e:
        logger.error("ws-video: DB error for camera %s: %s", camera_id, e)
        await websocket.close(code=4500, reason=str(e))
        return
    finally:
        if db:
            db.close()

    # Resolve VSS WebSocket params
    try:
        ws_params = build_vss_ws_params(rtsp_url)
    except Exception as e:
        logger.error("ws-video: VSS params error for camera %s: %s", camera_id, e)
        await websocket.close(code=4503, reason=f"VSS resolve failed: {e}")
        return

    selected_codec = codec.strip().lower() or HOWEN_WS_CODEC or "hevc"
    if selected_codec not in ("hevc", "h264"):
        selected_codec = "hevc"

    ffmpeg_path = _get_ffmpeg_path_cached()
    stop_event = threading.Event()
    loop = asyncio.get_event_loop()

    logger.info(
        "ws-video: starting proxy for camera %s (codec=%s, device=%s, ch=%s)",
        camera_id, selected_codec, ws_params["device_id"], ws_params["channel"],
    )

    # Start ffmpeg remux process: raw NAL → fMP4 (copy, no re-encode)
    ffmpeg_proc = subprocess.Popen(
        [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
            "-f", selected_codec,
            "-i", "pipe:0",
            "-c:v", "copy",
            "-an",
            "-f", "mp4",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 7,
    )

    async def _forward_fmp4_to_browser():
        """Read fMP4 chunks from ffmpeg stdout and send to browser WebSocket."""
        try:
            while not stop_event.is_set():
                chunk = await loop.run_in_executor(
                    None, lambda: ffmpeg_proc.stdout.read(65536) if ffmpeg_proc.stdout else b""
                )
                if not chunk:
                    break
                try:
                    await websocket.send_bytes(chunk)
                except (WebSocketDisconnect, Exception):
                    break
        except Exception as e:
            if not stop_event.is_set():
                logger.warning("ws-video: fMP4 forward error: %s", e)
        finally:
            stop_event.set()

    def _receive_from_vss():
        """Connect to VSS Howen WS, receive NAL data, pipe to ffmpeg stdin."""
        try:
            import websocket as ws_lib
        except ImportError:
            logger.error("ws-video: websocket-client not installed")
            stop_event.set()
            return

        howen_ws = None
        try:
            howen_ws = ws_lib.WebSocket()
            howen_ws.connect(ws_params["ws_url"], timeout=HOWEN_WS_CONNECT_TIMEOUT)
            howen_ws.settimeout(HOWEN_WS_READ_TIMEOUT_SEC)

            play_cmd = _build_howen_play_command(
                ws_params["device_id"],
                ws_params["channel"],
                ws_params.get("stream", "1"),
            )
            howen_ws.send(play_cmd, opcode=0x2)
            logger.info("ws-video: Howen WS connected, play command sent")

            video_frames = 0
            while not stop_event.is_set():
                try:
                    opcode, data = howen_ws.recv_data(control_frame=True)
                except Exception:
                    if stop_event.is_set():
                        break
                    continue

                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    howen_ws.pong(data)
                    continue
                if opcode not in (0x1, 0x2) or len(data) < HOWEN_WS_MSG_HEADER_LEN:
                    continue

                if data[0] != 0x48:
                    continue

                action = struct.unpack_from("<H", data, 2)[0]
                if action != 1000:
                    continue

                media = data[HOWEN_WS_MSG_HEADER_LEN:]
                if len(media) < HOWEN_WS_FRAME_HEADER_LEN:
                    continue

                frame_type = struct.unpack_from("<H", media, 0)[0]
                if frame_type not in HOWEN_WS_VIDEO_FRAME_TYPES:
                    continue

                nal_data = _prepare_howen_video_payload(media[HOWEN_WS_FRAME_HEADER_LEN:])
                if not nal_data:
                    continue

                video_frames += 1
                if ffmpeg_proc.stdin:
                    try:
                        ffmpeg_proc.stdin.write(nal_data)
                        ffmpeg_proc.stdin.flush()
                    except (BrokenPipeError, OSError):
                        break

        except Exception as e:
            if not stop_event.is_set():
                logger.error("ws-video: Howen WS error: %s", e)
        finally:
            stop_event.set()
            if howen_ws:
                try:
                    howen_ws.close()
                except Exception:
                    pass
            if ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass

    def _drain_stderr():
        """Log ffmpeg errors."""
        if ffmpeg_proc.stderr is None:
            return
        for line in ffmpeg_proc.stderr:
            msg = line.decode("utf-8", errors="replace").strip()
            if msg:
                logger.warning("ws-video ffmpeg: %s", msg)

    # Start background threads
    vss_thread = threading.Thread(target=_receive_from_vss, daemon=True)
    stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
    vss_thread.start()
    stderr_thread.start()

    try:
        await _forward_fmp4_to_browser()
    except WebSocketDisconnect:
        logger.info("ws-video: browser disconnected for camera %s", camera_id)
    except Exception as e:
        logger.error("ws-video: error for camera %s: %s", camera_id, e)
    finally:
        stop_event.set()
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        vss_thread.join(timeout=3)
        logger.info("ws-video: proxy ended for camera %s", camera_id)

