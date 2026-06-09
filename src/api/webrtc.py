"""
WebRTC Signaling API - HTTP endpoints for WebRTC session negotiation.
"""
import logging
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel as PydanticModel
from sqlalchemy.orm import Session

from src.core.capture_backends import build_vss_realvideo_url, is_http_flv_url
from src.core.webrtc_manager import webrtc_manager, WEBRTC_ENABLED, AIORTC_AVAILABLE
from src.core.thread_manager import thread_manager
from src.core.frame_buffer import frame_buffer_manager
from src.database import get_db
from src.models.camera import Camera

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webrtc", tags=["webrtc"])


def _http_error_for_camera_start_failure(camera_id: UUID) -> HTTPException:
    detail = thread_manager.get_last_start_error(camera_id)
    if not detail:
        detail = "Camera failed to connect to stream. Please check source configuration."

    lowered = detail.lower()
    status_code = 503
    if (
        "malformed" in lowered
        or "missing required query parameter" in lowered
        or "missing vss token" in lowered
        or "missing host or scheme" in lowered
    ):
        status_code = 422
    elif (
        "giới hạn đăng nhập" in lowered
        or "login too frequently" in lowered
        or "too many requests" in lowered
    ):
        status_code = 429

    return HTTPException(status_code=status_code, detail=detail)


def _resolve_camera_source_url(camera: Camera) -> str:
    """Resolve the runtime source URL from camera master data.

    For VSS cameras, runtime stream URL should be derived from stored VSS
    config (base_url/user/password/device/channel) so token/session stay
    internal to AI BE.
    """
    protocol = str(getattr(camera, "protocol", "") or "").strip().upper()
    current_url = str(getattr(camera, "rtsp_url", "") or "").strip()

    if protocol != "VSS":
        if not current_url:
            raise HTTPException(status_code=422, detail="Camera source URL is empty")
        return current_url

    base_url = str(getattr(camera, "vss_base_url", "") or "").strip()
    username = str(getattr(camera, "vss_username", "") or "").strip()
    password = str(getattr(camera, "vss_password", "") or "").strip()
    device_id = str(getattr(camera, "vss_device_id", "") or "").strip()
    channel = str(getattr(camera, "vss_channel", "") or "").strip() or "1"

    if not base_url or not username or not password or not device_id:
        raise HTTPException(
            status_code=422,
            detail=(
                "VSS camera requires vss_base_url, vss_username, vss_password, "
                "and vss_device_id"
            ),
        )

    preserved_token = ""
    if current_url:
        if is_http_flv_url(current_url):
            try:
                preserved_token = urlsplit(current_url).query[5:].rsplit("_", 3)[0]
            except ValueError:
                preserved_token = ""
        else:
            preserved_token = (
                parse_qs(urlsplit(current_url).query, keep_blank_values=True)
                .get("token", [""])[0]
                .strip()
            )

    return build_vss_realvideo_url(
        base_url,
        username=username,
        password_md5=password,
        device_id=device_id,
        channel=channel,
        stream="1",
        token=preserved_token or None,
    )


def _ensure_camera_ready_for_webrtc(camera_id: UUID, db: Session) -> Camera:
    # Exclude soft-deleted cameras so a stale viewer cannot respawn a stream
    # worker for a camera that has already been removed.
    camera = (
        db.query(Camera)
        .filter(Camera.id == camera_id, Camera.is_deleted == False)  # noqa: E712
        .first()
    )
    if camera is None:
        # If a worker is still in memory for this id, tear it down so the
        # deletion propagates even when the NestJS backend crashed mid-flow.
        if camera_id in thread_manager.cameras:
            try:
                thread_manager.stop_camera(camera_id)
                logger.warning(
                    "WebRTC: stopped orphan stream for deleted camera %s", camera_id
                )
            except Exception as exc:
                logger.error("WebRTC: failed to stop orphan %s: %s", camera_id, exc)
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    if not bool(getattr(camera, "status", False)):
        raise HTTPException(
            status_code=409,
            detail="Camera is stopped (status=false). Start camera before opening WebRTC.",
        )

    resolved_source_url = _resolve_camera_source_url(camera)
    if resolved_source_url != (camera.rtsp_url or ""):
        camera.rtsp_url = resolved_source_url
        db.commit()
        db.refresh(camera)
        logger.info("WebRTC: refreshed runtime source URL for camera %s", camera_id)

    if camera_id not in thread_manager.cameras:
        started = thread_manager.start_camera(camera_id, show_display=False)
        if not started:
            raise _http_error_for_camera_start_failure(camera_id)

    return camera


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class WebRTCOfferRequest(PydanticModel):
  camera_id: UUID
  sdp: str
  type: str = "offer"
  fps: int = 30


class WebRTCAnswerResponse(PydanticModel):
  connection_id: str
  sdp: str
  type: str
  camera_id: str
  fps: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/offer", response_model=WebRTCAnswerResponse)
async def webrtc_offer(request: WebRTCOfferRequest, db: Session = Depends(get_db)):
    """
    Receive a WebRTC SDP offer from the client, create a PeerConnection
    with a video track reading from the annotated frame buffer, and return
    the SDP answer.

    The client can then set this answer as the remote description to
    start receiving the video stream.
    """
    if request.camera_id not in thread_manager.cameras and not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=404,
            detail=f"Camera {request.camera_id} is not running",
        )

    if not AIORTC_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="WebRTC is not available. Install aiortc: pip install aiortc"
        )
    if not WEBRTC_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="WebRTC is disabled. Set WEBRTC_ENABLED=true"
        )

    # Resolve camera source from internal metadata and auto-start if needed.
    _ensure_camera_ready_for_webrtc(request.camera_id, db)

    try:
        result = await webrtc_manager.create_connection(
            camera_id=request.camera_id,
            sdp_offer=request.sdp,
            sdp_type=request.type,
            fps=request.fps,
        )
        return WebRTCAnswerResponse(**result)
    except Exception as exc:
        logger.error(f"WebRTC offer failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/connections")
async def list_connections():
    """List all active WebRTC connections."""
    conns = webrtc_manager.get_connections()
    return {
        "connections": conns,
        "total": len(conns),
    }


@router.delete("/connections/{connection_id}")
async def close_connection(connection_id: str):
    """Close a specific WebRTC connection.

    Idempotent: returns 200 even when the connection was already reaped by
    the server-side state-change handler. The client's intent is "make sure
    this is closed", and it is. Returning 404 here caused viewer pages to
    treat normal teardowns as errors and retry in a loop.
    """
    closed = await webrtc_manager.close_connection(connection_id)
    return {
        "status": "closed" if closed else "already_closed",
        "connection_id": connection_id,
    }


@router.get("/status")
async def webrtc_status():
    """Get WebRTC subsystem status."""
    return {
        "enabled": WEBRTC_ENABLED,
        "aiortc_available": AIORTC_AVAILABLE,
        "active_connections": len(webrtc_manager.get_connections()),
        "buffered_cameras": [str(c) for c in frame_buffer_manager.list_cameras()],
    }


@router.get("/play/{camera_id}", response_class=HTMLResponse)
async def webrtc_play(camera_id: UUID, request: Request, fps: int = 30):
    """
    Mở trang WebRTC stream viewer trực tiếp trong trình duyệt.

    Dùng:  GET http://192.168.1.15:8668/api/webrtc/play/{camera_id}
           GET http://192.168.1.15:8668/api/webrtc/play/{camera_id}?fps=10
    """
    base_url = str(request.base_url).rstrip("/")
    cam_str = str(camera_id)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WebRTC Stream — {cam_str[:8]}…</title>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  html, body {{ width:100%; height:100%; overflow:hidden; background:#000; }}
  video {{ width:100%; height:100%; object-fit:contain; display:block; }}
</style>
</head>
<body>
<video id="v" autoplay playsinline muted></video>
<script>
const SERVER = "{base_url}";
const CAM = "{cam_str}";
const FPS = {fps};
let pc = null, connId = null;

async function start() {{
  if(pc) await stop();
  pc = new RTCPeerConnection({{ iceServers:[{{urls:'stun:stun.l.google.com:19302'}}] }});
  pc.addTransceiver('video', {{ direction:'recvonly' }});
  pc.ontrack = e => {{ document.getElementById('v').srcObject=e.streams[0]; }};
  pc.oniceconnectionstatechange = () => {{
    const s=pc.iceConnectionState;
    if(s==='failed'||s==='disconnected') {{ setTimeout(()=>{{ stop().then(start); }}, 2000); }}
  }};
  try {{
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const r = await fetch(SERVER+'/api/webrtc/offer', {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{ camera_id:CAM, sdp:pc.localDescription.sdp, type:pc.localDescription.type, fps:FPS }})
    }});
    if(!r.ok) {{ const e=await r.json(); throw new Error(e.detail||r.statusText); }}
    const ans = await r.json();
    connId = ans.connection_id;
    await pc.setRemoteDescription(new RTCSessionDescription({{ sdp:ans.sdp, type:ans.type }}));
  }} catch(e) {{ console.error(e); pc.close(); pc=null; setTimeout(start, 3000); }}
}}

async function stop() {{
  if(pc) {{ pc.close(); pc=null; }}
  if(connId) {{ try {{ await fetch(SERVER+'/api/webrtc/connections/'+connId,{{method:'DELETE'}}); }} catch(_){{}} connId=null; }}
  document.getElementById('v').srcObject=null;
}}

// Auto-start
start();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
