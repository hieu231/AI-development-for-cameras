import logging
import uuid
import requests as http_requests
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.database import get_db
from src.models.camera import Camera
from src.core.capture_backends import (
    build_vss_flv_url,
    is_vss_realvideo_url,
    is_http_flv_url,
    uses_vss_backend,
    resolve_vss_token,
    resolve_vss_stream_host,
    _get_vss_credentials,
    _get_required_query_value,
    _extract_token_from_payload,
    VSS_STREAM_PORT_HTTP,
    VSS_STREAM_PORT_HTTPS,
)
from urllib.parse import parse_qs, urlsplit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vss", tags=["VSS"])


def _is_vss_rate_limit_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "giới hạn đăng nhập" in message or "login too frequently" in message or "too many requests" in message


# ── Schemas ──────────────────────────────────────────────────────────────────

class VSSResolveRequest(BaseModel):
    url: str
    force_login: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?token=3475675397ea4531b353d39f1aede9d2&deviceId=HAN3-20-7203&chs=1&stream=1&wnum=1&panel=0&buffer=2000",
                    "force_login": False,
                }
            ]
        }
    }


class VSSResolveResponse(BaseModel):
    flv_url: str
    ws_url: str
    token: str
    device_id: str
    stream_host: str
    channel: str
    stream: str
    is_vss: bool = True


class VSSLoginRequest(BaseModel):
    url: str
    force_login: bool = True

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?token=xxx&deviceId=HAN3-20-7203&chs=1&stream=1",
                    "force_login": True,
                }
            ]
        }
    }


class VSSLoginResponse(BaseModel):
    token: str
    login_url: str


class VSSDeviceRequest(BaseModel):
    url: str
    device_id: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?token=xxx&deviceId=HAN3-20-7203&chs=1&stream=1",
                    "device_id": "HAN3-20-7203",
                }
            ]
        }
    }


class VSSDeviceResponse(BaseModel):
    device_id: str
    stream_host: str


class VSSValidateResponse(BaseModel):
    url: str
    is_vss_realvideo: bool
    is_http_flv: bool
    uses_vss_backend: bool
    device_id: Optional[str] = None
    channel: Optional[str] = None
    stream: Optional[str] = None
    token_present: bool = False
    has_credentials: bool = False


class VSSBuildStreamRequest(BaseModel):
    """FE gửi thông tin kết nối VSS camera."""
    base_url: str = Field(..., description="Base URL của VSS server, ví dụ: http://203.171.17.183:9966/vss/apiPage/RealVideo.html")
    username: str = Field(..., description="Username để đăng nhập VSS")
    password: str = Field(..., description="Password (MD5 hash) để đăng nhập VSS")
    channel: str = Field(..., description="Kênh camera (chs)")
    device_id: str = Field(..., description="Device ID của camera")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "base_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
                    "username": "TEST1",
                    "password": "4de93544234adffbb681ed60ffcfb941",
                    "channel": "1",
                    "device_id": "HAN3-20-7203",
                }
            ]
        }
    }


class VSSBuildStreamResponse(BaseModel):
    """URL đầy đủ để xem camera stream."""
    id: str = Field(..., description="Unique request ID")
    stream_url: str = Field(..., description="URL đầy đủ với token để xem stream")
    token: str = Field(..., description="Token từ VSS login")
    device_id: str
    channel: str
    offer_url: Optional[str] = Field(default=None, description="WebRTC offer endpoint để FE gửi SDP offer")
    play_url: Optional[str] = Field(default=None, description="WebRTC play URL nếu tìm thấy camera tương ứng trong hệ thống")
    endpoint_url: Optional[str] = Field(
        default=None,
        description="Điểm cuối FE nên dùng để phát: ưu tiên WebRTC play_url, fallback sang stream_url",
    )
    endpoint_type: str = Field(
        default="stream",
        description="Loại endpoint được chọn trong endpoint_url: webrtc hoặc stream",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "stream_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?token=abc123&deviceId=HAN3-20-7203&chs=1&stream=1&wnum=1&panel=0&buffer=2000",
                    "token": "abc123",
                    "device_id": "HAN3-20-7203",
                    "channel": "1",
                    "offer_url": "http://192.168.1.15:8668/api/webrtc/offer",
                    "play_url": "http://192.168.1.15:8668/api/webrtc/play/2f4f7ec1-6f46-4f53-9a95-f3b0ff8e7f8d",
                    "endpoint_url": "http://192.168.1.15:8668/api/webrtc/play/2f4f7ec1-6f46-4f53-9a95-f3b0ff8e7f8d",
                    "endpoint_type": "webrtc",
                }
            ]
        }
    }


def _build_webrtc_offer_url(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/api/webrtc/offer"


def _build_webrtc_play_url(request: Optional[Request], camera_id: Optional[uuid.UUID]) -> Optional[str]:
    if request is None or camera_id is None:
        return None

    base_url = str(request.base_url).rstrip("/")
    return f"{base_url}/api/webrtc/play/{camera_id}"


def _find_camera_by_vss_fields(db: Optional[Session], device_id: str, channel: str) -> Optional[Camera]:
    if db is None or not hasattr(db, "query"):
        return None

    normalized_channel = str(channel or "").strip()

    # 1) Prefer exact match (device_id + channel)
    exact = (
        db.query(Camera)
        .filter(
            Camera.vss_device_id == device_id,
            Camera.vss_channel == normalized_channel,
        )
        .first()
    )
    if exact is not None:
        return exact

    # 2) Backward-compat: some legacy rows store device_id but leave channel blank
    #    while runtime defaults channel to "1".
    if normalized_channel == "1":
        fallback_blank_channel = (
            db.query(Camera)
            .filter(
                Camera.vss_device_id == device_id,
                or_(
                    Camera.vss_channel.is_(None),
                    Camera.vss_channel == "",
                ),
            )
            .first()
        )
        if fallback_blank_channel is not None:
            return fallback_blank_channel

    # 3) Last fallback: same device_id even when channel mismatches.
    #    Better to return a valid internal play URL than None, so FE does not
    #    fall back to external /stream.html links that may not exist.
    return (
        db.query(Camera)
        .filter(Camera.vss_device_id == device_id)
        .first()
    )


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/build-stream-url", response_model=VSSBuildStreamResponse)
def build_stream_url(
    body: VSSBuildStreamRequest,
    request: Request = None,
    db: Session = Depends(get_db),
):
    """Nhận thông tin camera từ FE, login VSS để lấy token, trả về URL stream đầy đủ.

    Flow:
    1. FE gửi: base_url, username, password, channel, device_id
    2. BE dùng resolve_vss_token (với global lock & cache) để lấy token
    3. Ghép URL: base_url?token=...&deviceId=...&chs=...&stream=1&wnum=1&panel=0&buffer=2000
    """
    base_url = body.base_url.strip()
    username = body.username.strip()
    password = body.password.strip()
    channel = body.channel.strip()
    device_id = body.device_id.strip()

    try:
        parts = urlsplit(base_url)
        if not parts.hostname or not parts.scheme:
            raise ValueError(f"base_url không hợp lệ: {base_url}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Không thể parse base_url: {e}")

    # Build a synthetic RealVideo URL so resolve_vss_token can use the shared
    # global login lock, token cache, cooldown, and retry logic.
    synthetic_url = (
        f"{base_url}"
        f"?username={username}"
        f"&password={password}"
        f"&deviceId={device_id}"
        f"&chs={channel}"
        f"&stream=1"
    )

    try:
        token = resolve_vss_token(synthetic_url, force_login=False)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        status_code = 429 if _is_vss_rate_limit_runtime_error(e) else 503
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.exception("VSS build-stream-url failed")
        raise HTTPException(status_code=500, detail=f"Lỗi không xác định: {e}")

    # Build full stream URL
    # TODO: stream, wnum, panel, buffer đang hard-coded, cần cho FE config sau
    stream_url = (
        f"{base_url}"
        f"?token={token}"
        f"&deviceId={device_id}"
        f"&chs={channel}"
        f"&stream=1"
        f"&wnum=1"
        f"&panel=0"
        f"&buffer=2000"
    )

    logger.info(f"VSS stream URL built successfully for device={device_id}")

    request_id = str(uuid.uuid4())
    camera = _find_camera_by_vss_fields(db, device_id, channel)
    offer_url = _build_webrtc_offer_url(request)
    play_url = _build_webrtc_play_url(request, getattr(camera, "id", None))
    endpoint_url = play_url or stream_url
    endpoint_type = "webrtc" if play_url else "stream"

    return VSSBuildStreamResponse(
        id=request_id,
        stream_url=stream_url,
        token=token,
        device_id=device_id,
        channel=channel,
        offer_url=offer_url,
        play_url=play_url,
        endpoint_url=endpoint_url,
        endpoint_type=endpoint_type,
    )


@router.post("/resolve", response_model=VSSResolveResponse)
def resolve_vss_url(body: VSSResolveRequest):
    """Gộp toàn bộ flow VSS: login → query device → build FLV + WebSocket URL.

    Nhận RealVideo URL, trả về cả FLV và WebSocket URL sẵn sàng dùng cho AI pipeline.
    """
    url = body.url.strip()

    if not is_vss_realvideo_url(url):
        raise HTTPException(
            status_code=422,
            detail=f"URL không phải VSS RealVideo format. Cần path chứa /vss/apiPage/RealVideo.html",
        )

    try:
        # 1. Resolve token (login nếu cần)
        token = resolve_vss_token(url, force_login=body.force_login)

        # 2. Parse query params
        parts = urlsplit(url)
        query = parse_qs(parts.query, keep_blank_values=True)
        device_id = _get_required_query_value(query, "deviceId")
        channels = _get_required_query_value(query, "chs")
        stream_val = _get_required_query_value(query, "stream")
        channel = channels.split("_")[0]

        # 3. Resolve stream host (query device gateway)
        stream_host = resolve_vss_stream_host(url, device_id)

        # 4. Build FLV URL (for FFmpeg-based consistent real-time capture)
        flv_url = build_vss_flv_url(url, force_login=body.force_login)

        # 5. Build WebSocket URL (for Howen protocol)
        secure = parts.scheme.lower() == "https"
        ws_scheme = "wss" if secure else "ws"
        ws_port = VSS_STREAM_PORT_HTTPS if secure else VSS_STREAM_PORT_HTTP
        ws_url = f"{ws_scheme}://{stream_host}:{ws_port}/stream"

        return VSSResolveResponse(
            flv_url=flv_url,
            ws_url=ws_url,
            token=token,
            device_id=device_id,
            stream_host=stream_host,
            channel=channel,
            stream=stream_val,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        status_code = 429 if _is_vss_rate_limit_runtime_error(e) else 503
        raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.exception("VSS resolve failed for %s", url)
        raise HTTPException(status_code=500, detail=f"VSS resolve error: {e}")


@router.post("/login", response_model=VSSLoginResponse)
def vss_login(body: VSSLoginRequest):
    """Login vào VSS server, trả về token.

    Yêu cầu VSS_USERNAME và VSS_PASSWORD_MD5 trong env hoặc token trong URL.
    """
    url = body.url.strip()

    if not is_vss_realvideo_url(url):
        raise HTTPException(status_code=422, detail="URL không phải VSS RealVideo format")

    try:
        from src.core.capture_backends import _build_vss_login_url

        login_url = _build_vss_login_url(url)
        token = resolve_vss_token(url, force_login=body.force_login)

        return VSSLoginResponse(token=token, login_url=login_url)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("VSS login failed for %s", url)
        raise HTTPException(status_code=500, detail=f"VSS login error: {e}")


@router.post("/device", response_model=VSSDeviceResponse)
def vss_query_device(body: VSSDeviceRequest):
    """Query gateway host thực tế của VSS device."""
    url = body.url.strip()

    if not is_vss_realvideo_url(url):
        raise HTTPException(status_code=422, detail="URL không phải VSS RealVideo format")

    try:
        stream_host = resolve_vss_stream_host(url, body.device_id)
        return VSSDeviceResponse(device_id=body.device_id, stream_host=stream_host)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("VSS device query failed for %s", body.device_id)
        raise HTTPException(status_code=500, detail=f"VSS device query error: {e}")


@router.get("/validate", response_model=VSSValidateResponse)
def vss_validate_url(url: str = Query(..., description="URL cần validate")):
    """Validate xem URL có phải VSS format không và trả về thông tin parsed."""
    url = url.strip()
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)

    device_id = None
    channel = None
    stream_val = None
    token_present = False

    try:
        device_id = _get_required_query_value(query, "deviceId")
    except ValueError:
        pass

    try:
        channels = _get_required_query_value(query, "chs")
        channel = channels.split("_")[0]
    except ValueError:
        pass

    try:
        stream_val = _get_required_query_value(query, "stream")
    except ValueError:
        pass

    token_val = query.get("token")
    token_present = bool(token_val and token_val[0])

    username, password_md5 = _get_vss_credentials(url)
    has_credentials = bool(username and password_md5)

    return VSSValidateResponse(
        url=url,
        is_vss_realvideo=is_vss_realvideo_url(url),
        is_http_flv=is_http_flv_url(url),
        uses_vss_backend=uses_vss_backend(url),
        device_id=device_id,
        channel=channel,
        stream=stream_val,
        token_present=token_present,
        has_credentials=has_credentials,
    )
