# src/endpoints/camera_endpoint.py
import logging
from urllib.parse import parse_qs, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, field_validator

from src.core.capture_backends import (
    build_vss_realvideo_url,
    is_http_flv_url,
    is_vss_realvideo_url,
    uses_vss_backend,
)
from src.models.camera import Camera

logger = logging.getLogger(__name__)
from src.models.location import Location
from src.database import get_db
from src.core.thread_manager import thread_manager


# Pydantic schemas
class CameraBase(BaseModel):
    name: str
    rtsp_url: str = ""
    location_id: UUID
    raw_resolution: Optional[str] = None
    preprocess_resolution: Optional[str] = None
    status: bool = True
    camera_spec_id: Optional[UUID] = None
    # VSS protocol fields (optional – used when protocol="VSS")
    protocol: Optional[str] = None
    vss_base_url: Optional[str] = None
    vss_username: Optional[str] = None
    vss_password: Optional[str] = None
    vss_device_id: Optional[str] = None
    vss_channel: Optional[str] = None

    @field_validator("rtsp_url", mode="before")
    @classmethod
    def normalize_rtsp_url(cls, value):
        if value is None:
            return ""
        return str(value).strip()


class CameraCreate(CameraBase):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Front Gate Camera",
                    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream",
                    "location_id": "46f8666c-07af-4ba2-9045-83d933b62e63",
                    "camera_spec_id": "62d2c567-b2ae-4bdb-9d8b-555e272db78d",
                    "raw_resolution": "1920x1080",
                    "preprocess_resolution": "640x480",
                    "status": False,
                },
                {
                    "name": "Go2RTC HTTP Stream",
                    "rtsp_url": "https://go2rtc.pathtech.net/api/stream.mp4?src=file1",
                    "location_id": "46f8666c-07af-4ba2-9045-83d933b62e63",
                    "camera_spec_id": None,
                    "raw_resolution": "1920x1080",
                    "status": True,
                },
                {
                    "name": "WebSocket Camera",
                    "rtsp_url": "ws://localhost:1984/api/ws?src=camera1",
                    "location_id": "46f8666c-07af-4ba2-9045-83d933b62e63",
                    "camera_spec_id": None,
                    "status": True,
                },
            ]
        }
    }


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location_id: Optional[UUID] = None
    raw_resolution: Optional[str] = None
    preprocess_resolution: Optional[str] = None
    status: Optional[bool] = None
    camera_spec_id: Optional[UUID] = None
    # VSS protocol fields (optional – used when protocol="VSS")
    protocol: Optional[str] = None
    vss_base_url: Optional[str] = None
    vss_username: Optional[str] = None
    vss_password: Optional[str] = None
    vss_device_id: Optional[str] = None
    vss_channel: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Updated Camera Name",
                    "camera_spec_id": UUID("62d2c567-b2ae-4bdb-9d8b-555e272db78d"),
                },
                {"status": True, "camera_spec_id": None},
            ]
        }
    }


class LocationInfo(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None


class CameraSpecInfo(BaseModel):
    id: UUID
    name: str
    manufacturer: str
    model_series: Optional[str] = None
    category: Optional[str] = None
    resolution: Optional[str] = None
    max_fps: Optional[int] = None
    ptz_support: bool = False
    audio_support: bool = False
    ir_support: bool = False
    ai_support: bool = False
    rtsp_format: Optional[str] = None


class CameraResponse(CameraBase):
    id: UUID
    location: LocationInfo
    camera_spec: Optional[CameraSpecInfo] = None
    created_at: datetime
    updated_at: datetime
    streams: Optional[dict] = None
    startup_error: Optional[str] = None
    has_startup_error: bool = False

    model_config = {"from_attributes": True}


router = APIRouter(prefix="/cameras", tags=["cameras"])

_VSS_REQUIRED_FIELDS = ("vss_base_url", "vss_username", "vss_password", "vss_device_id")


def _effective_vss_field(
    camera_data: dict, existing_camera: Camera | None, field_name: str
) -> str:
    value = camera_data.get(field_name)
    if value is None and existing_camera is not None:
        value = getattr(existing_camera, field_name, None)
    if value is None:
        return ""
    return str(value).strip()


def _normalize_camera_rtsp_url(
    camera_data: dict, existing_camera: Camera | None = None
) -> None:
    rtsp_url = str(camera_data.get("rtsp_url") or "").strip()
    protocol = _effective_vss_field(camera_data, existing_camera, "protocol").upper()
    is_direct_http_flv = bool(rtsp_url and is_http_flv_url(rtsp_url))
    direct_http_flv_token = ""
    direct_http_flv_device_id = ""
    direct_http_flv_channel = ""
    if rtsp_url:
        camera_data["rtsp_url"] = rtsp_url

    if is_direct_http_flv:
        try:
            stream_spec = urlsplit(rtsp_url).query[5:]
            (
                direct_http_flv_token,
                direct_http_flv_device_id,
                direct_http_flv_channel,
                _direct_http_flv_stream,
            ) = stream_spec.rsplit("_", 3)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Direct HTTP-FLV URLs cannot be stored because the URL format is invalid.",
            ) from exc
        protocol = "VSS"
        camera_data["protocol"] = "VSS"

    if protocol != "VSS":
        return

    # Rebuild VSS RealVideo URL from stored fields
    should_rebuild = (
        is_direct_http_flv
        or (not rtsp_url)
        or not is_vss_realvideo_url(rtsp_url)
        or any(
        field_name in camera_data for field_name in (*_VSS_REQUIRED_FIELDS, "vss_channel")
        )
    )
    if not should_rebuild:
        return

    vss_base = _effective_vss_field(camera_data, existing_camera, "vss_base_url")
    vss_user = _effective_vss_field(camera_data, existing_camera, "vss_username")
    vss_pass = _effective_vss_field(camera_data, existing_camera, "vss_password")
    vss_dev = (
        _effective_vss_field(camera_data, existing_camera, "vss_device_id")
        or direct_http_flv_device_id
    )
    vss_ch = (
        _effective_vss_field(camera_data, existing_camera, "vss_channel")
        or direct_http_flv_channel
        or "1"
    )

    # Try to preserve any existing token
    preserved_token = ""
    if is_direct_http_flv:
        preserved_token = direct_http_flv_token.strip()
    elif rtsp_url and is_vss_realvideo_url(rtsp_url):
        preserved_token = (parse_qs(urlsplit(rtsp_url).query, keep_blank_values=True).get("token") or [""])[0].strip()
    if not preserved_token and existing_camera is not None:
        existing_rtsp_url = str(getattr(existing_camera, "rtsp_url", "") or "").strip()
        if existing_rtsp_url and is_http_flv_url(existing_rtsp_url):
            try:
                preserved_token = urlsplit(existing_rtsp_url).query[5:].rsplit("_", 3)[0].strip()
            except ValueError:
                preserved_token = ""
        elif existing_rtsp_url:
            preserved_token = (parse_qs(urlsplit(existing_rtsp_url).query, keep_blank_values=True).get("token") or [""])[0].strip()

    if not vss_base or not vss_user or not vss_pass or not vss_dev:
        detail = "VSS camera requires vss_base_url, vss_username, vss_password, and vss_device_id."
        if is_direct_http_flv:
            detail = (
                "Direct HTTP-FLV URLs cannot be stored. Provide VSS metadata "
                "(vss_base_url, vss_username, vss_password, vss_device_id)."
            )
        raise HTTPException(
            status_code=422,
            detail=detail,
        )

    camera_data["protocol"] = "VSS"
    camera_data["rtsp_url"] = build_vss_realvideo_url(
        vss_base,
        username=vss_user,
        password_md5=vss_pass,
        device_id=vss_dev,
        channel=vss_ch,
        stream="1",
        token=preserved_token or None,
    )
    logger.info("Normalized VSS stream URL to RealVideo form for persistence")


def _ensure_camera_rtsp_url(camera_data: dict, detail: str) -> None:
    rtsp_url = str(camera_data.get("rtsp_url") or "").strip()
    if not rtsp_url:
        raise HTTPException(status_code=422, detail=detail)
    camera_data["rtsp_url"] = rtsp_url


def _camera_start_http_error(camera_id: UUID) -> HTTPException:
    detail = thread_manager.get_last_start_error(camera_id)
    if not detail:
        detail = "Camera failed to connect to stream. Please check the RTSP URL and ensure the camera is accessible."

    lowered = detail.lower()
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    if (
        "malformed" in lowered
        or "missing required query parameter" in lowered
        or "missing vss token" in lowered
        or "missing host or scheme" in lowered
    ):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif (
        "giới hạn đăng nhập" in lowered
        or "login too frequently" in lowered
        or "too many requests" in lowered
    ):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS

    return HTTPException(status_code=status_code, detail=detail)


def _attach_camera_runtime_state(camera: Camera) -> Camera:
    startup_error = thread_manager.get_last_start_error(camera.id)
    camera.startup_error = startup_error
    camera.has_startup_error = bool(startup_error)
    return camera


def _is_vss_camera(camera: Camera) -> bool:
    protocol = str(getattr(camera, "protocol", "") or "").strip().upper()
    if protocol == "VSS":
        return True

    rtsp_url = str(getattr(camera, "rtsp_url", "") or "").strip()
    if rtsp_url and uses_vss_backend(rtsp_url):
        return True

    return any(
        bool(str(getattr(camera, field, "") or "").strip())
        for field in ("vss_base_url", "vss_username", "vss_password", "vss_device_id")
    )


@router.get("/help")
def get_help():
    """Lấy hướng dẫn sử dụng Camera API"""
    return {
        "endpoints": {
            "GET /cameras/": "Lấy danh sách cameras (hỗ trợ filter)",
            "GET /cameras/help": "Hiển thị hướng dẫn này",
            "GET /cameras/{camera_id}": "Lấy thông tin chi tiết của 1 camera theo ID",
            "POST /cameras/": "Tạo camera mới",
            "PUT /cameras/{camera_id}": "Cập nhật thông tin camera",
            "DELETE /cameras/{camera_id}": "Xóa camera",
            "PUT /cameras/{camera_id}/start": "Bắt đầu chạy camera",
            "PUT /cameras/{camera_id}/stop": "Dừng chạy camera",
        },
        "camera_schema": {
            "name": "string - Tên camera",
            "rtsp_url": "string - Stream URL (support: rtsp://, http://, https://, ws://)",
            "location_id": "UUID - ID của location",
            "camera_spec_id": "UUID - ID của camera spec (optional)",
            "raw_resolution": "string - Độ phân giải gốc (vd: 1920x1080) (optional)",
            "preprocess_resolution": "string - Độ phân giải xử lý (vd: 640x480) (optional)",
            "status": "boolean - Trạng thái camera: true=active, false=inactive (default: true)",
        },
        "filter_parameters": {
            "status": "boolean - Lọc theo trạng thái (true/false)",
            "location_id": "UUID - Lọc theo location",
            "camera_spec_id": "UUID - Lọc theo camera spec",
            "name": "string - Tìm kiếm theo tên (partial match, không phân biệt hoa thường)",
            "skip": "int - Số bản ghi bỏ qua (default: 0)",
            "limit": "int - Số bản ghi tối đa trả về (default: 100)",
        },
        "filter_examples": [
            "GET /cameras/?status=true - Lấy tất cả cameras đang active",
            "GET /cameras/?status=false - Lấy tất cả cameras đang inactive",
            "GET /cameras/?location_id=UUID - Lấy cameras theo location",
            "GET /cameras/?name=gate - Tìm cameras có tên chứa 'gate'",
            "GET /cameras/?status=true&location_id=UUID - Kết hợp nhiều filter",
        ],
        "supported_stream_types": {
            "RTSP": "rtsp://192.168.1.1:554/stream",
            "HTTP/HTTPS": "https://go2rtc.pathtech.net/api/stream.mp4?src=file1",
            "WebSocket": "ws://localhost:1984/api/ws?src=camera1",
        },
        "response_includes": {
            "location": "Thông tin location (id, name, description)",
            "camera_spec": "Thông tin camera spec (id, name, manufacturer, model_series, etc.) - nullable",
        },
        "examples": [
            {
                "name": "Camera RTSP",
                "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream",
                "location_id": "⚠️ Use /api/locations/ to get a real UUID",
                "status": True,
            },
            {
                "name": "Go2RTC Stream",
                "rtsp_url": "https://go2rtc.pathtech.net/api/stream.mp4?src=file1",
                "location_id": "⚠️ Use /api/locations/ to get a real UUID",
                "status": True,
            },
            {
                "name": "WebSocket Camera",
                "rtsp_url": "ws://localhost:1984/api/ws?src=camera1",
                "location_id": "⚠️ Use /api/locations/ to get a real UUID",
                "status": True,
            },
        ],
        "important_note": "⚠️ location_id and camera_spec_id MUST be valid UUIDs from their respective endpoints. Run: ./get_test_ids.sh to get copy-paste ready JSON",
    }


@router.get("/", response_model=List[CameraResponse])
def get_cameras(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    status: Optional[bool] = None,
    location_id: Optional[UUID] = None,
    camera_spec_id: Optional[UUID] = None,
    name: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Lấy danh sách cameras với các filter tùy chọn

    - **status**: Lọc theo trạng thái (true=active, false=inactive)
    - **location_id**: Lọc theo location
    - **camera_spec_id**: Lọc theo camera spec
    - **name**: Tìm kiếm theo tên (partial match, không phân biệt hoa thường)
    """
    query = db.query(Camera).options(
        joinedload(Camera.location), joinedload(Camera.camera_spec)
    )
    query = query.filter(Camera.is_deleted == False)

    # Apply filters
    if status is not None:
        query = query.filter(Camera.status == status)
    if location_id is not None:
        query = query.filter(Camera.location_id == location_id)
    if camera_spec_id is not None:
        query = query.filter(Camera.camera_spec_id == camera_spec_id)
    if name is not None:
        query = query.filter(Camera.name.ilike(f"%{name}%"))

    cameras = query.offset(skip).limit(limit).all()

    # Populate stream links
    from src.api.stream import _build_stream_links

    base_url = str(request.base_url).rstrip("/")
    for cam in cameras:
        cam.streams = _build_stream_links(cam.id, base_url)
        _attach_camera_runtime_state(cam)

    return cameras


@router.get("/{camera_id}", response_model=CameraResponse)
def get_camera(camera_id: UUID, request: Request, db: Session = Depends(get_db)):
    """Lấy thông tin camera theo ID"""
    camera = (
        db.query(Camera)
        .options(joinedload(Camera.location), joinedload(Camera.camera_spec))
        .filter(Camera.id == camera_id, Camera.is_deleted == False)
        .first()
    )
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Populate stream links
    from src.api.stream import _build_stream_links

    base_url = str(request.base_url).rstrip("/")
    camera.streams = _build_stream_links(camera.id, base_url)
    _attach_camera_runtime_state(camera)

    return camera


@router.post("/", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def create_camera(
    camera: CameraCreate, request: Request, db: Session = Depends(get_db)
):
    """Tạo camera mới"""
    camera_data = camera.model_dump()

    _normalize_camera_rtsp_url(camera_data)
    _ensure_camera_rtsp_url(
        camera_data,
        "rtsp_url is required (or provide VSS fields with protocol=VSS)",
    )

    db_camera = Camera(**camera_data)

    db.add(db_camera)
    db.commit()
    db.refresh(db_camera)

    # Start camera threads nếu status=True (without display for API)
    if db_camera.status:
        success = thread_manager.start_camera(db_camera.id, show_display=False)

        # Nếu không kết nối được, giữ camera với status=False để có thể
        # cấu hình/chạy lại thủ công thay vì tự động xóa bản ghi.
        if not success:
            db_camera.status = False
            db.commit()
            db.refresh(db_camera)
            logger.warning(
                "Camera %s created but initial start failed: %s",
                db_camera.id,
                thread_manager.get_last_start_error(db_camera.id),
            )

    # Load location and camera_spec data for response
    camera_with_relations = (
        db.query(Camera)
        .options(joinedload(Camera.location), joinedload(Camera.camera_spec))
        .filter(Camera.id == db_camera.id)
        .first()
    )

    # Populate stream links
    from src.api.stream import _build_stream_links

    base_url = str(request.base_url).rstrip("/")
    camera_with_relations.streams = _build_stream_links(db_camera.id, base_url)
    _attach_camera_runtime_state(camera_with_relations)

    return camera_with_relations


@router.put("/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: UUID,
    camera: CameraUpdate,
    request: Request,
    db: Session = Depends(get_db),
):
    """Cập nhật camera"""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    update_data = camera.model_dump(exclude_unset=True)
    old_rtsp_url = db_camera.rtsp_url

    _normalize_camera_rtsp_url(update_data, existing_camera=db_camera)
    if "rtsp_url" in update_data:
        _ensure_camera_rtsp_url(
            update_data,
            "rtsp_url cannot be empty. Provide a valid stream URL or VSS fields with protocol=VSS.",
        )

    # Check if rtsp_url changed → need to restart camera synchronously
    url_changed = "rtsp_url" in update_data and update_data["rtsp_url"] != old_rtsp_url

    # Apply updates to DB
    for key, value in update_data.items():
        setattr(db_camera, key, value)
    db.commit()
    db.refresh(db_camera)

    # If URL changed and camera is active, restart synchronously
    if url_changed and db_camera.status:
        thread_manager.stop_camera(camera_id)
        import time

        time.sleep(1.0)  # Allow cleanup

        success = thread_manager.start_camera(camera_id, show_display=False)
        if not success:
            # Revert rtsp_url on connection failure
            db_camera.rtsp_url = old_rtsp_url
            db_camera.status = False
            db.commit()
            db.refresh(db_camera)
            raise _camera_start_http_error(camera_id)

    # Load location and camera_spec data for response
    camera_with_relations = (
        db.query(Camera)
        .options(joinedload(Camera.location), joinedload(Camera.camera_spec))
        .filter(Camera.id == db_camera.id)
        .first()
    )

    # Populate stream links
    from src.api.stream import _build_stream_links

    base_url = str(request.base_url).rstrip("/")
    camera_with_relations.streams = _build_stream_links(camera_id, base_url)
    _attach_camera_runtime_state(camera_with_relations)

    return camera_with_relations


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: UUID, db: Session = Depends(get_db)):
    """Xóa camera"""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Stop camera threads trước khi xóa
    thread_manager.stop_camera(camera_id)

    db_camera.is_deleted = True
    db_camera.status = False
    db.commit()
    return None


@router.put("/{camera_id}/start")
def start_camera(camera_id: UUID, request: Request, db: Session = Depends(get_db)):
    """Bắt đầu chạy camera"""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Start camera threads WITHOUT display (cv2.imshow không hoạt động trong API server)
    success = thread_manager.start_camera(camera_id, show_display=False)
    if not success:
        # Hardening for VSS: keep API stable and let FE read startup_error
        # instead of receiving immediate 503 for unstable upstream streams.
        if _is_vss_camera(db_camera):
            db_camera.status = False
            db.commit()
            db.refresh(db_camera)
            _attach_camera_runtime_state(db_camera)

            from src.api.stream import _build_stream_links

            base_url = str(request.base_url).rstrip("/")
            streams = _build_stream_links(camera_id, base_url)

            return {
                "message": "Camera start failed (VSS upstream unavailable)",
                "camera": db_camera,
                "streams": streams,
                "startup_error": db_camera.startup_error,
                "has_startup_error": db_camera.has_startup_error,
                "started": False,
            }
        raise _camera_start_http_error(camera_id)

    db_camera.status = True
    db.commit()
    db.refresh(db_camera)
    _attach_camera_runtime_state(db_camera)

    # Build stream links
    from src.api.stream import _build_stream_links

    base_url = str(request.base_url).rstrip("/")
    streams = _build_stream_links(camera_id, base_url)

    return {
        "message": "Camera started successfully",
        "camera": db_camera,
        "streams": streams,
        "startup_error": db_camera.startup_error,
        "has_startup_error": db_camera.has_startup_error,
    }


@router.put("/{camera_id}/stop")
def stop_camera(camera_id: UUID, db: Session = Depends(get_db)):
    """Stop camera synchronously."""
    db_camera = db.query(Camera).filter(Camera.id == camera_id).first()
    if not db_camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Persist stopped state first so downstream flows observe status=False.
    db_camera.status = False
    db.commit()
    db.refresh(db_camera)

    was_running = camera_id in thread_manager.cameras
    stopped = thread_manager.stop_camera(camera_id)

    if was_running and not stopped:
        logger.error("Synchronous stop failed for camera %s", camera_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop camera synchronously",
        )

    db.refresh(db_camera)
    return {
        "message": "Camera stopped successfully"
        if (was_running and stopped)
        else "Camera already stopped",
        "camera": db_camera,
    }
