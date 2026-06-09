from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from urllib.parse import parse_qs, urlsplit

from src.models.camera import Camera
from src.models.ai_model import AiModel
from src.models.event import Event
from src.models.camera_model import CameraModel
from src.database import get_db
from src.core.thread_manager import thread_manager
from src.core.websocket_manager import websocket_manager
import psutil
import shutil
import asyncio

router = APIRouter(prefix="/system", tags=["system"])


class CamerasSummary(BaseModel):
    total: int
    online: int
    offline: int


class ModelsSummary(BaseModel):
    total: int
    loaded: int
    not_loaded: int
    active: int
    inactive: int


class EventsSummary(BaseModel):
    today: int
    total: int


class StorageSummary(BaseModel):
    used_percent: float
    used_gb: float
    total_gb: float
    status: str


class SystemSummary(BaseModel):
    cameras: CamerasSummary
    models: ModelsSummary
    events: EventsSummary
    storage: StorageSummary


@router.get("/summary", response_model=SystemSummary)
def get_system_summary(db: Session = Depends(get_db)):
    """
    Get system summary including cameras, models, events, and storage statistics.

    Returns:
        SystemSummary: Complete system overview with:
        - Camera counts (total, online, offline)
        - Model counts (total, loaded, not_loaded, active, inactive)
        - Event counts (today, total)
        - Storage usage (used_percent, used_gb, total_gb, status)
    """

    # ===== CAMERAS =====
    # Total: all cameras in database
    total_cameras = db.query(Camera).filter(Camera.is_deleted == False).count()

    # Online: cameras with status=True (running)
    online_cameras = (
        db.query(Camera)
        .filter(Camera.status == True, Camera.is_deleted == False)
        .count()
    )

    # Offline: cameras with status=False (not running)
    offline_cameras = (
        db.query(Camera)
        .filter(Camera.status == False, Camera.is_deleted == False)
        .count()
    )

    cameras_summary = CamerasSummary(
        total=total_cameras, online=online_cameras, offline=offline_cameras
    )

    # ===== MODELS =====
    total_models = db.query(AiModel).count()
    active_models = db.query(AiModel).filter(AiModel.is_active == True).count()
    inactive_models = db.query(AiModel).filter(AiModel.is_active == False).count()

    # Get loaded models: models assigned to at least one running camera (status=True)
    running_cameras = (
        db.query(Camera.id)
        .filter(Camera.status == True, Camera.is_deleted == False)
        .all()
    )
    running_camera_ids = [cam.id for cam in running_cameras]

    if running_camera_ids:
        # Query models that are assigned to running cameras
        loaded_models_count = (
            db.query(AiModel.id)
            .join(CameraModel, AiModel.id == CameraModel.model_id)
            .filter(
                CameraModel.camera_id.in_(running_camera_ids),
                CameraModel.is_enabled == True,
            )
            .distinct()
            .count()
        )
    else:
        loaded_models_count = 0

    not_loaded_models = total_models - loaded_models_count

    models_summary = ModelsSummary(
        total=total_models,
        loaded=loaded_models_count,
        not_loaded=not_loaded_models,
        active=active_models,
        inactive=inactive_models,
    )

    # ===== EVENTS =====
    total_events = db.query(Event).count()

    # Get today's events (from midnight local time)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_events = db.query(Event).filter(Event.time >= today_start).count()

    events_summary = EventsSummary(today=today_events, total=total_events)

    # ===== STORAGE =====
    # Get disk usage for the data directory (where events are stored)
    try:
        # Try to get disk usage from the data/events directory
        from pathlib import Path

        data_path = Path(__file__).parent.parent.parent / "data" / "events"

        # Get disk usage of the partition where data is stored
        disk_usage = shutil.disk_usage(data_path if data_path.exists() else "/")

        total_gb = disk_usage.total / (1024**3)  # Convert to GB
        used_gb = disk_usage.used / (1024**3)
        used_percent = round((disk_usage.used / disk_usage.total) * 100, 2)

        # Determine status based on usage
        if used_percent >= 85:
            status = "critical"
        elif used_percent >= 70:
            status = "warning"
        else:
            status = "good"

        storage_summary = StorageSummary(
            used_percent=used_percent,
            used_gb=round(used_gb, 2),
            total_gb=round(total_gb, 2),
            status=status,
        )
    except Exception as e:
        # Fallback if unable to get disk usage
        storage_summary = StorageSummary(
            used_percent=0.0, used_gb=0.0, total_gb=0.0, status="unknown"
        )

    # ===== BUILD RESPONSE =====
    return SystemSummary(
        cameras=cameras_summary,
        models=models_summary,
        events=events_summary,
        storage=storage_summary,
    )


class SyncCameraResult(BaseModel):
    camera_id: str
    name: str


class SyncErrorResult(BaseModel):
    camera_id: Optional[str] = None
    name: Optional[str] = None
    error: str


class SynchronizationResult(BaseModel):
    started: List[SyncCameraResult]
    stopped: List[SyncCameraResult]
    reloaded: List[SyncCameraResult]
    already_synced: List[SyncCameraResult]
    errors: List[SyncErrorResult]
    summary: Dict[str, int]


class CameraMissingFieldsItem(BaseModel):
    camera_id: str
    name: str
    protocol: str
    missing_fields: List[str]
    warnings: List[str] = []


class CameraMissingFieldsSummary(BaseModel):
    total_cameras: int
    cameras_with_missing_fields: int
    vss_cameras_with_missing_fields: int
    items: List[CameraMissingFieldsItem]


VSS_REQUIRED_FIELDS = ["vss_base_url", "vss_username", "vss_password", "vss_device_id"]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _rtsp_url_looks_like_vss_realvideo(rtsp_url: str) -> bool:
    if not rtsp_url:
        return False
    parts = urlsplit(rtsp_url)
    return parts.path.lower().endswith("/vss/apipage/realvideo.html")


def _rtsp_url_has_vss_credentials(rtsp_url: str) -> bool:
    if not rtsp_url:
        return False
    query = parse_qs(urlsplit(rtsp_url).query, keep_blank_values=True)
    user = (query.get("username") or [""])[0].strip()
    password = (query.get("password") or [""])[0].strip()
    device_id = (query.get("deviceId") or [""])[0].strip()
    return bool(user and password and device_id)


@router.post("/sync", response_model=SynchronizationResult)
def synchronize_system(db: Session = Depends(get_db)):
    """
    Synchronize the system state between database and running cameras.

    This endpoint ensures:
    1. Cameras with status=True in database are running
    2. Cameras with status=False in database are stopped
    3. Database status is updated to match actual running state
    4. Models are reloaded for running cameras to ensure latest configuration

    Returns:
        SynchronizationResult: Detailed results of the synchronization process
    """
    sync_results = thread_manager.synchronize()

    # Convert to response format
    response = SynchronizationResult(
        started=[SyncCameraResult(**item) for item in sync_results["started"]],
        stopped=[SyncCameraResult(**item) for item in sync_results["stopped"]],
        reloaded=[SyncCameraResult(**item) for item in sync_results["reloaded"]],
        already_synced=[
            SyncCameraResult(**item) for item in sync_results["already_synced"]
        ],
        errors=[SyncErrorResult(**item) for item in sync_results["errors"]],
        summary={
            "started": len(sync_results["started"]),
            "stopped": len(sync_results["stopped"]),
            "reloaded": len(sync_results["reloaded"]),
            "already_synced": len(sync_results["already_synced"]),
            "errors": len(sync_results["errors"]),
        },
    )

    # Broadcast sync event via WebSocket
    if websocket_manager.event_loop is not None:
        sync_event_data = {
            "status": "completed",
            "summary": response.summary,
            "started_count": len(sync_results["started"]),
            "stopped_count": len(sync_results["stopped"]),
            "reloaded_count": len(sync_results["reloaded"]),
            "errors_count": len(sync_results["errors"]),
        }
        asyncio.run_coroutine_threadsafe(
            websocket_manager.broadcast_sync(sync_event_data),
            websocket_manager.event_loop,
        )

    return response


@router.get("/cameras/missing-fields", response_model=CameraMissingFieldsSummary)
def get_cameras_missing_fields(db: Session = Depends(get_db)):
    """Inspect camera records and report missing required fields.

    Focuses on fields that often cause start/connect failures.
    """
    cameras = db.query(Camera).filter(Camera.is_deleted == False).all()
    items: List[CameraMissingFieldsItem] = []
    vss_missing_count = 0

    for cam in cameras:
        protocol = str(cam.protocol or "").strip().upper()
        rtsp_url = str(cam.rtsp_url or "").strip()
        missing_fields: List[str] = []
        warnings: List[str] = []

        if _is_blank(cam.name):
            missing_fields.append("name")
        if cam.location_id is None:
            missing_fields.append("location_id")

        is_vss = protocol == "VSS" or _rtsp_url_looks_like_vss_realvideo(rtsp_url)
        if is_vss:
            for field_name in VSS_REQUIRED_FIELDS:
                if _is_blank(getattr(cam, field_name, None)):
                    missing_fields.append(field_name)

            if _is_blank(cam.vss_channel):
                warnings.append(
                    "vss_channel is empty (defaults to '1' in URL builders)"
                )

            if rtsp_url and not _rtsp_url_has_vss_credentials(rtsp_url):
                warnings.append(
                    "rtsp_url looks like VSS but query params username/password/deviceId are incomplete"
                )
        else:
            if _is_blank(rtsp_url):
                missing_fields.append("rtsp_url")

        if missing_fields or warnings:
            items.append(
                CameraMissingFieldsItem(
                    camera_id=str(cam.id),
                    name=cam.name,
                    protocol=protocol or "UNKNOWN",
                    missing_fields=missing_fields,
                    warnings=warnings,
                )
            )

        if is_vss and missing_fields:
            vss_missing_count += 1

    return CameraMissingFieldsSummary(
        total_cameras=len(cameras),
        cameras_with_missing_fields=sum(1 for item in items if item.missing_fields),
        vss_cameras_with_missing_fields=vss_missing_count,
        items=items,
    )
