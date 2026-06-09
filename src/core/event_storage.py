# src/services/event_storage.py
import cv2
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy.orm import Session

from src.utils.alert_levels import AlertLevel, ALERT_LEVEL_MAP, AlertColor
from src.utils.datetime_utils import serialize_utc_datetime
from src.utils.evidence_path import resolve_writable_evidence_base_path
from src.ai_models.base_model import DetectionResult
from src.models.event import Event
from src.models.camera import Camera
from src.core.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

# Đường dẫn lưu ảnh bằng chứng
EVIDENCE_BASE = resolve_writable_evidence_base_path()


class EventStorageService:
    """Lưu event + ảnh + broadcast realtime qua WebSocket"""

    def save_image(self, camera_id: str | UUID, frame: np.ndarray) -> Optional[str]:
        if frame is None or frame.size == 0:
            return None

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S.%f")[:-3]
        dir_path = EVIDENCE_BASE / date_str / str(camera_id)
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{time_str}.jpg"
        filepath = dir_path / filename

        if cv2.imwrite(str(filepath), frame):
            return f"/evidence_image/{date_str}/{camera_id}/{filename}"
        else:
            logger.error(f"Failed to save image: {filepath}")
            return None

    def save_event_image(
        self,
        camera_id: str | UUID,
        frame: np.ndarray,
        event_time: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Save event image with specific event time for filename consistency
        
        Args:
            camera_id: Camera ID
            frame: Image frame to save
            event_time: Optional datetime for filename (uses current time if not provided)
        
        Returns:
            Image path string or None if failed
        """
        if frame is None or frame.size == 0:
            return None

        # Use provided event_time or current time
        now = event_time if event_time is not None else datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S.%f")[:-3]
        dir_path = EVIDENCE_BASE / date_str / str(camera_id)
        dir_path.mkdir(parents=True, exist_ok=True)

        filename = f"{time_str}.jpg"
        filepath = dir_path / filename

        if cv2.imwrite(str(filepath), frame):
            return f"/evidence_image/{date_str}/{camera_id}/{filename}"
        else:
            logger.error(f"Failed to save image: {filepath}")
            return None

    def store_detection_event(
        self,
        detection: DetectionResult,
        camera_id: UUID | str,
        model_id: UUID | str,
        frame: Optional[np.ndarray] = None,
        db: Session = None,
        camera_name: str = "Unknown Camera",
        model_name: str = "Unknown Model",
        model_type: str = "unknown",
    ) -> Optional[UUID]:
        """
        Lưu event vào PostgreSQL + Gửi realtime qua WebSocket
        """
        if not detection.event or db is None:
            return None

        camera_record = db.query(Camera).filter(Camera.id == UUID(str(camera_id))).first()
        if (
            camera_record is None
            or bool(getattr(camera_record, "is_deleted", False))
            or not bool(getattr(camera_record, "status", False))
        ):
            logger.info(
                "Skipping detection event for inactive camera %s "
                "(exists=%s, deleted=%s, status=%s)",
                camera_id,
                camera_record is not None,
                bool(getattr(camera_record, "is_deleted", False))
                if camera_record is not None
                else None,
                bool(getattr(camera_record, "status", False))
                if camera_record is not None
                else None,
            )
            return None

        # 1. Lưu ảnh
        image_path = self.save_image(camera_id, frame) if frame is not None else None

        # 2. Xác định mức độ cảnh báo
        alert_info = detection.metadata.get("alert", {})
        level = AlertLevel.from_value(alert_info.get("level", "low"), AlertLevel.LOW)

        # 3. Tạo Event
        event = Event(
            time=datetime.now(),
            camera_id=UUID(str(camera_id)),
            model_id=UUID(str(model_id)),
            detection_data=detection.metadata,
            image_path=image_path,
            alert_level=level.value,
        )

        try:
            db.add(event)
            db.commit()
            db.refresh(event)

            # 4. BROADCAST REALTIME
            event_payload = {
                "event_id": str(event.id),
                "camera_id": str(event.camera_id),
                "camera_name": camera_name,
                "model_id": str(event.model_id),
                "model_name": model_name,
                "model_type": model_type,
                "time": serialize_utc_datetime(event.time),
                "metadata": event.detection_data,
                "image_path": event.image_path,
                "alert_level": event.alert_level,
            }
            asyncio.create_task(websocket_manager.broadcast_event(event_payload))

            # Gửi riêng cho /ws/alerts
            if event.alert_level in ["high", "low"]:
                alert_payload = {
                    "alert_level": event.alert_level,
                    "alert_color": ALERT_LEVEL_MAP.get(
                        AlertLevel.from_value(event.alert_level, AlertLevel.LOW),
                        AlertColor.GRAY,
                    ).value,
                    "message": alert_info.get("message", "Phát hiện sự kiện"),
                    "confidence": alert_info.get("confidence", 0.0),
                    "detected_objects": alert_info.get("detected_objects", []),
                    "camera_id": str(camera_id),
                    "camera_name": camera_name,
                    "timestamp": serialize_utc_datetime(event.time),
                    "image_path": image_path,
                }
                asyncio.create_task(websocket_manager.broadcast_alert(alert_payload))

            logger.info(f"EVENT SAVED + BROADCAST | {level.value.upper()} | {camera_name}")
            return event.id

        except Exception as e:
            logger.error(f"Failed to save event: {e}", exc_info=True)
            db.rollback()
            return None

    def cleanup_old_images(self, days: int = 30):
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0
        for date_dir in EVIDENCE_BASE.iterdir():
            if date_dir.is_dir():
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date.date() < cutoff.date():
                        import shutil
                        shutil.rmtree(date_dir)
                        deleted += 1
                except ValueError:
                    continue
        logger.info(f"Cleaned {deleted} old evidence folders")

    def get_stats(self):
        total = size = cameras = dates = 0
        for date_dir in EVIDENCE_BASE.iterdir():
            if not date_dir.is_dir(): continue
            dates += 1
            for cam_dir in date_dir.iterdir():
                if not cam_dir.is_dir(): continue
                cameras += 1
                for img in cam_dir.glob("*.jpg"):
                    total += 1
                    size += img.stat().st_size
        return {
            "total_images": total,
            "total_size_mb": round(size / (1024**2), 2),
            "date_folders": dates,
            "cameras": cameras,
        }


# Singleton instance
event_storage_service = EventStorageService()
