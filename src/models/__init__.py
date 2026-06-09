# src/models/__init__.py
from src.models.base import BaseModel, TimestampMixin, UUIDMixin
from src.models.camera import Camera
from src.models.camera_model import CameraModel
from src.models.ai_model import AiModel
from src.models.location import Location
from src.models.camera_spec import CameraSpec
from src.models.event import Event
from src.models.performance import Performance

__all__ = [
    "BaseModel",
    "TimestampMixin",
    "UUIDMixin",
    "Camera",
    "CameraModel",
    "AiModel",
    "Location",
    "CameraSpec",
    "Event",
    "Performance",
]
