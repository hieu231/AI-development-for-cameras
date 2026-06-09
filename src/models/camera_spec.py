# src/models/camera_spec.py
from sqlalchemy import Column, String, Integer, Boolean, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from datetime import datetime, timezone, timedelta
from src.database import Base

class CameraSpec(Base):
    __tablename__ = "camera_spec"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    manufacturer = Column(String(100), nullable=False)
    model_series = Column(String(100), nullable=True)
    category = Column(String(50), nullable=True)
    resolution = Column(String(20), nullable=True)
    max_fps = Column(Integer, nullable=True)
    ptz_support = Column(Boolean, nullable=False, default=False)
    audio_support = Column(Boolean, nullable=False, default=False)
    ir_support = Column(Boolean, nullable=False, default=False)
    ai_support = Column(Boolean, nullable=False, default=False)
    onvif_profile = Column(String(20), nullable=True)
    brand_website = Column(String(255), nullable=True)
    brand_description = Column(String, nullable=True)
    rtsp_format = Column(String, nullable=True)  # RTSP URL format template (e.g., "rtsp://username:password@IP:554/stream/main")
    spec_metadata = Column('metadata', JSONB, nullable=False, default={})
    is_active = Column(Boolean, nullable=False, default=True)

    # Naive local time for timestamps
    created_at = Column(DateTime(timezone=False), nullable=False, default=lambda: datetime.now())
    updated_at = Column(DateTime(timezone=False), nullable=False, default=lambda: datetime.now(), onupdate=lambda: datetime.now())
