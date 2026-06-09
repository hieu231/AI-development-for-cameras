"""
Performance Model - Stores system performance metrics
"""
from sqlalchemy import Column, Float, Integer, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone, timedelta
from src.models.base import BaseModel


class Performance(BaseModel):
    """System performance metrics"""
    __tablename__ = "performance_metrics"

    # Timestamp when metrics were collected - naive local time
    timestamp = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(),
        nullable=False,
        index=True
    )

    # System metrics
    cpu_percent = Column(Float, nullable=True)  # CPU usage %
    memory_percent = Column(Float, nullable=True)  # Memory usage %
    memory_used_mb = Column(Float, nullable=True)  # Memory used in MB

    # Camera metrics
    total_cameras = Column(Integer, nullable=True, default=0)  # Total cameras
    running_cameras = Column(Integer, nullable=True, default=0)  # Running cameras
    total_models = Column(Integer, nullable=True, default=0)  # Total models loaded

    # Processing metrics
    total_fps = Column(Float, nullable=True, default=0.0)  # Total FPS across all cameras
    avg_fps_per_camera = Column(Float, nullable=True, default=0.0)  # Average FPS per camera
    total_events = Column(Integer, nullable=True, default=0)  # Total events in database
    events_last_hour = Column(Integer, nullable=True, default=0)  # Events in last hour

    # Per-camera details (JSONB for better performance)
    camera_details = Column(
        JSONB,
        nullable=True,
        default=dict
    )  # {camera_id: {fps, models, queue_size}}

    def __repr__(self):
        return f"<Performance(id={self.id}, timestamp={self.timestamp}, CPU={self.cpu_percent}%, MEM={self.memory_percent}%, FPS={self.total_fps})>"
