# src/models/event.py
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from src.models.base import BaseModel


class Event(BaseModel):
    """Event model - stores detection events from cameras"""
    __tablename__ = "events"

    # Event timestamp (when the event occurred) - naive local time (matches image filename)
    time = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(),
        nullable=False,
        index=True
    )

    # Foreign keys
    camera_id = Column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_models.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    # Event data - using 'detection_data' attribute mapped to 'metadata' column to avoid SQLAlchemy reserved name
    detection_data = Column("metadata", JSONB, nullable=False, default=dict)
    image_path = Column(String, nullable=True)
    is_synced = Column(Boolean, nullable=False, default=False)
    ready_for_mqtt_deployment = Column(Boolean, nullable=False, default=True)  # Mark events as ready for MQTT sync
    
    # Alert level for filtering and categorization (high/low)
    alert_level = Column(String(20), nullable=False, default="low", server_default="low")

    # Relationships
    camera = relationship("Camera", back_populates="events")
    ai_model = relationship("AiModel", back_populates="events")

    def __repr__(self):
        return f"<Event(id={self.id}, camera_id={self.camera_id}, time={self.time})>"

