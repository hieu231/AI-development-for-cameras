# src/models/camera_model.py
from sqlalchemy import Column, Boolean, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone, timedelta
from src.models.base import BaseModel


class CameraModel(BaseModel):
    """Camera Model Assignment - maps AI models to cameras"""
    __tablename__ = "camera_models"

    # Foreign keys
    camera_id = Column(
        UUID(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ai_models.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Assignment settings
    is_enabled = Column(Boolean, nullable=False, default=True)
    additional_parameters = Column(JSONB, nullable=False, default=dict)
    assigned_at = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(),
        nullable=False
    )

    # Relationships
    camera = relationship("Camera", back_populates="camera_models")
    ai_model = relationship("AiModel", back_populates="camera_models")

    def __repr__(self):
        return f"<CameraModel(id={self.id}, camera_id={self.camera_id}, model_id={self.model_id}, enabled={self.is_enabled})>"

