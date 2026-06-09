# src/models/ai_model.py
from sqlalchemy import Column, String, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from src.models.base import BaseModel


class AiModel(BaseModel):
    """AI Model - represents an AI detection model"""
    __tablename__ = "ai_models"

    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_path = Column(String, nullable=True)  # Nullable for models that don't need files (e.g., Face Recognition, ALPR)
    parameters = Column(JSONB, nullable=False, default=dict)
    is_active = Column(Boolean, nullable=False, default=True)
    is_latest_used = Column(Boolean, nullable=False, default=False)
    previous_used_version_id = Column(UUID(as_uuid=True), nullable=True)
    changelog = Column(String, nullable=True)

    # Relationships
    events = relationship("Event", back_populates="ai_model")
    camera_models = relationship("CameraModel", back_populates="ai_model")

    @property
    def model_type(self):
        """Get model_type from parameters"""
        return self.parameters.get('model_type', 'object_detection')

    def __repr__(self):
        return f"<AiModel(id={self.id}, name='{self.name}', version='{self.version}')>"
