# src/models/camera.py
from sqlalchemy import Column, String, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.models.base import BaseModel


class Camera(BaseModel):
    """Camera model - represents a physical camera in the system"""

    __tablename__ = "cameras"

    name = Column(String, nullable=False)
    rtsp_url = Column(String, nullable=False)

    # Foreign keys
    location_id = Column(
        UUID(as_uuid=True),
        ForeignKey("locations.id", ondelete="SET NULL"),
        nullable=False,
        index=True,
    )

    # Camera settings
    raw_resolution = Column(String, nullable=True)
    preprocess_resolution = Column(String, nullable=True)
    status = Column(Boolean, nullable=False, default=True)
    is_deleted = Column(Boolean, nullable=False, default=False, server_default="false")
    camera_spec_id = Column(
        UUID(as_uuid=True), ForeignKey("camera_spec.id"), nullable=True
    )

    # VSS protocol fields (optional – used when protocol="VSS")
    protocol = Column(String, nullable=True)
    vss_base_url = Column(String, nullable=True)
    vss_username = Column(String, nullable=True)
    vss_password = Column(String, nullable=True)
    vss_device_id = Column(String, nullable=True)
    vss_channel = Column(String, nullable=True)

    # Relationships
    location = relationship("Location")
    camera_spec = relationship("CameraSpec")
    events = relationship(
        "Event", back_populates="camera", cascade="all, delete-orphan"
    )
    camera_models = relationship(
        "CameraModel", back_populates="camera", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Camera(id={self.id}, name='{self.name}', status={self.status})>"
