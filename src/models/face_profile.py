"""
Face Profile and Face Photo Models
Database models for storing face recognition data
"""
from sqlalchemy import Column, String, Boolean, Date, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from src.models.base import BaseModel


class FacePhoto(BaseModel):
    """Face Photo model - represents a single face photo"""
    __tablename__ = "face_photo"

    image_path = Column(String(500), nullable=True)
    face_embedding = Column(Vector(512), nullable=True)
    is_primary = Column(Boolean, nullable=False, default=False)

    def __repr__(self):
        return f"<FacePhoto(id={self.id}, is_primary={self.is_primary})>"


class FaceProfile(BaseModel):
    """Face Profile model - represents a person's face profile"""
    __tablename__ = "face_profiles"

    employee_id = Column(String(50), nullable=False, unique=True, index=True)
    employee_name = Column(String(255), nullable=False)
    avg_face_embedding = Column(Vector(512), nullable=True)
    photo_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False, default=list)
    profile_metadata = Column(JSONB, nullable=False, default=dict)
    consent_date = Column(Date, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)

    def __repr__(self):
        return f"<FaceProfile(id={self.id}, employee_id='{self.employee_id}', name='{self.employee_name}', active={self.is_active})>"
