# src/models/location.py
from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from src.models.base import BaseModel


class Location(BaseModel):
    """Location model - represents physical locations/sites"""
    __tablename__ = "locations"

    name = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)

    # Relationships
    cameras = relationship("Camera", back_populates="location")

    def __repr__(self):
        return f"<Location(id={self.id}, name='{self.name}')>"