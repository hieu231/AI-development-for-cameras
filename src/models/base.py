# src/models/base.py
"""
Base model for all database models
Provides common fields and functionality
"""
from sqlalchemy import Column, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
import uuid
from datetime import datetime, timezone, timedelta
from src.database import Base


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps - naive local time"""

    created_at = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(),
        onupdate=lambda: datetime.now(),
        nullable=False
    )


class UUIDMixin:
    """Mixin for UUID primary key"""

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False
    )


class BaseModel(Base, UUIDMixin, TimestampMixin):
    """
    Base model that all models should inherit from
    Provides:
    - UUID primary key (id)
    - Timestamp fields (created_at, updated_at)
    - Common methods
    """
    __abstract__ = True

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }

    def __repr__(self):
        """String representation"""
        return f"<{self.__class__.__name__}(id={self.id})>"
