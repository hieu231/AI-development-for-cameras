"""
Pydantic schemas for face recognition API
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import date, datetime
import base64


class FacePhotoBase(BaseModel):
    """Base schema for face photo"""
    image_path: Optional[str] = None
    is_primary: bool = False


class FacePhotoCreate(BaseModel):
    """Schema for creating face photo (with base64 image)"""
    image_base64: str = Field(..., description="Base64 encoded image")
    is_primary: bool = False

    @field_validator('image_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 string"""
        try:
            # Try to decode to verify it's valid base64
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 string")


class FacePhotoResponse(BaseModel):
    """Schema for face photo response"""
    id: UUID
    image_path: Optional[str] = None
    is_primary: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class FaceProfileBase(BaseModel):
    """Base schema for face profile"""
    employee_id: str = Field(..., max_length=50, description="Employee ID (unique)")
    employee_name: str = Field(..., max_length=255, description="Employee name")
    profile_metadata: Optional[Dict[str, Any]] = None
    consent_date: Optional[date] = None


class FaceProfileCreate(FaceProfileBase):
    """Schema for creating face profile"""
    images_base64: List[str] = Field(
        ...,
        min_length=3,
        max_length=20,
        description=(
            "Between 3 and 20 face photos (base64). More photos → more"
            " stable averaged embedding, but each one runs MTCNN+FaceNet"
            " at registration time."
        ),
    )

    @field_validator('images_base64')
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        """Validate all images are valid base64"""
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("All images must be valid base64 strings")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "employee_id": "EMP001",
                    "employee_name": "Nguyen Van A",
                    "images_base64": [
                        "iVBORw0KGgoAAAANS...",
                        "iVBORw0KGgoAAAANS...",
                        "iVBORw0KGgoAAAANS..."
                    ],
                    "consent_date": "2025-01-01"
                }
            ]
        }
    }


class FaceProfileUpdate(BaseModel):
    """Schema for updating face profile"""
    employee_id: Optional[str] = Field(None, max_length=50)
    employee_name: Optional[str] = Field(None, max_length=255)
    profile_metadata: Optional[Dict[str, Any]] = None
    consent_date: Optional[date] = None
    is_active: Optional[bool] = None


class FaceProfileResponse(BaseModel):
    """Schema for face profile response"""
    id: UUID
    employee_id: str
    employee_name: str
    photo_ids: List[UUID]
    profile_metadata: Dict[str, Any]
    consent_date: Optional[date]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    num_photos: int = 0

    model_config = {"from_attributes": True}


class ReplacePhotoRequest(BaseModel):
    """Schema for replacing a single photo"""
    image_base64: str = Field(..., description="New photo (base64)")

    @field_validator('image_base64')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 string"""
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 string")


class ReplaceAllPhotosRequest(BaseModel):
    """Schema for replacing all photos of a profile"""
    images_base64: List[str] = Field(
        ...,
        min_length=3,
        max_length=20,
        description=(
            "Between 3 and 20 new face photos (base64) — replaces the"
            " profile's existing photos and recomputes the averaged"
            " embedding."
        ),
    )

    @field_validator('images_base64')
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        """Validate all images are valid base64"""
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("All images must be valid base64 strings")
        return v


class ValidationResponse(BaseModel):
    """Schema for validation response"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class FaceDetectionEvent(BaseModel):
    """Schema for face detection event metadata"""
    type: str = Field(..., description="'known' or 'unknown'")
    employee_id: Optional[str] = None
    employee_name: Optional[str] = None
    confidence: float
    face_bbox: Optional[List[int]] = None
    timestamp: datetime
