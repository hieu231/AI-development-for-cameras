"""
Face Recognition API Endpoints
Handles face profile management and photo operations
"""
import os
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import logging
import numpy as np

from src.database import get_db
from src.face_recognition.schemas import (
    FaceProfileCreate,
    FaceProfileUpdate,
    FaceProfileResponse,
    FacePhotoResponse,
    ReplacePhotoRequest,
    ReplaceAllPhotosRequest,
    ValidationResponse
)
from src.face_recognition.repository import FaceProfileRepository
from src.face_recognition.validator import get_validator, FacePhotoValidator
from src.face_recognition.face_engine import get_face_engine

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/face_recognition", tags=["Face Recognition"])


def check_face_recognition_enabled():
    """Check if face recognition is enabled"""
    enable_face_recognition = os.getenv('ENABLE_FACE_RECOGNITION', 'true').lower() == 'true'
    if not enable_face_recognition:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Face recognition feature is disabled"
        )


# ============================================
# Face Profile Endpoints
# ============================================

@router.post("/profiles", response_model=FaceProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_face_profile(
    profile_data: FaceProfileCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new face profile with photos

    - Validates that each photo contains exactly one face
    - Validates that all photos are of the same person
    - Requires exactly 3 photos in base64 format
    """
    check_face_recognition_enabled()

    validator = FacePhotoValidator(min_resolution=0)  # No resolution limit
    engine = get_face_engine()
    repo = FaceProfileRepository(db)

    # Check if employee_id already exists
    existing = repo.get_face_profile_by_employee_id(profile_data.employee_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Employee ID '{profile_data.employee_id}' already exists"
        )

    # Validate all photos
    validation_result = validator.validate_multiple_photos(
        profile_data.images_base64,
        check_same_person=True
    )

    if not validation_result['valid']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_result['error']
        )

    embeddings = validation_result['embeddings']

    # Compute average embedding
    avg_embedding = engine.compute_average_embedding([np.array(emb) for emb in embeddings])

    # Save photos and create records
    photo_ids = []
    profile_id = UUID(int=0)  # Temporary, will be replaced

    try:
        # Create profile first to get ID
        profile = repo.create_face_profile(
            employee_id=profile_data.employee_id,
            employee_name=profile_data.employee_name,
            photo_ids=[],
            avg_embedding=avg_embedding.tolist(),
            profile_metadata=profile_data.profile_metadata or {},
            consent_date=profile_data.consent_date
        )
        profile_id = profile.id

        # Save photos
        profile_dir = repo.get_profile_storage_path(profile_id)

        for i, (img_b64, embedding) in enumerate(zip(profile_data.images_base64, embeddings)):
            # Save image
            filename = f"photo_{i + 1}"
            success, file_path, error = validator.save_image(img_b64, profile_dir, filename)

            if not success:
                raise Exception(f"Failed to save photo {i + 1}: {error}")

            # Create photo record
            photo = repo.create_face_photo(
                image_path=file_path,
                face_embedding=embedding,
                is_primary=(i == 0)
            )
            photo_ids.append(photo.id)

        # Update profile with photo IDs
        repo.update_profile_photos(profile_id, photo_ids, avg_embedding.tolist())

        db.commit()

        # Return response
        profile = repo.get_face_profile(profile_id)
        return FaceProfileResponse(
            id=profile.id,
            employee_id=profile.employee_id,
            employee_name=profile.employee_name,
            photo_ids=profile.photo_ids,
            profile_metadata=profile.profile_metadata,
            consent_date=profile.consent_date,
            is_active=profile.is_active,
            created_at=profile.created_at,
            updated_at=profile.updated_at,
            num_photos=len(profile.photo_ids)
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error creating face profile: {e}", exc_info=True)

        # Clean up: delete profile and photos
        if profile_id:
            try:
                repo.delete_face_profile(profile_id)
                db.commit()
            except:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create face profile: {str(e)}"
        )


@router.get("/profiles", response_model=List[FaceProfileResponse])
async def get_all_face_profiles(
    active_only: bool = True,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all face profiles"""
    check_face_recognition_enabled()

    repo = FaceProfileRepository(db)
    profiles = repo.get_all_face_profiles(active_only=active_only, skip=skip, limit=limit)

    return [
        FaceProfileResponse(
            id=p.id,
            employee_id=p.employee_id,
            employee_name=p.employee_name,
            photo_ids=p.photo_ids,
            profile_metadata=p.profile_metadata,
            consent_date=p.consent_date,
            is_active=p.is_active,
            created_at=p.created_at,
            updated_at=p.updated_at,
            num_photos=len(p.photo_ids)
        )
        for p in profiles
    ]


@router.get("/profiles/{profile_id}", response_model=FaceProfileResponse)
async def get_face_profile(
    profile_id: UUID,
    db: Session = Depends(get_db)
):
    """Get face profile by ID"""
    check_face_recognition_enabled()

    repo = FaceProfileRepository(db)
    profile = repo.get_face_profile(profile_id)

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    return FaceProfileResponse(
        id=profile.id,
        employee_id=profile.employee_id,
        employee_name=profile.employee_name,
        photo_ids=profile.photo_ids,
        profile_metadata=profile.profile_metadata,
        consent_date=profile.consent_date,
        is_active=profile.is_active,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        num_photos=len(profile.photo_ids)
    )


@router.put("/profiles/{profile_id}", response_model=FaceProfileResponse)
async def update_face_profile(
    profile_id: UUID,
    profile_data: FaceProfileUpdate,
    db: Session = Depends(get_db)
):
    """Update face profile information (not photos)"""
    check_face_recognition_enabled()

    repo = FaceProfileRepository(db)

    try:
        profile = repo.update_face_profile(
            profile_id=profile_id,
            employee_id=profile_data.employee_id,
            employee_name=profile_data.employee_name,
            profile_metadata=profile_data.profile_metadata,
            consent_date=profile_data.consent_date,
            is_active=profile_data.is_active
        )
    except ValueError as e:
        # Handle unique constraint violation for employee_id
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    db.commit()

    return FaceProfileResponse(
        id=profile.id,
        employee_id=profile.employee_id,
        employee_name=profile.employee_name,
        photo_ids=profile.photo_ids,
        profile_metadata=profile.profile_metadata,
        consent_date=profile.consent_date,
        is_active=profile.is_active,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        num_photos=len(profile.photo_ids)
    )


@router.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face_profile(
    profile_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete face profile and all associated photos"""
    check_face_recognition_enabled()

    repo = FaceProfileRepository(db)
    success = repo.delete_face_profile(profile_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    db.commit()


# ============================================
# Photo Management Endpoints
# ============================================

@router.put("/profiles/{profile_id}/photos/{photo_id}", response_model=ValidationResponse)
async def replace_single_photo(
    profile_id: UUID,
    photo_id: UUID,
    photo_data: ReplacePhotoRequest,
    db: Session = Depends(get_db)
):
    """
    Replace a single photo in a profile

    - Validates that the new photo is of the same person
    - If profile has only 1 photo, replacement is always allowed
    """
    check_face_recognition_enabled()

    validator = FacePhotoValidator(min_resolution=0)  # No resolution limit
    engine = get_face_engine()
    repo = FaceProfileRepository(db)

    # Get profile
    profile = repo.get_face_profile(profile_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    # Check if photo belongs to profile
    if photo_id not in profile.photo_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Photo {photo_id} does not belong to profile {profile_id}"
        )

    # Get existing photos (excluding the one being replaced)
    other_photo_ids = [pid for pid in profile.photo_ids if pid != photo_id]

    # If only 1 photo, allow replacement without validation
    if len(profile.photo_ids) == 1:
        # Just validate the new photo has exactly one face
        validation_result = validator.validate_single_photo(photo_data.image_base64)

        if not validation_result['valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result['error']
            )

        new_embedding = validation_result['embedding']

    else:
        # Get other photos' embeddings
        other_photos = repo.get_face_photos(other_photo_ids)
        other_embeddings = [np.array(p.face_embedding) for p in other_photos]

        # Validate new photo against existing ones
        validation_result = validator.validate_photo_against_existing(
            photo_data.image_base64,
            other_embeddings
        )

        if not validation_result['valid']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=validation_result['error']
            )

        new_embedding = validation_result['embedding']

    try:
        # Delete old photo
        repo.delete_face_photo(photo_id)

        # Save new photo
        profile_dir = repo.get_profile_storage_path(profile_id)
        photo_index = profile.photo_ids.index(photo_id)
        filename = f"photo_{photo_index + 1}"

        success, file_path, error = validator.save_image(
            photo_data.image_base64,
            profile_dir,
            filename
        )

        if not success:
            raise Exception(f"Failed to save photo: {error}")

        # Create new photo record
        new_photo = repo.create_face_photo(
            image_path=file_path,
            face_embedding=new_embedding,
            is_primary=(photo_index == 0)
        )

        # Update profile's photo_ids
        new_photo_ids = profile.photo_ids.copy()
        new_photo_ids[photo_index] = new_photo.id

        # Recalculate average embedding
        all_photos = repo.get_face_photos(new_photo_ids)
        all_embeddings = [np.array(p.face_embedding) for p in all_photos]
        avg_embedding = engine.compute_average_embedding(all_embeddings)

        # Update profile
        repo.update_profile_photos(profile_id, new_photo_ids, avg_embedding.tolist())

        db.commit()

        return ValidationResponse(
            success=True,
            message="Photo replaced successfully"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error replacing photo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to replace photo: {str(e)}"
        )


@router.delete("/profiles/{profile_id}/photos/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_single_photo(
    profile_id: UUID,
    photo_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a single photo from a profile

    - Cannot delete if profile has only 1 photo
    - Recalculates average embedding after deletion
    """
    check_face_recognition_enabled()

    engine = get_face_engine()
    repo = FaceProfileRepository(db)

    # Get profile
    profile = repo.get_face_profile(profile_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    # Check if photo belongs to profile
    if photo_id not in profile.photo_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Photo {photo_id} does not belong to profile {profile_id}"
        )

    # Cannot delete if only 1 photo
    if len(profile.photo_ids) <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the last photo. Profile must have at least one photo."
        )

    try:
        # Delete photo
        repo.delete_face_photo(photo_id)

        # Update profile's photo_ids
        new_photo_ids = [pid for pid in profile.photo_ids if pid != photo_id]

        # Recalculate average embedding
        remaining_photos = repo.get_face_photos(new_photo_ids)
        remaining_embeddings = [np.array(p.face_embedding) for p in remaining_photos]
        avg_embedding = engine.compute_average_embedding(remaining_embeddings)

        # Update profile
        repo.update_profile_photos(profile_id, new_photo_ids, avg_embedding.tolist())

        db.commit()

    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting photo: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete photo: {str(e)}"
        )


@router.put("/profiles/{profile_id}/photos", response_model=ValidationResponse)
async def replace_all_photos(
    profile_id: UUID,
    photos_data: ReplaceAllPhotosRequest,
    db: Session = Depends(get_db)
):
    """
    Replace all photos of a profile

    - Validates that all new photos are of the same person
    - Requires exactly 3 photos
    """
    check_face_recognition_enabled()

    validator = FacePhotoValidator(min_resolution=0)  # No resolution limit
    engine = get_face_engine()
    repo = FaceProfileRepository(db)

    # Get profile
    profile = repo.get_face_profile(profile_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Face profile {profile_id} not found"
        )

    # Validate all new photos
    validation_result = validator.validate_multiple_photos(
        photos_data.images_base64,
        check_same_person=True
    )

    if not validation_result['valid']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=validation_result['error']
        )

    embeddings = validation_result['embeddings']

    try:
        # Delete all old photos
        for photo_id in profile.photo_ids:
            repo.delete_face_photo(photo_id)

        # Save new photos
        new_photo_ids = []
        profile_dir = repo.get_profile_storage_path(profile_id)

        for i, (img_b64, embedding) in enumerate(zip(photos_data.images_base64, embeddings)):
            filename = f"photo_{i + 1}"
            success, file_path, error = validator.save_image(img_b64, profile_dir, filename)

            if not success:
                raise Exception(f"Failed to save photo {i + 1}: {error}")

            # Create photo record
            photo = repo.create_face_photo(
                image_path=file_path,
                face_embedding=embedding,
                is_primary=(i == 0)
            )
            new_photo_ids.append(photo.id)

        # Compute new average embedding
        avg_embedding = engine.compute_average_embedding([np.array(emb) for emb in embeddings])

        # Update profile
        repo.update_profile_photos(profile_id, new_photo_ids, avg_embedding.tolist())

        db.commit()

        return ValidationResponse(
            success=True,
            message="All photos replaced successfully"
        )

    except Exception as e:
        db.rollback()
        logger.error(f"Error replacing all photos: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to replace photos: {str(e)}"
        )
