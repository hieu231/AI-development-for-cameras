"""
Repository layer for face recognition
Handles database operations for face profiles and photos
"""
import os
import shutil
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime, date
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import logging

from src.models.face_profile import FaceProfile, FacePhoto
from src.face_recognition.face_engine import get_face_engine

logger = logging.getLogger(__name__)


class FaceProfileRepository:
    """Repository for face profile operations"""

    def __init__(self, db: Session, face_photos_dir: str = "face_photos"):
        """
        Initialize repository

        Args:
            db: Database session
            face_photos_dir: Base directory for storing face photos
        """
        self.db = db
        self.face_photos_dir = face_photos_dir
        self.engine = get_face_engine()

    # ============================================
    # Face Photo Operations
    # ============================================

    def create_face_photo(
        self,
        image_path: str,
        face_embedding: List[float],
        is_primary: bool = False
    ) -> FacePhoto:
        """
        Create a new face photo record

        Args:
            image_path: Path to saved image
            face_embedding: Face embedding vector
            is_primary: Whether this is the primary photo

        Returns:
            FacePhoto instance
        """
        photo = FacePhoto(
            id=uuid4(),
            image_path=image_path,
            face_embedding=face_embedding,
            is_primary=is_primary
        )
        self.db.add(photo)
        self.db.flush()
        return photo

    def get_face_photo(self, photo_id: UUID) -> Optional[FacePhoto]:
        """Get face photo by ID"""
        return self.db.query(FacePhoto).filter(FacePhoto.id == photo_id).first()

    def get_face_photos(self, photo_ids: List[UUID]) -> List[FacePhoto]:
        """Get multiple face photos by IDs"""
        return self.db.query(FacePhoto).filter(FacePhoto.id.in_(photo_ids)).all()

    def delete_face_photo(self, photo_id: UUID) -> bool:
        """
        Delete face photo and its file

        Args:
            photo_id: Photo ID

        Returns:
            True if deleted, False if not found
        """
        photo = self.get_face_photo(photo_id)
        if not photo:
            return False

        # Delete file if exists
        if photo.image_path and os.path.exists(photo.image_path):
            try:
                os.remove(photo.image_path)
            except Exception as e:
                logger.warning(f"Failed to delete photo file {photo.image_path}: {e}")

        # Delete record
        self.db.delete(photo)
        self.db.flush()
        return True

    # ============================================
    # Face Profile Operations
    # ============================================

    def create_face_profile(
        self,
        employee_id: str,
        employee_name: str,
        photo_ids: List[UUID],
        avg_embedding: List[float],
        profile_metadata: Optional[Dict[str, Any]] = None,
        consent_date: Optional[date] = None
    ) -> FaceProfile:
        """
        Create a new face profile

        Args:
            employee_id: Employee ID (unique)
            employee_name: Employee name
            photo_ids: List of photo IDs
            avg_embedding: Average face embedding
            profile_metadata: Additional metadata
            consent_date: Consent date

        Returns:
            FaceProfile instance
        """
        profile = FaceProfile(
            id=uuid4(),
            employee_id=employee_id,
            employee_name=employee_name,
            photo_ids=photo_ids,
            avg_face_embedding=avg_embedding,
            profile_metadata=profile_metadata or {},
            consent_date=consent_date,
            is_active=True
        )
        self.db.add(profile)
        self.db.flush()
        return profile

    def get_face_profile(self, profile_id: UUID) -> Optional[FaceProfile]:
        """Get face profile by ID"""
        return self.db.query(FaceProfile).filter(FaceProfile.id == profile_id).first()

    def get_face_profile_by_employee_id(self, employee_id: str) -> Optional[FaceProfile]:
        """Get face profile by employee ID"""
        return self.db.query(FaceProfile).filter(FaceProfile.employee_id == employee_id).first()

    def get_all_face_profiles(
        self,
        active_only: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[FaceProfile]:
        """
        Get all face profiles

        Args:
            active_only: Only return active profiles
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of FaceProfile instances
        """
        query = self.db.query(FaceProfile)

        if active_only:
            query = query.filter(FaceProfile.is_active == True)

        return query.offset(skip).limit(limit).all()

    def update_face_profile(
        self,
        profile_id: UUID,
        employee_id: Optional[str] = None,
        employee_name: Optional[str] = None,
        profile_metadata: Optional[Dict[str, Any]] = None,
        consent_date: Optional[date] = None,
        is_active: Optional[bool] = None
    ) -> Optional[FaceProfile]:
        """
        Update face profile

        Args:
            profile_id: Profile ID
            employee_id: New employee ID
            employee_name: New employee name
            profile_metadata: New metadata
            consent_date: New consent date
            is_active: New active status

        Returns:
            Updated FaceProfile instance or None if not found
        """
        profile = self.get_face_profile(profile_id)
        if not profile:
            return None

        if employee_id is not None:
            # Check if employee_id is already used by another profile
            existing = self.get_face_profile_by_employee_id(employee_id)
            if existing and existing.id != profile_id:
                raise ValueError(f"Employee ID '{employee_id}' is already in use by another profile")
            profile.employee_id = employee_id
        if employee_name is not None:
            profile.employee_name = employee_name
        if profile_metadata is not None:
            profile.profile_metadata = profile_metadata
        if consent_date is not None:
            profile.consent_date = consent_date
        if is_active is not None:
            profile.is_active = is_active

        profile.updated_at = datetime.utcnow()
        self.db.flush()
        return profile

    def delete_face_profile(self, profile_id: UUID) -> bool:
        """
        Delete face profile and all associated photos

        Args:
            profile_id: Profile ID

        Returns:
            True if deleted, False if not found
        """
        profile = self.get_face_profile(profile_id)
        if not profile:
            return False

        # Delete all photos
        for photo_id in profile.photo_ids:
            self.delete_face_photo(photo_id)

        # Delete profile directory if exists
        profile_dir = os.path.join(self.face_photos_dir, str(profile_id))
        if os.path.exists(profile_dir):
            try:
                shutil.rmtree(profile_dir)
            except Exception as e:
                logger.warning(f"Failed to delete profile directory {profile_dir}: {e}")

        # Delete profile record
        self.db.delete(profile)
        self.db.flush()
        return True

    def update_profile_photos(
        self,
        profile_id: UUID,
        new_photo_ids: List[UUID],
        new_avg_embedding: List[float]
    ) -> Optional[FaceProfile]:
        """
        Update profile photos and average embedding

        Args:
            profile_id: Profile ID
            new_photo_ids: New list of photo IDs
            new_avg_embedding: New average embedding

        Returns:
            Updated FaceProfile instance or None if not found
        """
        profile = self.get_face_profile(profile_id)
        if not profile:
            return None

        profile.photo_ids = new_photo_ids
        profile.avg_face_embedding = new_avg_embedding
        profile.updated_at = datetime.utcnow()
        self.db.flush()
        return profile

    # ============================================
    # Search Operations
    # ============================================

    def search_similar_faces(
        self,
        query_embedding: List[float],
        threshold: float = 0.45,
        limit: int = 10,
        active_only: bool = True
    ) -> List[Tuple[FaceProfile, float]]:
        """
        Search for similar faces using cosine similarity

        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold (0-1)
            limit: Maximum number of results
            active_only: Only search active profiles

        Returns:
            List of tuples (FaceProfile, similarity_score)
        """
        query = self.db.query(FaceProfile)

        if active_only:
            query = query.filter(FaceProfile.is_active == True)

        # Filter profiles with embeddings
        query = query.filter(FaceProfile.avg_face_embedding.isnot(None))

        # Get all profiles
        profiles = query.all()

        # Compute similarities
        results = []
        query_emb = np.array(query_embedding)

        for profile in profiles:
            profile_emb = np.array(profile.avg_face_embedding)
            similarity = float(np.dot(query_emb, profile_emb))

            if similarity >= threshold:
                results.append((profile, similarity))

        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def find_best_match(
        self,
        query_embedding: List[float],
        threshold: float = 0.45,
        active_only: bool = True
    ) -> Optional[Tuple[FaceProfile, float]]:
        """
        Find best matching face profile

        Args:
            query_embedding: Query face embedding
            threshold: Similarity threshold
            active_only: Only search active profiles

        Returns:
            Tuple of (FaceProfile, similarity) or None if no match found
        """
        results = self.search_similar_faces(
            query_embedding,
            threshold=threshold,
            limit=1,
            active_only=active_only
        )

        if results:
            return results[0]
        return None

    # ============================================
    # Utility Operations
    # ============================================

    def count_profiles(self, active_only: bool = True) -> int:
        """
        Count total number of profiles

        Args:
            active_only: Only count active profiles

        Returns:
            Number of profiles
        """
        query = self.db.query(func.count(FaceProfile.id))

        if active_only:
            query = query.filter(FaceProfile.is_active == True)

        return query.scalar()

    def get_profile_storage_path(self, profile_id: UUID) -> str:
        """
        Get storage path for profile photos

        Args:
            profile_id: Profile ID

        Returns:
            Path to profile photo directory
        """
        return os.path.join(self.face_photos_dir, str(profile_id))
