"""
Face Photo Validator
Validates face photos before adding to database
"""
import base64
import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
from PIL import Image
import io
import logging

from src.face_recognition.face_engine import get_face_engine

logger = logging.getLogger(__name__)


class FacePhotoValidator:
    """
    Validates face photos according to requirements:
    - Exactly one face per photo
    - Valid image format
    """

    def __init__(self, min_resolution: int = 0, same_person_threshold: float = None):
        """
        Initialize validator

        Args:
            min_resolution: Minimum face resolution (width and height), 0 = no limit
            same_person_threshold: Threshold for checking if photos are of same person
                                   If None, reads from DEFAULT_FACE_SIMILARITY_THRESHOLD env var
        """
        self.min_resolution = min_resolution

        # Read threshold from environment if not provided
        if same_person_threshold is None:
            same_person_threshold = float(os.getenv('DEFAULT_FACE_SIMILARITY_THRESHOLD', '0.45'))

        self.same_person_threshold = same_person_threshold
        self.engine = get_face_engine()

        if self.engine is None:
            raise RuntimeError("Face recognition is disabled or engine failed to initialize")

        logger.info(f"FacePhotoValidator initialized with same_person_threshold={self.same_person_threshold}")

    def decode_base64_image(self, base64_str: str) -> Tuple[bool, np.ndarray, str]:
        """
        Decode base64 string to image

        Args:
            base64_str: Base64 encoded image

        Returns:
            Tuple of (success, image_array, error_message)
        """
        try:
            # Decode base64
            img_data = base64.b64decode(base64_str)

            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)

            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return False, None, "Failed to decode image"

            return True, image, ""

        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return False, None, f"Invalid image format: {str(e)}"

    def validate_single_photo(self, base64_str: str) -> Dict[str, Any]:
        """
        Validate a single photo

        Args:
            base64_str: Base64 encoded image

        Returns:
            Dictionary with validation results:
            - valid: bool
            - error: str (if not valid)
            - embedding: list (if valid)
            - num_faces: int
        """
        # Decode image
        success, image, error = self.decode_base64_image(base64_str)
        if not success:
            return {
                'valid': False,
                'error': error,
                'num_faces': 0,
                'embedding': None
            }

        # Process with face engine
        result = self.engine.process_image(image, min_resolution=self.min_resolution)

        if not result['success']:
            return {
                'valid': False,
                'error': result['error'],
                'num_faces': 0,
                'embedding': None
            }

        num_faces = result['num_faces']

        # Must have exactly one face
        if num_faces == 0:
            return {
                'valid': False,
                'error': 'No face detected in image',
                'num_faces': 0,
                'embedding': None
            }

        if num_faces > 1:
            return {
                'valid': False,
                'error': f'Multiple faces detected ({num_faces}). Each photo must contain exactly one face',
                'num_faces': num_faces,
                'embedding': None
            }

        # Get embedding
        face_data = result['faces'][0]
        embedding = face_data['embedding']

        return {
            'valid': True,
            'error': None,
            'num_faces': 1,
            'embedding': embedding,
            'confidence': face_data['confidence']
        }

    def validate_multiple_photos(
        self,
        base64_images: List[str],
        check_same_person: bool = True
    ) -> Dict[str, Any]:
        """
        Validate multiple photos

        Args:
            base64_images: List of base64 encoded images
            check_same_person: Whether to check if all photos are of the same person

        Returns:
            Dictionary with validation results:
            - valid: bool
            - error: str (if not valid)
            - embeddings: list of embeddings (if valid)
            - individual_results: list of individual validation results
        """
        if not base64_images:
            return {
                'valid': False,
                'error': 'No images provided',
                'embeddings': [],
                'individual_results': []
            }

        # Validate each photo
        individual_results = []
        embeddings = []

        for i, img_b64 in enumerate(base64_images):
            result = self.validate_single_photo(img_b64)
            individual_results.append({
                'index': i,
                'valid': result['valid'],
                'error': result['error'],
                'num_faces': result['num_faces']
            })

            if result['valid']:
                embeddings.append(result['embedding'])
            else:
                # If any photo is invalid, return error
                return {
                    'valid': False,
                    'error': f"Photo {i + 1}: {result['error']}",
                    'embeddings': [],
                    'individual_results': individual_results
                }

        # Check if all photos are of the same person
        if check_same_person and len(embeddings) > 1:
            embeddings_np = [np.array(emb) for emb in embeddings]
            is_same, min_similarity = self.engine.validate_same_person(
                embeddings_np,
                threshold=self.same_person_threshold
            )

            logger.info(f"Same person validation: threshold={self.same_person_threshold:.2f}, min_similarity={min_similarity:.2f}, is_same={is_same}")

            if not is_same:
                return {
                    'valid': False,
                    'error': f'Photos appear to be of different people (similarity: {min_similarity:.2f}, threshold: {self.same_person_threshold:.2f})',
                    'embeddings': [],
                    'individual_results': individual_results
                }

        return {
            'valid': True,
            'error': None,
            'embeddings': embeddings,
            'individual_results': individual_results
        }

    def validate_photo_against_existing(
        self,
        new_photo_base64: str,
        existing_embeddings: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Validate a new photo against existing photos of the same profile

        Args:
            new_photo_base64: Base64 encoded new photo
            existing_embeddings: List of existing face embeddings

        Returns:
            Dictionary with validation results
        """
        # Validate new photo
        result = self.validate_single_photo(new_photo_base64)

        if not result['valid']:
            return result

        # If there are no existing embeddings, accept the new photo
        if not existing_embeddings:
            return result

        # Check if new photo matches existing photos
        new_embedding = np.array(result['embedding'])
        all_embeddings = existing_embeddings + [new_embedding]

        is_same, min_similarity = self.engine.validate_same_person(
            all_embeddings,
            threshold=self.same_person_threshold
        )

        if not is_same:
            result['valid'] = False
            result['error'] = f'New photo does not match existing photos (similarity: {min_similarity:.2f})'
            result['embedding'] = None

        return result

    def save_image(
        self,
        base64_str: str,
        save_dir: str,
        filename: str
    ) -> Tuple[bool, str, str]:
        """
        Save base64 image to file

        Args:
            base64_str: Base64 encoded image
            save_dir: Directory to save image
            filename: Filename (without extension)

        Returns:
            Tuple of (success, file_path, error_message)
        """
        try:
            # Decode image
            success, image, error = self.decode_base64_image(base64_str)
            if not success:
                return False, "", error

            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save image as PNG
            file_path = os.path.join(save_dir, f"{filename}.png")
            cv2.imwrite(file_path, image)

            return True, file_path, ""

        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False, "", f"Failed to save image: {str(e)}"


# Global validator instance
_validator: FacePhotoValidator = None


def get_validator(force_reload: bool = False) -> FacePhotoValidator:
    """
    Get global validator instance

    Args:
        force_reload: Force reload the validator

    Returns:
        FacePhotoValidator instance
    """
    global _validator

    if _validator is None or force_reload:
        _validator = FacePhotoValidator()

    return _validator
