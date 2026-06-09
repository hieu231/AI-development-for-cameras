# src/utils/image_utils.py
"""
Image Processing Utilities
Functions for image preprocessing, resizing, etc.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Any
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


def make_cv2_safe_text(
    text: Any,
    fallback: str = "object",
    max_length: int = 96,
) -> str:
    """
    Convert text to an ASCII-only form that cv2.putText can render reliably.

    OpenCV's Hershey fonts do not support full Unicode; passing accented text
    often yields '?' glyphs on output frames.
    """
    raw = str(text or "").strip()
    if not raw:
        raw = fallback

    normalized = unicodedata.normalize("NFKD", raw)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    ascii_text = re.sub(r"[^\x20-\x7E]", "", ascii_text)

    if not ascii_text:
        ascii_text = fallback

    if max_length > 0 and len(ascii_text) > max_length:
        ascii_text = ascii_text[:max_length]

    return ascii_text


def parse_resolution(resolution_str: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Parse resolution string to (width, height) tuple

    Args:
        resolution_str: Resolution string in format "WIDTHxHEIGHT" (e.g., "640x480")

    Returns:
        Tuple of (width, height) or None if invalid

    Example:
        >>> parse_resolution("1920x1080")
        (1920, 1080)
        >>> parse_resolution("640x480")
        (640, 480)
        >>> parse_resolution(None)
        None
    """
    if not resolution_str:
        return None

    try:
        parts = resolution_str.lower().strip().split('x')
        if len(parts) != 2:
            logger.warning(f"Invalid resolution format: {resolution_str}")
            return None

        width = int(parts[0])
        height = int(parts[1])

        if width <= 0 or height <= 0:
            logger.warning(f"Invalid resolution dimensions: {resolution_str}")
            return None

        return (width, height)

    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to parse resolution '{resolution_str}': {e}")
        return None


def resize_frame(
    frame: np.ndarray,
    target_resolution: Optional[Tuple[int, int]] = None,
    maintain_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Resize frame to target resolution

    Args:
        frame: Input frame (numpy array, BGR format)
        target_resolution: Target (width, height) or None to skip resize
        maintain_aspect_ratio: If True, maintain aspect ratio and pad if needed

    Returns:
        Resized frame

    Example:
        >>> frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> resized = resize_frame(frame, (640, 480))
        >>> resized.shape
        (480, 640, 3)
    """
    if target_resolution is None or frame is None or frame.size == 0:
        return frame

    target_width, target_height = target_resolution
    current_height, current_width = frame.shape[:2]

    # Already at target resolution
    if current_width == target_width and current_height == target_height:
        return frame

    if maintain_aspect_ratio:
        # Calculate scaling factor to fit within target resolution
        scale = min(target_width / current_width, target_height / current_height)
        new_width = int(current_width * scale)
        new_height = int(current_height * scale)

        # Resize
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Pad to exact target resolution if needed
        if new_width != target_width or new_height != target_height:
            # Create black canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

            # Center the resized frame on canvas
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

            return canvas

        return resized

    else:
        # Simple resize without maintaining aspect ratio
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def preprocess_frame(
    frame: np.ndarray,
    preprocess_resolution: Optional[str] = None
) -> np.ndarray:
    """
    Preprocess frame before AI inference

    Args:
        frame: Input frame (BGR)
        preprocess_resolution: Target resolution string (e.g., "640x480")

    Returns:
        Preprocessed frame

    Example:
        >>> frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        >>> processed = preprocess_frame(frame, "640x480")
        >>> processed.shape
        (480, 640, 3)
    """
    if frame is None or frame.size == 0:
        return frame

    # Parse and apply resolution
    target_res = parse_resolution(preprocess_resolution)
    if target_res:
        frame = resize_frame(frame, target_res, maintain_aspect_ratio=False)

    return frame
