# src/utils/roi_utils.py
"""
ROI Utilities - Normalize and denormalize ROI coordinates
ROI format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
All coordinates stored as normalized values [0.0, 1.0]
"""

from typing import Any, List, Tuple, Optional
import numpy as np


DEFAULT_NORMALIZED_ROI: List[List[float]] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
]

MIN_ROI_POINTS = 3
MAX_ROI_POINTS = 30


def is_roi_polygon_set(roi: Any) -> bool:
    """Return True when ROI is shaped as a list of polygons."""
    if not isinstance(roi, list) or not roi:
        return False

    first = roi[0]
    if not isinstance(first, (list, tuple)) or not first:
        return False

    first_item = first[0]
    return isinstance(first_item, (list, tuple))


def get_roi_polygons(roi: Any) -> List[List[List[float]]]:
    """Normalize ROI input into a list of polygons."""
    if not roi:
        return []

    def _normalize_polygon(polygon: Any) -> List[List[float]]:
        if not isinstance(polygon, list):
            return []

        normalized_points: List[List[float]] = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            x, y = point
            normalized_points.append([float(x), float(y)])
        return normalized_points

    if is_roi_polygon_set(roi):
        normalized_polygons: List[List[List[float]]] = []
        for polygon in roi:
            normalized_polygon = _normalize_polygon(polygon)
            if normalized_polygon:
                normalized_polygons.append(normalized_polygon)
        return normalized_polygons

    if isinstance(roi, list):
        normalized_polygon = _normalize_polygon(roi)
        return [normalized_polygon] if normalized_polygon else []

    return []


def get_effective_roi(roi: Any) -> Optional[List[List[float]]]:
    """Return the first polygon for backward-compatible single-ROI consumers."""
    polygons = get_roi_polygons(roi)
    if not polygons:
        return None
    return polygons[0]


def build_roi_poly_arrays(
    roi: Any,
    frame_width: int,
    frame_height: int,
) -> List[np.ndarray]:
    """Build a list of pixel-coordinate numpy polygon arrays from any ROI input.

    Accepts both single-polygon ``[[x,y], ...]`` and polygon-set
    ``[[[x,y], ...], [[x,y], ...]]`` formats.  Falls back to the full frame
    when *roi* is ``None`` or empty.

    Each returned array has ``dtype=np.int32`` and shape ``(N, 2)`` — ready for
    ``cv2.pointPolygonTest`` and ``cv2.polylines``.
    """
    polygons = get_roi_polygons(roi) or [DEFAULT_NORMALIZED_ROI]
    result: List[np.ndarray] = []
    for poly in polygons:
        px = denormalize_roi(poly, frame_width, frame_height)
        if px:
            result.append(np.array(px, dtype=np.int32))
    if not result:
        # Absolute fallback – full frame
        px = denormalize_roi(DEFAULT_NORMALIZED_ROI, frame_width, frame_height)
        result.append(np.array(px, dtype=np.int32))
    return result


def is_point_in_any_roi(
    point: Tuple[float, float],
    roi_polys: List[np.ndarray],
) -> bool:
    """Return True when *point* (x, y) is inside at least one ROI polygon."""
    import cv2

    for rp in roi_polys:
        if cv2.pointPolygonTest(rp, point, False) >= 0:
            return True
    return False


def draw_roi_overlays(
    frame: np.ndarray,
    roi_polys: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw all ROI polygons onto *frame* (mutates in place)."""
    import cv2

    for idx, rp in enumerate(roi_polys):
        cv2.polylines(frame, [rp.reshape((-1, 1, 2))], True, color, thickness)
        label = f"ROI{idx}" if len(roi_polys) > 1 else "ROI"
        tx, ty = int(rp[0][0]) + 10, int(rp[0][1]) + 25
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def validate_normalized_roi_input(
    roi: Any,
    *,
    max_polygons: int = 5,
) -> Tuple[bool, Optional[str]]:
    """Validate either a single ROI polygon or a polygon set."""
    polygons = get_roi_polygons(roi)
    if not polygons:
        return False, "ROI is empty"

    if is_roi_polygon_set(roi):
        if len(polygons) > max_polygons:
            return (
                False,
                f"ROI polygon set can have max {max_polygons} polygons, got {len(polygons)}",
            )

        for index, polygon in enumerate(polygons):
            is_valid, error = validate_normalized_roi(polygon)
            if not is_valid:
                return False, f"ROI polygon {index}: {error}"
        return True, None

    return validate_normalized_roi(polygons[0])


def normalize_roi(
    roi: List[List[float]], frame_width: int, frame_height: int
) -> List[List[float]]:
    """
    Normalize ROI coordinates to [0, 1] range

    Args:
        roi: List of [x, y] points in pixel coordinates
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels

    Returns:
        Normalized ROI with values in [0, 1]

    Example:
        >>> roi = [[100, 200], [500, 200], [500, 600], [100, 600]]
        >>> normalize_roi(roi, 1920, 1080)
        [[0.052, 0.185], [0.260, 0.185], [0.260, 0.556], [0.052, 0.556]]
    """
    if not roi or frame_width <= 0 or frame_height <= 0:
        return []

    normalized = []
    for point in roi:
        if len(point) != 2:
            continue
        x, y = point
        norm_x = max(0.0, min(1.0, x / frame_width))
        norm_y = max(0.0, min(1.0, y / frame_height))
        normalized.append([round(norm_x, 4), round(norm_y, 4)])

    return normalized


def denormalize_roi(
    normalized_roi: List[List[float]], frame_width: int, frame_height: int
) -> List[List[int]]:
    """
    Denormalize ROI coordinates from [0, 1] to pixel coordinates

    Args:
        normalized_roi: List of [x, y] points in normalized [0, 1] range
        frame_width: Target frame width in pixels
        frame_height: Target frame height in pixels

    Returns:
        ROI in pixel coordinates

    Example:
        >>> roi = [[0.052, 0.185], [0.260, 0.185], [0.260, 0.556], [0.052, 0.556]]
        >>> denormalize_roi(roi, 1920, 1080)
        [[100, 200], [499, 200], [499, 600], [100, 600]]
    """
    if not normalized_roi or frame_width <= 0 or frame_height <= 0:
        return []

    denormalized = []
    for point in normalized_roi:
        if len(point) != 2:
            continue
        norm_x, norm_y = point
        x = int(norm_x * frame_width)
        y = int(norm_y * frame_height)
        denormalized.append([x, y])

    return denormalized


def validate_normalized_roi(roi: List[List[float]]) -> Tuple[bool, Optional[str]]:
    """
    Validate that ROI is properly normalized

    Args:
        roi: ROI to validate

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> validate_normalized_roi([[0.1, 0.2], [0.5, 0.2], [0.5, 0.8]])
        (True, None)
        >>> validate_normalized_roi([[0.1, 1.5], [0.5, 0.2]])
        (False, "Point [0.1, 1.5] has values outside [0, 1] range")
    """
    if not roi:
        return False, "ROI is empty"

    if not isinstance(roi, list):
        return False, "ROI must be a list"

    if len(roi) < MIN_ROI_POINTS:
        return False, f"ROI must have at least {MIN_ROI_POINTS} points"

    if len(roi) > MAX_ROI_POINTS:
        return False, f"ROI must have at most {MAX_ROI_POINTS} points"

    for i, point in enumerate(roi):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False, f"Point {i} must be a list/tuple of 2 values [x, y]"

        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return False, f"Point {i} coordinates must be numbers"

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False, f"Point {point} has values outside [0, 1] range"

    return True, None


def roi_to_rect(roi: List[List[float]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Convert ROI polygon to bounding rectangle [x1, y1, x2, y2]

    Args:
        roi: ROI polygon points

    Returns:
        Tuple (x1, y1, x2, y2) or None if invalid
    """
    if not roi or len(roi) < 2:
        return None

    xs = [p[0] for p in roi if len(p) >= 2]
    ys = [p[1] for p in roi if len(p) >= 2]

    if not xs or not ys:
        return None

    return (min(xs), min(ys), max(xs), max(ys))


def rect_to_roi(rect: Tuple[float, float, float, float]) -> List[List[float]]:
    """
    Convert rectangle [x1, y1, x2, y2] to ROI polygon (4 corners)

    Args:
        rect: Rectangle coordinates [x1, y1, x2, y2]

    Returns:
        ROI as 4-point polygon
    """
    x1, y1, x2, y2 = rect
    return [
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ]


def get_default_roi_params(
    roi_width_ratio: float = 0.6,
    roi_height_ratio: float = 0.45,
    roi_x_ratio: float = 0.2,
    roi_y_ratio: float = 0.275,
) -> List[List[float]]:
    """
    Generate default normalized ROI from ratio parameters

    Args:
        roi_width_ratio: Width of ROI as ratio of frame
        roi_height_ratio: Height of ROI as ratio of frame
        roi_x_ratio: X position of ROI as ratio of frame
        roi_y_ratio: Y position of ROI as ratio of frame

    Returns:
        Normalized ROI polygon
    """
    x1 = roi_x_ratio
    y1 = roi_y_ratio
    x2 = roi_x_ratio + roi_width_ratio
    y2 = roi_y_ratio + roi_height_ratio

    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return rect_to_roi((x1, y1, x2, y2))
