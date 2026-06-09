"""Reflective wire based auto-ROI detector (MVP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ReflectiveWireDetectionResult:
    """Single-frame reflective wire detection output."""

    roi: Optional[List[List[float]]]
    confidence: float
    debug: Dict[str, Any]


class ReflectiveWireDetector:
    """Detect reflective wire regions and convert to normalized ROI polygon."""

    def __init__(
        self,
        *,
        min_mask_ratio: float = 0.0015,
        min_area_ratio: float = 0.001,
        kernel_size: int = 5,
    ) -> None:
        self.min_mask_ratio = float(max(0.0, min_mask_ratio))
        self.min_area_ratio = float(max(0.0, min_area_ratio))
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size)
        )

    def detect(self, frame: np.ndarray) -> ReflectiveWireDetectionResult:
        """Detect reflective wire and return an ROI candidate."""
        if frame is None or frame.size == 0:
            return ReflectiveWireDetectionResult(
                roi=None,
                confidence=0.0,
                debug={"status": "empty_frame"},
            )

        frame_h, frame_w = frame.shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return ReflectiveWireDetectionResult(
                roi=None,
                confidence=0.0,
                debug={"status": "invalid_frame_size"},
            )

        frame_area = float(frame_h * frame_w)
        mask = self._build_reflective_mask(frame)
        mask_pixels = int(np.count_nonzero(mask))
        mask_ratio = mask_pixels / frame_area

        if mask_ratio < self.min_mask_ratio:
            return ReflectiveWireDetectionResult(
                roi=None,
                confidence=0.0,
                debug={
                    "status": "mask_too_small",
                    "mask_ratio": round(mask_ratio, 6),
                    "min_mask_ratio": self.min_mask_ratio,
                },
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_ratio * frame_area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        if not valid_contours:
            return ReflectiveWireDetectionResult(
                roi=None,
                confidence=0.0,
                debug={
                    "status": "no_valid_contours",
                    "mask_ratio": round(mask_ratio, 6),
                    "min_area_ratio": self.min_area_ratio,
                },
            )

        contour_points = np.vstack([cnt.reshape(-1, 2) for cnt in valid_contours]).astype(
            np.float32
        )
        if contour_points.shape[0] < 4:
            return ReflectiveWireDetectionResult(
                roi=None,
                confidence=0.0,
                debug={"status": "insufficient_points"},
            )

        rect = cv2.minAreaRect(contour_points)
        rect_points = cv2.boxPoints(rect).astype(np.float32)
        ordered_points = self._order_points_clockwise(rect_points)
        normalized_roi = self._normalize_points(ordered_points, frame_w, frame_h)

        hull = cv2.convexHull(contour_points.reshape(-1, 1, 2))
        hull_area = float(cv2.contourArea(hull))
        area_ratio = hull_area / frame_area if frame_area > 0 else 0.0

        mask_score = min(
            1.0,
            mask_ratio / max(self.min_mask_ratio * 2.0, 1e-6),
        )
        area_score = min(
            1.0,
            area_ratio / max(self.min_area_ratio * 2.0, 1e-6),
        )
        confidence = float(max(0.0, min(1.0, 0.55 * mask_score + 0.45 * area_score)))

        return ReflectiveWireDetectionResult(
            roi=normalized_roi,
            confidence=confidence,
            debug={
                "status": "ok",
                "mask_ratio": round(mask_ratio, 6),
                "area_ratio": round(area_ratio, 6),
                "valid_contours": len(valid_contours),
            },
        )

    def _build_reflective_mask(self, frame: np.ndarray) -> np.ndarray:
        """Build a mask for reflective yellow/white wire colors."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Yellow reflective tape / paint
        yellow_mask = cv2.inRange(
            hsv,
            np.array([15, 60, 120], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8),
        )
        # Orange-ish reflective variants
        orange_mask = cv2.inRange(
            hsv,
            np.array([5, 90, 120], dtype=np.uint8),
            np.array([20, 255, 255], dtype=np.uint8),
        )
        # Bright white reflective parts
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 170], dtype=np.uint8),
            np.array([180, 70, 255], dtype=np.uint8),
        )

        mask = cv2.bitwise_or(yellow_mask, orange_mask)
        mask = cv2.bitwise_or(mask, white_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    @staticmethod
    def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
        """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
        if points.shape != (4, 2):
            raise ValueError("Expected shape (4, 2)")

        sums = points.sum(axis=1)
        diffs = np.diff(points, axis=1).reshape(-1)

        top_left = points[np.argmin(sums)]
        bottom_right = points[np.argmax(sums)]
        top_right = points[np.argmin(diffs)]
        bottom_left = points[np.argmax(diffs)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    @staticmethod
    def _normalize_points(
        points: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> List[List[float]]:
        """Normalize pixel points into [0, 1] ROI polygon."""
        normalized: List[List[float]] = []
        for x, y in points:
            nx = max(0.0, min(1.0, float(x) / float(frame_w)))
            ny = max(0.0, min(1.0, float(y) / float(frame_h)))
            normalized.append([round(nx, 4), round(ny, 4)])
        return normalized
