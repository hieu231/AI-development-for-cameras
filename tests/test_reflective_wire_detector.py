import cv2
import numpy as np

from src.ai_models.auto_roi.reflective_wire_detector import ReflectiveWireDetector


def test_reflective_wire_detector_returns_roi_for_yellow_boundary():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    boundary = np.array([[100, 80], [540, 80], [540, 380], [100, 380]], dtype=np.int32)
    cv2.polylines(frame, [boundary], True, (0, 255, 255), thickness=16)

    detector = ReflectiveWireDetector(min_mask_ratio=0.001, min_area_ratio=0.001)
    result = detector.detect(frame)

    assert result.roi is not None
    assert len(result.roi) == 4
    assert result.confidence > 0.0
    for x, y in result.roi:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_reflective_wire_detector_returns_none_for_empty_scene():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = ReflectiveWireDetector(min_mask_ratio=0.001, min_area_ratio=0.001)
    result = detector.detect(frame)

    assert result.roi is None
    assert result.confidence == 0.0
