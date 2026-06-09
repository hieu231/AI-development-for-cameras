"""
src/ai_models/yolov8m_model.py
Generic YOLOv8 object detection model - filter / annotation only.
Used for model types that only need detection overlays without emitting
events or alerts (e.g. object_detection, yolo, yolov8, yolov11, alpr,
oil_dumping).
"""

from typing import Dict, Any, List, Optional
import threading
import time
import numpy as np
import cv2
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, resolve_engine_path
from src.utils.roi_utils import (
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)
from src.utils.alert_levels import AlertLevel


class YOLOv8Model(BaseModel):
    """Generic YOLO detection model for filter/annotation logic only.
    Never emits events or alerts."""

    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        **kwargs,
    ):
        super().__init__(
            model_name="YOLOv8Model",
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.iou_threshold = kwargs.get("iou_threshold", iou_threshold)
        self.conf_threshold = kwargs.get("conf_threshold", confidence_threshold)

        # Thread safety for the non-reentrant YOLO tracker.
        self._state_lock = threading.Lock()

        self.logger.info(
            f"YOLOv8Model loaded (filter-only) | Device: {self.device} | "
            f"Conf: {self.conf_threshold} | IoU: {self.iou_threshold} | "
            f"Model: {model_path}"
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)

        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)

        results = self._run_yolo_track(
            self.model,
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False,
        )

        annotated_frame: Optional[np.ndarray] = frame.copy() if annotate else None
        all_detections: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                class_name = result.names.get(cls_id, f"class_{cls_id}")
                if conf < self.conf_threshold:
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if not is_point_in_any_roi((float(cx), float(cy)), roi_polys):
                    continue

                track_id = track_ids[i] if i < len(track_ids) else None

                all_detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": track_id,
                })

                if annotate and annotated_frame is not None:
                    color = (0, 255, 0)
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        2,
                    )
                    label = f"{class_name} {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1) - th - 10),
                        (int(x1) + tw, int(y1)),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys, color=(0, 255, 0))

        # Filter-only model: never emit events
        metadata: Dict[str, Any] = {
            "type": "object_detection",
            "severity": "low",
            "detections": all_detections,
            "count": len(all_detections),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "YOLOv8",
        }

        return DetectionResult(
            frame=annotated_frame,
            event=False,
            metadata=metadata,
        )
