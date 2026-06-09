"""
src/ai_models/vehicle_gate_model.py
Phát hiện xe qua cổng/khu vực – Mức cảnh báo LOW (thông tin)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import time
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator
from src.utils.roi_utils import denormalize_roi
from src.utils.alert_levels import AlertLevel


class VehicleGateModel(BaseModel):
    """
    Model chuyên phát hiện xe qua cổng, bãi đỗ, khu vực hạn chế
    Mức cảnh báo: LOW (chỉ thông báo, không nguy hiểm)
    """

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
    
    # Mapping class name từ model sang tiếng Việt
    CLASS_NAME_MAPPING = {
        'car': 'Xe ô tô',
        'truck': 'Xe tải',
        'bus': 'Xe buýt',
        'motorcycle': 'Xe máy',
        'bicycle': 'Xe đạp',
        'train': 'Tàu hỏa'
    }

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_vehicles_to_alert: int = 1,
        **kwargs
    ):
        super().__init__(
            model_name="VehicleGateModel",
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith('.pt') and self.device in ['cuda', 'mps']:
            self.model.to(self.device)

        self.iou_threshold = iou_threshold
        self.min_vehicles_to_alert = min_vehicles_to_alert
        self._fallback_track_grid_size = max(
            1,
            int(kwargs.get("fallback_track_grid_size", 32)),
        )

        # Mỗi xe chỉ ghi 1 lần trong 30 phút
        self.object_tracker = ObjectTracker(reset_interval=1800)
        self._event_dedup_window_seconds = float(
            kwargs.get("event_dedup_window_seconds", 2.0)
        )
        self._event_dedup_iou_threshold = float(
            kwargs.get("event_dedup_iou_threshold", 0.5)
        )
        self.recent_violation_deduplicator = RecentViolationDeduplicator(
            window_seconds=self._event_dedup_window_seconds,
            iou_threshold=self._event_dedup_iou_threshold,
        )

        self.logger.info(
            f"VehicleGateModel LOADED | {model_path} | Device: {self.device} | "
            f"Conf: {confidence_threshold} | Min vehicles: {min_vehicles_to_alert}"
        )

    def _get_recent_violation_deduplicator(self) -> RecentViolationDeduplicator:
        deduplicator = getattr(self, "recent_violation_deduplicator", None)
        if deduplicator is None:
            deduplicator = RecentViolationDeduplicator(
                window_seconds=float(
                    getattr(self, "_event_dedup_window_seconds", 2.0)
                ),
                iou_threshold=float(
                    getattr(self, "_event_dedup_iou_threshold", 0.5)
                ),
            )
            self.recent_violation_deduplicator = deduplicator
        return deduplicator

    def _resolve_effective_track_id(
        self,
        track_id: Optional[int],
        bbox: Tuple[int, int, int, int],
        class_id: int,
    ) -> int:
        """
        Ensure each detection has a stable-ish object key for event dedup.

        If tracker IDs are unavailable (e.g. YOLO track() fallback to predict()),
        derive a deterministic key from quantized bbox geometry.
        """
        if track_id is not None:
            return int(track_id)

        x1, y1, x2, y2 = bbox
        step = int(self._fallback_track_grid_size)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        def _q(value: float) -> int:
            return int(round(value / step) * step)

        return hash((int(class_id), _q(cx), _q(cy), _q(w), _q(h))) & 0x7FFFFFFF

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        recent_violation_deduplicator = self._get_recent_violation_deduplicator()
        h, w = frame.shape[:2]
        annotate = kwargs.get('annotate', True)
        current_time = time.time()

        normalized_roi = kwargs.get('roi', [[0, 0], [1, 0], [1, 1], [0, 1]])
        roi_polygon = denormalize_roi(normalized_roi, w, h)
        roi_poly_np = np.array(roi_polygon, dtype=np.int32)

        classes = kwargs.get('classes', None)

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame

        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_vehicles = []

        for result in results:
            boxes = result.boxes
            if not boxes:
                continue

            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = result.names.get(cls_id, f"Class_{cls_id}").lower()
                effective_track_id = self._resolve_effective_track_id(
                    track_id,
                    (x1, y1, x2, y2),
                    cls_id,
                )

                if not any(v in class_name for v in self.VEHICLE_CLASSES):
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if cv2.pointPolygonTest(roi_poly_np, center, False) < 0:
                    continue

                raw_class_name = result.names.get(cls_id, f"Class_{cls_id}")
                # Chuyển sang tiếng Việt
                class_name_vn = self.CLASS_NAME_MAPPING.get(raw_class_name.lower(), raw_class_name)
                
                obj = DetectedObject(
                    label=class_name_vn,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    extra={'track_id': effective_track_id, 'class_id': cls_id, 'raw_class': raw_class_name}
                )
                detections.append(obj)

                is_recent_duplicate = recent_violation_deduplicator.is_recent_duplicate(
                    "VEHICLE",
                    effective_track_id,
                    (x1, y1, x2, y2),
                    current_time=current_time,
                )
                if is_recent_duplicate:
                    self.logger.debug(
                        "Suppressed recent duplicate vehicle gate event: ID=%s",
                        effective_track_id,
                    )

                if (
                    not is_recent_duplicate
                    and self.object_tracker.should_record_event(effective_track_id, "VEHICLE")
                ):
                    recent_violation_deduplicator.remember_event(
                        "VEHICLE",
                        effective_track_id,
                        (x1, y1, x2, y2),
                        current_time=current_time,
                    )
                    new_vehicles.append({
                        'track_id': effective_track_id,
                        'type': class_name_vn,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                    })

                # Vẽ annotation
                if annotate and annotated_frame is not None:
                    color = (255, 255, 0)  # Cyan
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    label = f"{raw_class_name} {conf:.2f} ID:{effective_track_id}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 15), (x1 + tw + 10, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        # Vẽ ROI
        if annotate and annotated_frame is not None:
            cv2.polylines(annotated_frame, [roi_poly_np.reshape((-1, 1, 2))], True, (255, 255, 0), 3)
            cv2.putText(annotated_frame, "VEHICLE GATE ZONE", (roi_polygon[0][0] + 10, roi_polygon[0][1] + 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

        # Có xe mới đạt ngưỡng không?
        event_triggered = len(new_vehicles) >= self.min_vehicles_to_alert

        # Chuẩn hóa metadata 
        # Build detections 
        detection_dicts: List[Dict[str, Any]] = []
        for d in detections:
            track_id = None
            class_id = None
            if d.extra and isinstance(d.extra, dict):
                track_id = d.extra.get("track_id")
                class_id = d.extra.get("class_id")
            
            detection_dicts.append(
                {
                    "class_id": class_id,
                    "class_name": d.label,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "track_id": track_id,
                }
            )

        # Violations
        violations = [
            {
                "track_id": v["track_id"],
                "violation_type": v["type"],
                "confidence": v["confidence"],
                "bbox": v["bbox"],
            }
            for v in new_vehicles
        ]

        primary_event = violations[0] if violations else None
        event_type = "Phương tiện ra/vào khu vực"

        metadata: Dict[str, Any] = {
            "type": "Xe qua cổng",
            "severity": "high" if event_triggered else "low",
            "detections": detection_dicts,
            "violations": violations,  # All violations to be recorded
            "count": len(detection_dicts),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "vehicle_gate",
        }

        metadata["type"] = event_type
        metadata["eventType"] = event_type
        metadata["title"] = event_type
        metadata["description"] = "Phát hiện phương tiện ra/vào khu vực giám sát"

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["violation_type"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]
            metadata["bbox"] = primary_event["bbox"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )
