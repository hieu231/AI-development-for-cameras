"""
src/ai_models/oil_cap_detection_model.py
"""

from typing import Any, Dict, List, Optional, Tuple
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, resolve_engine_path
from src.core.object_tracker import ObjectTracker
from src.utils.alert_levels import AlertLevel
from src.utils.roi_utils import denormalize_roi


class OilCapDetectionModel(BaseModel):
    """Detect oil cap states and trigger events when the cap is open."""

    LABEL_ALIASES = {
        "oil-cap-opened": "oil_cap_opened",
        "oil_cap_opened": "oil_cap_opened",
        "opened": "oil_cap_opened",
        "open": "oil_cap_opened",
        "oil-cap-closed": "oil_cap_closed",
        "oil_cap_closed": "oil_cap_closed",
        "closed": "oil_cap_closed",
        "close": "oil_cap_closed",
    }

    DISPLAY_LABELS = {
        "oil_cap_opened": "Nap dau mo",
        "oil_cap_closed": "Nap dau dong",
    }

    DEFAULT_VIOLATION_LABELS = {"oil_cap_opened"}

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/oil_cap_detection.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        detection_cooldown: int = 300,
        **kwargs,
    ):
        super().__init__(
            model_name="OilCapDetectionModel",
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.person_model_path = resolve_engine_path(
            kwargs.get("person_model_path", "src/ai_models/model_weights/yolo11n.pt"),
            runtime_device=self.device,
        )
        self.person_model = YOLO(self.person_model_path)
        if self.person_model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.person_model.to(self.device)

        self.conf_threshold = float(kwargs.get("conf_threshold", confidence_threshold))
        self.iou_threshold = float(kwargs.get("iou_threshold", iou_threshold))
        self.detection_cooldown = int(kwargs.get("detection_cooldown", detection_cooldown))
        self.track_match_iou_threshold = float(kwargs.get("track_match_iou_threshold", 0.5))
        self.track_stale_timeout = float(kwargs.get("track_stale_timeout", 3.0))
        self.violation_min_center_y_ratio = float(
            kwargs.get("violation_min_center_y_ratio", 0.35)
        )
        self.open_cap_max_top_y_ratio = float(
            kwargs.get("open_cap_max_top_y_ratio", 0.6)
        )
        self.no_person_repeat_interval = float(
            kwargs.get("no_person_repeat_interval", 5.0)
        )
        self.skip_when_person_present = bool(
            kwargs.get("skip_when_person_present", False)
        )
        self.person_conf_threshold = float(
            kwargs.get("person_conf_threshold", 0.35)
        )
        self.person_iou_threshold = float(
            kwargs.get("person_iou_threshold", 0.35)
        )

        raw_violation_labels = kwargs.get("violation_labels", list(self.DEFAULT_VIOLATION_LABELS))
        self.violation_labels = {
            normalized
            for normalized in (
                self._normalize_label(str(label))
                for label in raw_violation_labels
            )
            if normalized
        } or set(self.DEFAULT_VIOLATION_LABELS)

        self.class_names: Dict[int, str] = {}
        self.classes_to_keep: List[int] = []
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict):
            for class_id, name in model_names.items():
                normalized = self._normalize_label(str(name))
                if normalized:
                    self.class_names[int(class_id)] = normalized
                    self.classes_to_keep.append(int(class_id))

        self.person_class_names: Dict[int, str] = {}
        self.person_classes_to_keep: List[int] = []
        person_model_names = getattr(self.person_model, "names", {})
        if isinstance(person_model_names, dict):
            for class_id, name in person_model_names.items():
                normalized = self._normalize_label(str(name))
                if normalized == "person":
                    self.person_class_names[int(class_id)] = normalized
                    self.person_classes_to_keep.append(int(class_id))
        if not self.person_classes_to_keep:
            self.person_class_names[0] = "person"
            self.person_classes_to_keep = [0]

        self.object_tracker = ObjectTracker(reset_interval=self.detection_cooldown)
        self._stable_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_synthetic_track_id = 900000
        self._state_lock = threading.RLock()

        self.logger.info(
            "OilCapDetectionModel loaded | Device: %s | Conf: %.2f | IoU: %.2f | Violation labels: %s | Person aux: %s | Open-cap gate: top<%.0f%% | No-person interval: %.1fs",
            self.device,
            self.conf_threshold,
            self.iou_threshold,
            sorted(self.violation_labels),
            self.person_model_path,
            self.open_cap_max_top_y_ratio * 100.0,
            self.no_person_repeat_interval,
        )

    @classmethod
    def _normalize_label(cls, raw_label: str) -> str:
        normalized = raw_label.strip().lower().replace(" ", "_")
        return cls.LABEL_ALIASES.get(normalized, normalized)

    def _display_label(self, normalized_label: str) -> str:
        return self.DISPLAY_LABELS.get(
            normalized_label,
            normalized_label.replace("_", " ").title(),
        )

    def _resolve_track_id(
        self,
        tracker_id: Optional[int],
        bbox: tuple[int, int, int, int],
    ) -> int:
        now = time.time()
        self._cleanup_stale_tracks(now)

        if tracker_id is not None and tracker_id in self._stable_tracks:
            self._stable_tracks[tracker_id]["bbox"] = bbox
            self._stable_tracks[tracker_id]["last_seen"] = now
            return tracker_id

        best_id: Optional[int] = None
        best_iou = self.track_match_iou_threshold
        for stable_id, info in self._stable_tracks.items():
            iou = self._compute_iou(bbox, info["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id = stable_id

        if best_id is not None:
            self._stable_tracks[best_id]["bbox"] = bbox
            self._stable_tracks[best_id]["last_seen"] = now
            return best_id

        stable_id = int(tracker_id) if tracker_id is not None else self._next_synthetic_track_id
        if tracker_id is None:
            self._next_synthetic_track_id += 1

        self._stable_tracks[stable_id] = {"bbox": bbox, "last_seen": now}
        return stable_id

    def _cleanup_stale_tracks(self, now: float) -> None:
        stale_ids = [
            stable_id
            for stable_id, info in self._stable_tracks.items()
            if now - float(info["last_seen"]) > self.track_stale_timeout
        ]
        for stable_id in stale_ids:
            del self._stable_tracks[stable_id]

    @staticmethod
    def _compute_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def _detect_persons_in_roi(
        self,
        frame: np.ndarray,
        roi_poly_np: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        roi_x2: int,
        roi_y2: int,
    ) -> List[Dict[str, Any]]:
        person_results = self._run_yolo_track(
            self.person_model,
            frame,
            conf=self.person_conf_threshold,
            iou=self.person_iou_threshold,
            classes=self.person_classes_to_keep if self.person_classes_to_keep else None,
            persist=False,
            verbose=False,
        )

        person_detections: List[Dict[str, Any]] = []
        for result in person_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                if center_x < roi_x1 or center_x > roi_x2 or center_y < roi_y1 or center_y > roi_y2:
                    continue
                if cv2.pointPolygonTest(roi_poly_np, (float(center_x), float(center_y)), False) < 0:
                    continue

                raw_label = result.names.get(cls_id, f"class_{cls_id}")
                class_name = self.person_class_names.get(cls_id, self._normalize_label(str(raw_label)))
                if class_name != "person":
                    continue

                person_detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return person_detections

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)

        normalized_roi = kwargs.get("roi", [[0, 0], [1, 0], [1, 1], [0, 1]])
        roi_polygon = denormalize_roi(normalized_roi, w, h)
        roi_poly_np = np.array(roi_polygon, dtype=np.int32)
        roi_x1, roi_y1 = roi_poly_np[:, 0].min(), roi_poly_np[:, 1].min()
        roi_x2, roi_y2 = roi_poly_np[:, 0].max(), roi_poly_np[:, 1].max()
        roi_height = max(1, roi_y2 - roi_y1)
        frame_height = max(1, h)

        person_detections: List[Dict[str, Any]] = []
        person_present_in_roi = False
        if self.skip_when_person_present:
            person_detections = self._detect_persons_in_roi(
                frame,
                roi_poly_np,
                roi_x1,
                roi_y1,
                roi_x2,
                roi_y2,
            )
            person_present_in_roi = len(person_detections) > 0

        classes = self.classes_to_keep if self.classes_to_keep else None
        results = self._run_yolo_track(
            self.model,
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        all_detections: List[Dict[str, Any]] = []
        events_to_create: List[Dict[str, Any]] = []
        compliant_detections: List[Dict[str, Any]] = []

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
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                if center_x < roi_x1 or center_x > roi_x2 or center_y < roi_y1 or center_y > roi_y2:
                    continue
                if cv2.pointPolygonTest(roi_poly_np, (float(center_x), float(center_y)), False) < 0:
                    continue

                raw_label = result.names.get(cls_id, f"class_{cls_id}")
                class_name = self.class_names.get(cls_id, self._normalize_label(str(raw_label)))
                display_name = self._display_label(class_name)
                raw_track_id = track_ids[i] if i < len(track_ids) else None
                effective_track_id = self._resolve_track_id(raw_track_id, (x1, y1, x2, y2))
                top_y_ratio = y1 / frame_height

                detection_info = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "display_name": display_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": effective_track_id,
                    "source_track_id": raw_track_id,
                }
                all_detections.append(detection_info)

                is_violation = class_name in self.violation_labels
                center_y_ratio_in_roi = (center_y - roi_y1) / roi_height
                # Apply top-y suppression only when person-skip mode is enabled.
                # When skip_when_person_present is off, violations should still emit events.
                is_too_high_for_alert = (
                    is_violation
                    and self.skip_when_person_present
                    and top_y_ratio <= self.open_cap_max_top_y_ratio
                )
                should_skip_due_to_person = is_violation and self.skip_when_person_present and person_present_in_roi

                if is_too_high_for_alert:
                    detection_info["suppressed_reason"] = "opened_cap_too_high"
                    detection_info["top_y_ratio"] = round(top_y_ratio, 4)
                if should_skip_due_to_person:
                    detection_info["suppressed_reason"] = "person_present"
                    detection_info["person_present_in_roi"] = True
                    detection_info["center_y_ratio_in_roi"] = round(center_y_ratio_in_roi, 4)

                if is_violation:
                    should_emit_event = (
                        (not is_too_high_for_alert)
                        and (not should_skip_due_to_person)
                    )
                    if should_emit_event and self.object_tracker.should_record_event(
                        effective_track_id,
                        f"OIL_CAP_{class_name.upper()}",
                        repeat_interval=self.no_person_repeat_interval,
                    ):
                        events_to_create.append(
                            {
                                "track_id": effective_track_id,
                                "violation_type": class_name,
                                "display_name": display_name,
                                "confidence": conf,
                                "bbox": [x1, y1, x2, y2],
                                "top_y_ratio": round(top_y_ratio, 4),
                                "center_y_ratio_in_roi": round(center_y_ratio_in_roi, 4),
                                "person_present_in_roi": person_present_in_roi,
                            }
                        )
                        self.logger.warning(
                            "OIL CAP VIOLATION | %s | ID: %s | Conf: %.2f | person_present=%s | top_ratio=%.2f",
                            class_name,
                            effective_track_id,
                            conf,
                            person_present_in_roi,
                            top_y_ratio,
                        )
                else:
                    compliant_detections.append(
                        {
                            "track_id": effective_track_id,
                            "class_name": class_name,
                            "display_name": display_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

                if annotate and annotated_frame is not None:
                    if is_too_high_for_alert:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255) if is_violation else (0, 200, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{display_name} {conf:.2f} ID:{effective_track_id}"
                    if is_too_high_for_alert:
                        label += " IGNORE-HIGH"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - th - 10),
                        (x1 + tw + 4, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

        if annotate and annotated_frame is not None:
            cv2.polylines(
                annotated_frame,
                [roi_poly_np.reshape((-1, 1, 2))],
                True,
                (0, 255, 255),
                2,
            )
            tx, ty = roi_poly_np[0]
            cv2.putText(
                annotated_frame,
                "OIL CAP ROI",
                (int(tx) + 5, int(ty) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        event_triggered = len(events_to_create) > 0
        primary_event = events_to_create[0] if event_triggered else None
        metadata: Dict[str, Any] = {
            "type": "Giam sat nap dau",
            "eventType": "Giam sat nap dau",
            "severity": "high" if event_triggered else "low",
            "description": "Phat hien trang thai dong/mo nap dau",
            "detections": all_detections,
            "violations": events_to_create,
            "compliant": compliant_detections,
            "person_present_in_roi": person_present_in_roi,
            "person_detections": person_detections,
            "open_cap_max_top_y_ratio": self.open_cap_max_top_y_ratio,
            "no_person_repeat_interval": self.no_person_repeat_interval,
            "violation_min_center_y_ratio": self.violation_min_center_y_ratio,
            "count": len(all_detections),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "oil_cap_detection",
        }

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )
