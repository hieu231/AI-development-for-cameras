"""
Helmet & PPE models

- HelmetModel: legacy event metadata format, kept for backward compatibility with
  older systems that expect detection_data.type = 'HELMET' and a `violations`
  list containing `violation_type`, etc.
- PPEModel: new standardized PPE helmet violation model using BaseModel helpers
  and richer metadata, used by newer pipelines.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
import time
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker
from src.utils.roi_utils import (
    DEFAULT_NORMALIZED_ROI,
    denormalize_roi,
    get_effective_roi,
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)
from src.utils.alert_levels import AlertLevel


class HelmetModel(BaseModel):
    """
    Legacy Helmet & Vest Detection Model

    Detects NO-Vest violations and returns metadata in the original format:
      {
        "type": "An toàn lao động",
        "detections": [...],
        "violations": [
          {
            "track_id": ...,
            "violation_type": "...",
            "confidence": ...,
            "bbox": [...]
          },
          ...
        ],
        "count": <int>,
        "timestamp": "...",
        "model_type": "helmet",
        "violation": "...",        # first violation_type
        "confidence": <float>,     # first violation confidence
        "track_id": <int>,         # first violation track_id
      }
    """

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/atld_92.pt",
        **kwargs: Any,
    ):
        super().__init__(
            model_name="HelmetModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=kwargs.get("conf_threshold", 0.45),
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        # Only move to device for PyTorch models, not TensorRT/ONNX
        if self.device and model_path.endswith(".pt"):
            self.model.to(self.device)

        self.conf_threshold: float = kwargs.get("conf_threshold", 0.45)
        self.person_conf_threshold: float = kwargs.get("person_conf_threshold", 0.5)

        # Class mappings - Only NO-Vest detection
        self.classes_to_keep = [2]  # NO-Vest only
        self.class_names = {
            2: "no_vest",
        }

        # Object tracking with 30-minute reset
        self.object_tracker = ObjectTracker(reset_interval=1800)  # 30 minutes

        self.logger.info(
            f"HelmetModel initialized on {self.device} with tracking enabled"
        )

    def process_frame(self, frame: np.ndarray, **kwargs: Any) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)

        # Build multi-ROI polygon arrays (supports single + multi-ROI input)
        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame

        # Run inference with tracking
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.conf_threshold,
            classes=self.classes_to_keep,
            persist=True,
            verbose=False,
        )

        annotated_frame: Optional[np.ndarray] = frame.copy() if annotate else None
        violations_to_record = []  # List of (track_id, violation_type, confidence, bbox)
        all_detections: List[Dict[str, Any]] = []

        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []

            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.class_names.get(cls, f"Class_{cls}")

                # Apply NO-Vest confidence threshold
                if class_name == "no_vest" and conf < self.person_conf_threshold:
                    continue

                # Check if detection center is in any ROI polygon
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                if not is_point_in_any_roi((float(cx), float(cy)), roi_polys):
                    continue

                # Get track ID
                track_id = track_ids[i] if i < len(track_ids) else None

                # Synthetic track_id fallback khi YOLO tracker không gán được ID
                # (thường xảy ra với VSS HTTP-FLV streams)
                if track_id is None:
                    gx, gy = int(cx) // 50, int(cy) // 50
                    track_id = hash((class_name, gx, gy)) & 0x7FFFFFFF

                detection_info = {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "track_id": track_id,
                }
                all_detections.append(detection_info)

                # Check for violations (NO-Vest only)
                if cls == 2:  # 2: NO-Vest
                    violations_to_record.append(
                        (
                            track_id,
                            class_name,
                            conf,
                            [int(x1), int(y1), int(x2), int(y2)],
                        )
                    )

                # Annotate frame
                if annotate and annotated_frame is not None:
                    color = (0, 0, 255) if class_name == "no_vest" else (0, 255, 0)
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

                    (label_w, label_h), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (int(x1), int(y1) - label_h - 10),
                        (int(x1) + label_w, int(y1)),
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

        # Draw ROI polygon
        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys, color=(0, 0, 255))

        # Check which violations should be recorded using object tracker
        # Important: Each violation type for each object creates a separate event
        events_to_create: List[Dict[str, Any]] = []
        for track_id, violation_type, conf, bbox in violations_to_record:
            if self.object_tracker.should_record_event(track_id, violation_type):
                events_to_create.append(
                    {
                        "track_id": track_id,
                        "violation_type": violation_type,
                        "confidence": conf,
                        "bbox": bbox,
                    }
                )
                self.logger.info(
                    f"Ghi nhận vi phạm: {violation_type} | ID: {track_id} | Conf: {conf:.2f}"
                )

        # Prepare metadata (record first event if multiple)
        event_triggered = len(events_to_create) > 0
        primary_event = events_to_create[0] if event_triggered else None

        metadata: Dict[str, Any] = {
            "type": "Bảo hộ lao động",
            "severity": "high" if event_triggered else "low",
            "detections": all_detections,
            "violations": events_to_create,  # All violations to be recorded
            "count": len(all_detections),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "helmet",
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


class PPEModel(BaseModel):
    """Phát hiện vi phạm PPE: không đội mũ bảo hộ"""

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/atld_92.pt",
        confidence_threshold: float = 0.45,
        **kwargs,
    ):
        super().__init__(
            model_name="PPEModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.conf_threshold = kwargs.get("conf_threshold", confidence_threshold)
        self.no_hardhat_threshold = kwargs.get(
            "no_hardhat_threshold", self.conf_threshold
        )

        self.class_names = {
            1: "No Hardhat",
            # 0: "Đội mũ bảo hộ"
        }

        self.violation_classes = {"No Hardhat"}
        self.processed_classes = {"Hardhat"}

        self.object_tracker = ObjectTracker(reset_interval=1800)  # 30 phút

        self.logger.info(
            f"PPEModel loaded | Device: {self.device} | "
            f"Conf: {self.conf_threshold} | Tracking: ON"
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)

        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame

        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.conf_threshold,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_violations = []
        new_compliant = []

        for result in results:
            boxes = result.boxes
            if not boxes:
                continue

            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.class_names.get(cls_id, f"Class_{cls_id}")

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if not is_point_in_any_roi(
                    (float(center[0]), float(center[1])), roi_polys
                ):
                    continue

                # Synthetic track_id fallback khi YOLO tracker không gán được ID
                if track_id is None:
                    gx, gy = center[0] // 50, center[1] // 50
                    track_id = hash((class_name, gx, gy)) & 0x7FFFFFFF

                obj = DetectedObject(
                    label=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    extra={"track_id": track_id, "class_id": cls_id},
                )
                detections.append(obj)

                if True:  # Always process violations (track_id guaranteed by fallback)
                    # VI PHẠM: Không đội mũ bảo hộ
                    if (
                        class_name in self.violation_classes
                        and conf >= self.no_hardhat_threshold
                    ):
                        key = f"PPE_{class_name}"
                        if self.object_tracker.should_record_event(track_id, key):
                            new_violations.append(
                                {
                                    "track_id": track_id,
                                    "type": class_name,
                                    "raw_type": class_name,
                                    "confidence": conf,
                                    "bbox": (x1, y1, x2, y2),
                                    "processed": False,
                                }
                            )
                            self.logger.warning(
                                f"VI PHẠM PPE | {class_name} | ID: {track_id} | Conf: {conf:.2f} | CHƯA XỬ LÝ"
                            )

                    # ĐÃ XỬ LÝ: Đội mũ bảo hộ
                    elif class_name in self.processed_classes:
                        key = f"PPE_{class_name}"
                        if self.object_tracker.should_record_event(track_id, key):
                            new_compliant.append(
                                {
                                    "track_id": track_id,
                                    "type": class_name,
                                    "raw_type": class_name,
                                    "confidence": conf,
                                    "bbox": (x1, y1, x2, y2),
                                    "processed": True,
                                }
                            )
                            self.logger.info(
                                f"ĐÃ XỬ LÝ | {class_name} | ID: {track_id} | Conf: {conf:.2f}"
                            )

                # Vẽ annotation
                if annotate and annotated_frame is not None:
                    color = (
                        (0, 0, 255)
                        if class_name in self.violation_classes
                        else (0, 255, 0)
                    )
                    thickness = 4 if class_name in self.violation_classes else 2
                    status_text = (
                        "UNRESOLVED"
                        if class_name in self.violation_classes
                        else "RESOLVED"
                    )

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"{class_name} {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id} [{status_text}]"

                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - th - 12),
                        (x1 + tw + 10, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        # Vẽ ROI
        if annotate and annotated_frame is not None:
            draw_roi_overlays(
                annotated_frame, roi_polys, color=(0, 255, 255), thickness=3
            )

        # Có event mới không? (chỉ tính vi phạm)
        event_triggered = len(new_violations) > 0

        # Tạo AlertInfo nếu có vi phạm mới
        alert_info = None
        if event_triggered:
            primary = new_violations[0]
            alert_info = self._create_alert_info(
                message=f"CẢNH BÁO CAO: {primary['type']} (ID: {primary['track_id']}) - CHƯA XỬ LÝ",
                confidence=primary["confidence"],
                detected_objects=detections,
                level=AlertLevel.HIGH,
            )

        # === CHÍNH XÁC THEO DIFF ===
        max_conf = max((d.confidence for d in detections), default=0.0)
        primary_conf = new_violations[0]["confidence"] if new_violations else max_conf

        metadata = self._build_metadata(
            alert_info=alert_info,
            extra={
                "type": "Vi phạm mũ bảo hộ",
                "detection_count": len(detections),
                "violation_count": len(new_violations),
                "compliant_count": len(new_compliant),
                "violations": [
                    {
                        "track_id": v["track_id"],
                        "type": v["type"],
                        "display": v["type"],
                        "confidence": v["confidence"],
                        "processed": v["processed"],
                    }
                    for v in new_violations
                ],
                "compliant": [
                    {
                        "track_id": c["track_id"],
                        "type": c["type"],
                        "display": c["type"],
                        "confidence": c["confidence"],
                        "processed": c["processed"],
                    }
                    for c in new_compliant
                ],
                "primary_confidence": primary_conf,
                "max_confidence": max_conf,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        return DetectionResult(
            frame=annotated_frame, event=event_triggered, metadata=metadata
        )
