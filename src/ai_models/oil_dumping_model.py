"""
src/ai_models/oil_dumping_model.py
Oil Dumping (Loitering) Detection Model – HIGH level
Dùng tracker có sẵn + cơ chế cooldown toàn cục + cooldown per-object
"""

from typing import List, Dict, Any, Optional
import threading
import numpy as np
import cv2
import time
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator
from src.utils.roi_utils import (
    DEFAULT_NORMALIZED_ROI,
    denormalize_roi,
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)
from src.utils.alert_levels import AlertLevel


class OilDumpingModel(BaseModel):
    """Phát hiện hành vi đứng lâu trong khu vực cấm (nghi ngờ xả đáy dầu) – HIGH"""

    VIOLATION_KEY = "OIL_DUMPING"

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/yolo11n.pt",
        confidence_threshold: float = 0.4,
        loitering_time_threshold: int = 20,  # đứng bao lâu thì coi là vi phạm (giây)
        global_cooldown: int = 28800,  # 8 tiếng mới được báo lại cùng 1 khu vực
        **kwargs,
    ):
        super().__init__(
            model_name="OilDumpingModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        # Load YOLO (chỉ detect class person = 0)
        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.conf_threshold = kwargs.get("conf_threshold", confidence_threshold)
        self.loitering_time_threshold = kwargs.get(
            "loitering_time_threshold", loitering_time_threshold
        )
        self.global_cooldown = kwargs.get("global_cooldown", global_cooldown)

        # Per-object dedup is entirely delegated to ObjectTracker; there is no
        # additional global cooldown because a global gate swallows independent
        # loitering events from different people.
        save_cooldown = float(kwargs.get("save_cooldown", 2.0))
        self.object_tracker = ObjectTracker(
            reset_interval=self.global_cooldown,
            save_cooldown=save_cooldown,
        )
        self._event_dedup_window_seconds = float(
            kwargs.get("event_dedup_window_seconds", 5.0)
        )
        self._event_dedup_iou_threshold = float(
            kwargs.get("event_dedup_iou_threshold", 0.5)
        )
        self.recent_violation_deduplicator = RecentViolationDeduplicator(
            window_seconds=self._event_dedup_window_seconds,
            iou_threshold=self._event_dedup_iou_threshold,
        )

        # Local per-track time-in-zone bookkeeping. ObjectTracker has no
        # get_time_in_zone() method, so we compute it here from the first
        # time we saw a given track_id inside the ROI.
        # {track_id: first_seen_in_roi_ts}
        self._track_first_seen_in_roi: Dict[int, float] = {}
        # {track_id: last_seen_in_roi_ts} — used to evict tracks that leave
        # the ROI so their loitering timer resets on re-entry.
        self._track_last_seen_in_roi: Dict[int, float] = {}
        self._track_idle_timeout = float(kwargs.get("track_idle_timeout", 5.0))

        # IoU-based fallback for when YOLO's tracker fails to assign an ID.
        # Monotonic counter in a high namespace so it never collides with
        # real ByteTrack IDs.
        self._next_fallback_id = 10_000_000
        self._fallback_iou_threshold = float(kwargs.get("fallback_iou_threshold", 0.3))
        self._last_fallback_bboxes: Dict[int, Dict[str, Any]] = {}
        self._fallback_stale_timeout = float(kwargs.get("fallback_stale_timeout", 3.0))

        # All mutable state is guarded by this lock. YOLO tracker state is
        # per-instance and non-reentrant, so serializing the frame body is
        # both correct and cheap.
        self._state_lock = threading.Lock()

        self.logger.info(
            f"OilDumpingModel loaded | Device: {self.device} | "
            f"Loitering ≥ {self.loitering_time_threshold}s | "
            f"Global cooldown: {self.global_cooldown}s"
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def _get_recent_violation_deduplicator(self) -> RecentViolationDeduplicator:
        deduplicator = getattr(self, "recent_violation_deduplicator", None)
        if deduplicator is None:
            deduplicator = RecentViolationDeduplicator(
                window_seconds=float(
                    getattr(self, "_event_dedup_window_seconds", 5.0)
                ),
                iou_threshold=float(
                    getattr(self, "_event_dedup_iou_threshold", 0.5)
                ),
            )
            self.recent_violation_deduplicator = deduplicator
        return deduplicator

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        recent_violation_deduplicator = self._get_recent_violation_deduplicator()
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)
        current_time = time.time()

        # ROI
        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame

        # Inference + tracking (chỉ person)
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.conf_threshold,
            classes=[0],  # 0 = person trong COCO
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_violations = []

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
                bbox_tuple = (x1, y1, x2, y2)

                # Resolve a stable track_id. IoU-based fallback is used when
                # YOLO's tracker fails to assign one — much more robust than
                # grid-cell hashing, which reset the loitering timer every
                # time a person drifted across a 50px grid boundary.
                effective_track_id = self._resolve_track_id(
                    track_id, bbox_tuple, current_time
                )

                conf = float(box.conf[0])

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                inside_roi = is_point_in_any_roi(center, roi_polys)
                if not inside_roi:
                    # Person left the ROI — drop their loitering timer so a
                    # re-entry starts fresh rather than resuming where they
                    # stopped.
                    self._track_first_seen_in_roi.pop(effective_track_id, None)
                    self._track_last_seen_in_roi.pop(effective_track_id, None)
                    continue

                # Tạo object chuẩn
                obj = DetectedObject(
                    label="Person",
                    confidence=conf,
                    bbox=bbox_tuple,
                    extra={"track_id": effective_track_id},
                )
                detections.append(obj)

                # Locally computed time-in-zone. ObjectTracker does NOT
                # expose get_time_in_zone(), which is why the previous
                # implementation crashed with AttributeError on the first
                # person ever detected inside the ROI.
                first_seen = self._track_first_seen_in_roi.get(effective_track_id)
                if first_seen is None:
                    self._track_first_seen_in_roi[effective_track_id] = current_time
                    first_seen = current_time
                self._track_last_seen_in_roi[effective_track_id] = current_time
                time_in_roi = current_time - first_seen

                # Dedup / burst control is delegated entirely to ObjectTracker.
                # No global cooldown — a global gate swallows independent
                # loitering events from other people in multi-person scenes.
                if (
                    time_in_roi >= self.loitering_time_threshold
                ):
                    is_recent_duplicate = recent_violation_deduplicator.is_recent_duplicate(
                        self.VIOLATION_KEY,
                        effective_track_id,
                        bbox_tuple,
                        current_time=current_time,
                        )
                    if is_recent_duplicate:
                        self.logger.debug(
                            "Suppressed recent duplicate oil dumping event: ID=%s",
                            effective_track_id,
                        )

                    if (
                        not is_recent_duplicate
                        and self.object_tracker.should_record_event(
                            effective_track_id, self.VIOLATION_KEY
                        )
                    ):
                        recent_violation_deduplicator.remember_event(
                            self.VIOLATION_KEY,
                            effective_track_id,
                            bbox_tuple,
                            current_time=current_time,
                        )
                        new_violations.append(
                            {
                                "track_id": effective_track_id,
                                "time_in_roi": round(time_in_roi, 1),
                                "confidence": conf,
                                "bbox": bbox_tuple,
                            }
                        )

                        self.logger.warning(
                        f"NGHI NGỜ XẢ ĐÁY DẦU | ID: {effective_track_id} | "
                        f"Thời gian trong ROI: {time_in_roi:.1f}s | Conf: {conf:.2f}"
                    )

                # Vẽ annotation
                if annotate and annotated_frame is not None:
                    color = (
                        (0, 0, 255)
                        if time_in_roi >= self.loitering_time_threshold
                        else (0, 255, 255)
                    )
                    thickness = 4 if time_in_roi >= self.loitering_time_threshold else 2

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    label = f"ID:{effective_track_id} {time_in_roi:.1f}s"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

        # Vẽ ROI
        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys, thickness=3)

        # Evict tracks we haven't seen inside the ROI for a while. This also
        # clears the loitering timer so a returning person starts fresh.
        self._cleanup_idle_tracks(current_time)
        self._cleanup_stale_fallback_bboxes(current_time)

        # metadata
        event_triggered = len(new_violations) > 0

        detection_dicts: List[Dict[str, Any]] = []
        for d in detections:
            track_id = None
            class_id = None
            if d.extra and isinstance(d.extra, dict):
                track_id = d.extra.get("track_id")
                class_id = d.extra.get("class_id", 0)  # Person class = 0 in COCO

            detection_dicts.append(
                {
                    "class_id": class_id,
                    "class_name": d.label,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "track_id": track_id,
                }
            )

        violations = [
            {
                "track_id": v["track_id"],
                "violation_type": "Nghi ngờ xả đáy dầu",
                "confidence": v["confidence"],
                "bbox": v["bbox"],
            }
            for v in new_violations
        ]

        event_type = "Phát hiện xả đáy"
        for violation in violations:
            violation["violation_type"] = event_type
            violation["event_type"] = event_type
            violation["description"] = "Phát hiện hành vi nghi ngờ xả đáy dầu trong khu vực giám sát"

        primary_event = violations[0] if violations else None

        metadata: Dict[str, Any] = {
            "type": "Xả đáy dầu",
            "severity": "high" if event_triggered else "low",
            "detections": detection_dicts,
            "violations": violations,
            "count": len(detection_dicts),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "oil_dumping",
        }

        metadata["type"] = event_type
        metadata["eventType"] = event_type
        metadata["title"] = event_type
        metadata["description"] = "Phát hiện hành vi nghi ngờ xả đáy dầu trong khu vực giám sát"

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )

    def _resolve_track_id(
        self,
        tracker_id: Optional[int],
        bbox: tuple,
        current_time: float,
    ) -> int:
        """
        Resolve a stable track_id. Trust YOLO's tracker when it gives us one;
        otherwise match against recent fallback bboxes by IoU. The old grid-
        hash fallback reset the loitering timer every time a person drifted
        across a 50px grid boundary, which silently lost 8-hour-window
        loitering events in practice.
        """
        if tracker_id is not None:
            tid = int(tracker_id)
            self._last_fallback_bboxes.pop(tid, None)
            return tid

        best_id: Optional[int] = None
        best_iou = self._fallback_iou_threshold
        for tid, info in self._last_fallback_bboxes.items():
            iou = self._bbox_iou(bbox, info["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id = tid

        if best_id is not None:
            self._last_fallback_bboxes[best_id] = {
                "bbox": bbox,
                "last_seen": current_time,
            }
            return best_id

        new_id = self._next_fallback_id
        self._next_fallback_id += 1
        self._last_fallback_bboxes[new_id] = {
            "bbox": bbox,
            "last_seen": current_time,
        }
        return new_id

    def _cleanup_idle_tracks(self, current_time: float) -> None:
        stale = [
            tid
            for tid, last_seen in self._track_last_seen_in_roi.items()
            if current_time - last_seen > self._track_idle_timeout
        ]
        for tid in stale:
            self._track_first_seen_in_roi.pop(tid, None)
            self._track_last_seen_in_roi.pop(tid, None)

    def _cleanup_stale_fallback_bboxes(self, current_time: float) -> None:
        stale = [
            tid
            for tid, info in self._last_fallback_bboxes.items()
            if current_time - info["last_seen"] > self._fallback_stale_timeout
        ]
        for tid in stale:
            del self._last_fallback_bboxes[tid]

    @staticmethod
    def _bbox_iou(box_a: tuple, box_b: tuple) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union
