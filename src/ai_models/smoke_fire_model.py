"""
src/ai_models/smoke_fire_model.py
"""

from typing import List, Optional, Dict, Any
import threading
import numpy as np
import cv2
import time
import logging
from ultralytics import YOLO

# Tắt warning "not enough matching points" từ BoT-SORT tracker
# IoU fallback tracker xử lý trường hợp này rồi
logging.getLogger("ultralytics").setLevel(logging.ERROR)

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator
from src.utils.roi_utils import (
    denormalize_roi,
    get_roi_polygons,
    DEFAULT_NORMALIZED_ROI,
)
from src.utils.alert_levels import AlertLevel
from src.utils.image_utils import make_cv2_safe_text


class SmokeFireModel(BaseModel):
    """Model phát hiện khói và lửa - CRITICAL level"""
    EVENT_TYPE_MAPPING = {
        "fire": "Phát hiện cháy",
        "smoke": "Phát hiện khói bất thường",
    }
    """Model phát hiện khói và lửa - CRITICAL level"""

    # COCO class ids for person, bicycle, car, motorcycle, bus, truck.
    # The smoke/fire detector frequently fires on humans and vehicles
    # (red clothing, helmets, tail lights, hot exhaust), so detections
    # overlapping these objects are treated as false positives.
    FOREGROUND_CLASS_IDS = [0, 1, 2, 3, 5, 7]

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/YOLOv10-FireSmoke-X.pt",
        confidence_threshold: float = 0.30,
        iou_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            model_name="SmokeFireModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        # Auxiliary detector for persons/vehicles used to filter false positives.
        self.aux_model_path = resolve_engine_path(kwargs.get(
            "aux_model_path",
            kwargs.get("person_model_path", "src/ai_models/model_weights/yolo11n.pt"),
        ), runtime_device=self.device)
        self.aux_model = YOLO(self.aux_model_path)
        if self.aux_model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.aux_model.to(self.device)

        self.conf_threshold = float(kwargs.get("conf_threshold", confidence_threshold))
        self.fire_conf_threshold = float(
            kwargs.get("fire_conf_threshold", self.conf_threshold)
        )
        # Smoke threshold lowered for better recall — false positives are filtered
        # by the static pixel filter + temporal persistence gate below.
        self.smoke_conf_threshold = float(kwargs.get("smoke_conf_threshold", 0.15))
        self.track_conf_threshold = min(
            self.conf_threshold,
            self.fire_conf_threshold,
            self.smoke_conf_threshold,
        )
        self.iou_threshold = kwargs.get("iou_threshold", iou_threshold)

        # Foreground (person / vehicle) false-positive filter.
        # Smoke/fire detections overlapping a detected person or vehicle bbox
        # are dropped. Real fire/smoke in a station is almost always on the
        # ground, signage, or equipment — not on top of a moving person or car.
        self.foreground_conf_threshold = float(
            kwargs.get(
                "foreground_conf_threshold", kwargs.get("person_conf_threshold", 0.35)
            )
        )
        self.foreground_iou_threshold = float(
            kwargs.get(
                "foreground_iou_threshold", kwargs.get("person_iou_threshold", 0.35)
            )
        )
        self.foreground_cover_ratio_threshold = float(
            kwargs.get(
                "foreground_cover_ratio_threshold",
                kwargs.get("person_cover_ratio_threshold", 0.6),
            )
        )
        self.foreground_proximity_ratio = float(
            kwargs.get("foreground_proximity_ratio", 0.18)
        )
        self.recent_foreground_hold_seconds = float(
            kwargs.get("recent_foreground_hold_seconds", 1.0)
        )
        self._recent_foreground_boxes: List[Dict[str, Any]] = []

        # Temporal persistence filter (smoke only):
        # Smoke must be detected continuously for smoke_persist_seconds before
        # an event is recorded.  Fire used to fire immediately, but that
        # caused burned-in camera overlays (timestamps, channel IDs, red
        # characters in watermarks) to trigger fire events at 1.0 confidence
        # on every frame. Requiring even a brief temporal persistence on
        # fire (default 2 s) rejects the "single-frame phantom" FPs without
        # significantly delaying real-fire alerting.
        self.smoke_persist_seconds = float(kwargs.get("smoke_persist_seconds", 1.0))
        self.fire_persist_seconds = float(kwargs.get("fire_persist_seconds", 2.0))
        # Stale timeout: if a track_id vanishes for this many seconds, reset its timer.
        self._stale_timeout = float(kwargs.get("smoke_stale_timeout", 2.0))
        # {track_id: {'first_seen': float, 'last_seen': float, 'class': str}}
        self._pending_detections: Dict[int, Dict[str, Any]] = {}

        # Static pixel filter for smoke:
        # Reject bboxes whose HSV saturation mean is above this ratio — a
        # strongly-colorful region is much more likely to be a false positive
        # than real gray/white smoke.
        self.smoke_max_saturation = float(kwargs.get("smoke_max_saturation", 60.0))  # 0-255 scale

        # Bbox area filtering: loại bỏ detection quá nhỏ hoặc quá lớn (false positive)
        # Tính theo % diện tích frame (0.005 = 0.5%, 0.60 = 60%)
        self.min_bbox_area_ratio = float(kwargs.get("min_bbox_area_ratio", 0.0008))
        self.max_bbox_area_ratio = float(kwargs.get("max_bbox_area_ratio", 0.60))
        self.tail_light_max_area_ratio = float(
            kwargs.get("tail_light_max_area_ratio", 0.012)
        )
        self.tail_light_min_red_ratio = float(
            kwargs.get("tail_light_min_red_ratio", 0.55)
        )
        self.tail_light_max_orange_ratio = float(
            kwargs.get("tail_light_max_orange_ratio", 0.18)
        )
        self.tail_light_min_bright_ratio = float(
            kwargs.get("tail_light_min_bright_ratio", 0.35)
        )
        self.tail_light_max_value_std = float(
            kwargs.get("tail_light_max_value_std", 0.10)
        )
        self.tail_light_min_aspect_ratio = float(
            kwargs.get("tail_light_min_aspect_ratio", 0.35)
        )
        self.tail_light_max_aspect_ratio = float(
            kwargs.get("tail_light_max_aspect_ratio", 3.50)
        )
        self.tail_light_max_motion_ratio = float(
            kwargs.get("tail_light_max_motion_ratio", 0.06)
        )
        self.tail_light_motion_diff_threshold = int(
            kwargs.get("tail_light_motion_diff_threshold", 18)
        )
        self.detection_iou_merge_threshold = float(
            kwargs.get("detection_iou_merge_threshold", 0.6)
        )
        # Guard rails for high-load scenes: avoid one noisy frame producing
        # hundreds of boxes that flood overlays/events.
        self.max_raw_detections = max(1, int(kwargs.get("max_raw_detections", 80)))
        self.max_detections_per_frame = max(
            1, int(kwargs.get("max_detections_per_frame", 25))
        )

        # Tracker: mỗi object chỉ ghi event 1 lần trong reset_interval giây.
        # Per-object dedup + short save cooldown prevents the event flood the
        # "bypass cooldown" path used to produce (one DB row per frame per
        # active fire/smoke detection).
        reset_interval = int(kwargs.get("detection_cooldown", 1800))
        save_cooldown = float(kwargs.get("save_cooldown", 2.0))
        self.object_tracker = ObjectTracker(
            reset_interval=reset_interval,
            save_cooldown=save_cooldown,
        )
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

        # IoU-based fallback khi Ultralytics tracker fail ("not enough matching points")
        # {track_id: {'bbox': (x1,y1,x2,y2), 'class': str, 'last_seen': float}}
        self._last_tracked_boxes: Dict[int, Dict[str, Any]] = {}
        self._next_synthetic_id = 900000  # counter tăng dần, tránh trùng real ID
        self._prev_gray_frame: Optional[np.ndarray] = None

        # Guards all mutable per-instance state AND the YOLO tracker itself
        # (per-instance, non-reentrant).
        self._state_lock = threading.Lock()

        # Mapping class name from model to display label
        self.class_name_mapping = {
            "Fire": "Lua",
            "Smoke": "Khoi",
            "fire": "Lua",
            "smoke": "Khoi",
        }
        self.target_classes = {"fire", "smoke"}

        # Color for each class
        self.colors = {
            "Lua": (0, 0, 255),  # Red
            "Khoi": (128, 128, 128),  # Gray
            "fire": (0, 0, 255),
            "smoke": (128, 128, 128),
        }
        self.logger.info(
            f"SmokeFireModel loaded | Device: {self.device} | "
            f"Track conf: {self.track_conf_threshold} | "
            f"Fire conf: {self.fire_conf_threshold} | "
            f"Smoke conf: {self.smoke_conf_threshold} | "
            f"Smoke persist: {self.smoke_persist_seconds}s | "
            f"Smoke max sat: {self.smoke_max_saturation} | "
            f"BBox area: {self.min_bbox_area_ratio:.3f}-{self.max_bbox_area_ratio:.2f} | "
            f"Merge IoU: {self.detection_iou_merge_threshold:.2f} | "
            f"Max raw/frame: {self.max_raw_detections} | "
            f"Max accepted/frame: {self.max_detections_per_frame} | "
            f"Tail-light filter: Enabled | "
            f"Foreground filter: Enabled (aux={self.aux_model_path}, "
            f"cover>={self.foreground_cover_ratio_threshold:.2f}, "
            f"iou>={self.foreground_iou_threshold:.2f})"
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def cleanup(self) -> None:
        """Explicitly release both YOLO models, the Ultralytics tracker
        state, and all per-instance accumulator dicts.

        The base ``cleanup()`` only calls ``torch.cuda.empty_cache()`` which
        does NOT free model weights until Python's GC decides to reap the
        instance. When ``reload_models()`` fires repeatedly (e.g. the UI
        reassigns a model a few times), every reload used to leave the
        previous ``self.model`` + ``self.aux_model`` pair sitting in GPU
        memory with a persistent Ultralytics tracker holding threads open.

        Order matters: drop refs under the state lock so any in-flight
        inference on the same instance is already finished, wipe the state
        dicts, then delegate to ``BaseModel.cleanup()`` which flushes
        ``torch.cuda.empty_cache()`` AFTER the refs are gone.
        """
        with self._state_lock:
            for attr in ("model", "aux_model"):
                instance = getattr(self, attr, None)
                if instance is None:
                    continue
                try:
                    predictor = getattr(instance, "predictor", None)
                    if predictor is not None:
                        trackers = getattr(predictor, "trackers", None)
                        if trackers:
                            for tracker in trackers:
                                reset = getattr(tracker, "reset", None)
                                if callable(reset):
                                    try:
                                        reset()
                                    except Exception:
                                        pass
                        setattr(predictor, "trackers", None)
                except Exception as exc:
                    self.logger.debug(
                        "SmokeFireModel: failed to reset %s tracker: %s", attr, exc
                    )
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass

            self._pending_detections.clear()
            self._last_tracked_boxes.clear()
            self._recent_foreground_boxes.clear()
            self._prev_gray_frame = None
            self._next_synthetic_id = 900000

        super().cleanup()

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

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)
        inference_frame = kwargs.get("inference_frame")
        recent_violation_deduplicator = self._get_recent_violation_deduplicator()
        max_raw_detections = max(1, int(getattr(self, "max_raw_detections", 80)))
        max_detections_per_frame = max(
            1, int(getattr(self, "max_detections_per_frame", 25))
        )

        # Build denormalized ROI polygons (supports single or multi-ROI input)
        raw_roi = kwargs.get("roi")
        roi_polygon_list = get_roi_polygons(raw_roi) or [DEFAULT_NORMALIZED_ROI]
        roi_polys_px = [
            np.array(denormalize_roi(poly, w, h), dtype=np.int32)
            for poly in roi_polygon_list
        ]

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = (
            inference_frame
            if isinstance(inference_frame, np.ndarray) and inference_frame.size > 0
            else frame
        )
        gray_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)

        # Detect persons / vehicles inside ROI for FP rejection.
        current_time = time.time()
        foreground_boxes = self._detect_foreground_boxes(clean_frame, roi_polys_px)
        self._remember_foreground_boxes(foreground_boxes, current_time)

        # Inference với tracking
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.track_conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            max_det=max_raw_detections,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_violations = []
        dropped_overflow_detections = 0

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
                raw_class_name = result.names.get(cls_id, f"Class_{cls_id}")
                normalized_class_name = str(raw_class_name).strip().lower()
                if normalized_class_name not in self.target_classes:
                    continue
                if conf < self._get_class_conf_threshold(normalized_class_name):
                    continue
                # Chuyển sang tiếng Việt
                class_name = self.class_name_mapping.get(
                    raw_class_name,
                    self.class_name_mapping.get(normalized_class_name, raw_class_name),
                )

                # Kiểm tra tâm bbox trong bất kỳ ROI polygon nào
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                in_any_roi = any(
                    cv2.pointPolygonTest(rp, center, False) >= 0 for rp in roi_polys_px
                )
                if not in_any_roi:
                    continue

                # === Bbox area filtering ===
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = w * h
                area_ratio = bbox_area / frame_area if frame_area > 0 else 0
                if area_ratio < self.min_bbox_area_ratio:
                    continue
                if area_ratio > self.max_bbox_area_ratio:
                    continue
                if self._is_tail_light_false_positive(
                    clean_frame,
                    gray_frame,
                    (x1, y1, x2, y2),
                    area_ratio,
                    normalized_class_name,
                ):
                    continue

                # === Static pixel filter (smoke only) ===
                # Real smoke is gray/white with low saturation.  Reject bbox
                # regions with strong color (likely a solid object or light).
                if normalized_class_name == "smoke":
                    if self._is_colorful_false_positive(clean_frame, (x1, y1, x2, y2)):
                        continue

                # === Foreground (person / vehicle) filter ===
                # Drop detections whose bbox overlaps a person or vehicle, or
                # sits inside a region where one was seen in the last second.
                if self._overlaps_foreground_object(
                    (x1, y1, x2, y2), foreground_boxes
                ):
                    self.logger.debug(
                        "Rejected %s detection overlapping foreground object | bbox=%s",
                        normalized_class_name,
                        (x1, y1, x2, y2),
                    )
                    continue
                if self._has_recent_foreground_activity(
                    (x1, y1, x2, y2), current_time
                ):
                    self.logger.debug(
                        "Rejected %s detection near recent foreground activity | bbox=%s",
                        normalized_class_name,
                        (x1, y1, x2, y2),
                    )
                    continue

                if self._is_duplicate_detection(
                    detections,
                    (x1, y1, x2, y2),
                    normalized_class_name,
                ):
                    continue

                if len(detections) >= max_detections_per_frame:
                    dropped_overflow_detections += 1
                    continue

                # Ultralytics tracker sometimes fails to assign IDs ("not enough matching points").
                # Use IoU-based fallback to maintain stable track_ids.
                effective_track_id = self._resolve_track_id(
                    track_id, (x1, y1, x2, y2), raw_class_name
                )

                obj = DetectedObject(
                    label=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    extra={
                        "track_id": effective_track_id,
                        "class_id": cls_id,
                        "raw_class": raw_class_name,
                        "normalized_class": normalized_class_name,
                        "source_track_id": track_id,
                    },
                )
                detections.append(obj)

                # Ghi event mới (chỉ 1 lần/object) - có temporal filtering
                now = time.time()

                # === Temporal persistence filter ===
                # BOTH smoke AND fire must persist across frames before an
                # event is recorded. Fire used to be immediate-confirm, but
                # that let burned-in camera overlays (timestamp text, channel
                # watermarks) fire at 1.0 confidence on a single frame. A
                # 2-second persistence gate rejects those phantom positives
                # without meaningfully delaying real fire alerting.
                confirmed = False
                if normalized_class_name == "smoke":
                    required_persist = self.smoke_persist_seconds
                else:  # fire
                    required_persist = self.fire_persist_seconds

                if required_persist > 0:
                    if effective_track_id not in self._pending_detections:
                        self._pending_detections[effective_track_id] = {
                            "first_seen": now,
                            "last_seen": now,
                            "class": raw_class_name,
                        }
                    else:
                        pending = self._pending_detections[effective_track_id]
                        # Reset timer if track went stale (gap > stale_timeout)
                        if now - pending["last_seen"] > self._stale_timeout:
                            pending["first_seen"] = now
                        pending["last_seen"] = now
                        elapsed = now - pending["first_seen"]
                        if elapsed >= required_persist:
                            self.logger.info(
                                "%s confirmed: ID=%s — continuous %.1fs ≥ %.1fs",
                                raw_class_name.title(),
                                effective_track_id,
                                elapsed,
                                required_persist,
                            )
                            confirmed = True
                else:
                    # Persistence disabled for this class → confirm immediately.
                    confirmed = True
                    self._pending_detections.pop(effective_track_id, None)

                if confirmed:
                    # Per-object dedup: only record this (track, class) combo
                    # once per reset_interval. Different objects and different
                    # classes on the same object are NOT blocked by any global
                    # cooldown, so concurrent fires/smokes all fire.
                    violation_key = f"SMOKE_FIRE:{raw_class_name.upper()}"
                    is_recent_duplicate = recent_violation_deduplicator.is_recent_duplicate(
                        violation_key,
                        effective_track_id,
                        (x1, y1, x2, y2),
                        current_time=now,
                    )
                    if is_recent_duplicate:
                        self.logger.debug(
                            "Suppressed recent duplicate smoke/fire: %s | ID=%s",
                            raw_class_name,
                            effective_track_id,
                        )

                    if (
                        not is_recent_duplicate
                        and self.object_tracker.should_record_event(
                        effective_track_id, violation_key
                        )
                    ):
                        recent_violation_deduplicator.remember_event(
                            violation_key,
                            effective_track_id,
                            (x1, y1, x2, y2),
                            current_time=now,
                        )
                        new_violations.append(
                            {
                                "track_id": effective_track_id,
                                "type": class_name,
                                "event_type": self.EVENT_TYPE_MAPPING.get(
                                    normalized_class_name,
                                    "Phát hiện khói/cháy",
                                ),
                                "description": (
                                    "Phát hiện cháy rõ ràng trong khu vực giám sát"
                                    if normalized_class_name == "fire"
                                    else "Phát hiện khói bất thường trong khu vực giám sát"
                                ),
                                "confidence": conf,
                                "bbox": (x1, y1, x2, y2),
                            }
                        )
                        self.logger.warning(
                            f"PHÁT HIỆN {class_name.upper()} | ID: {effective_track_id} | Conf: {conf:.2f}"
                        )

                # Vẽ annotation
                if annotate and annotated_frame is not None:
                    color = self.colors.get(class_name, (0, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                    label = f"{class_name} {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    label = make_cv2_safe_text(label, fallback="SmokeFire")

                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
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
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        # Vẽ ROI
        if annotate and annotated_frame is not None:
            for roi_idx, rp in enumerate(roi_polys_px):
                cv2.polylines(
                    annotated_frame, [rp.reshape((-1, 1, 2))], True, (0, 255, 0), 2
                )
                label_pt = rp[0]
                cv2.putText(
                    annotated_frame,
                    f"ROI{roi_idx}" if len(roi_polys_px) > 1 else "ROI",
                    (int(label_pt[0]) + 10, int(label_pt[1]) + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        # Cleanup pending detections cũ (không còn xuất hiện)
        self._cleanup_stale_pending()
        self._cleanup_recent_foreground_boxes(current_time)
        self._prev_gray_frame = gray_frame
        if dropped_overflow_detections > 0:
            self.logger.warning(
                "SmokeFire overflow guard dropped %s detections in one frame (accepted=%s, raw_cap=%s)",
                dropped_overflow_detections,
                max_detections_per_frame,
                max_raw_detections,
            )

        # metadata
        event_triggered = len(new_violations) > 0

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
                    "class_name": d.label,  # Tiếng Việt: Lửa/Khói
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
            for v in new_violations
        ]

        for violation, source in zip(violations, new_violations):
            violation["event_type"] = source.get("event_type")
            violation["description"] = source.get("description")

        primary_event = violations[0] if violations else None
        event_type = (
            primary_event.get("event_type")
            if primary_event
            else "Phát hiện khói/cháy"
        )

        metadata: Dict[str, Any] = {
            "type": "Báo cháy",
            "severity": "high" if event_triggered else "low",
            "detections": detection_dicts,
            "violations": violations,  # All violations to be recorded
            "count": len(detection_dicts),
            "dropped_overflow_detections": dropped_overflow_detections,
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "smoke_fire",
        }

        metadata["type"] = event_type
        metadata["eventType"] = event_type
        metadata["title"] = event_type
        metadata["description"] = (
            primary_event.get("description")
            if primary_event
            else "Phát hiện khói hoặc cháy trong khu vực giám sát"
        )

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )

    def _get_class_conf_threshold(self, normalized_class_name: str) -> float:
        if normalized_class_name == "fire":
            return self.fire_conf_threshold
        if normalized_class_name == "smoke":
            return self.smoke_conf_threshold
        return self.conf_threshold

    def _is_colorful_false_positive(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> bool:
        """Return True if the bbox region is too colorful to be smoke.

        Real smoke / haze is gray or white (low HSV saturation).
        A strongly-colored region suggests a false positive (equipment, signal
        light, painted surface, etc.).

        Parameters
        ----------
        frame : BGR frame
        bbox  : (x1, y1, x2, y2) pixel coordinates
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mean_saturation = float(hsv[:, :, 1].mean())

        if mean_saturation > self.smoke_max_saturation:
            self.logger.debug(
                f"Static pixel filter: rejected smoke bbox ({x1},{y1},{x2},{y2}) "
                f"— saturation {mean_saturation:.1f} > {self.smoke_max_saturation}"
            )
            return True
        return False

    def _is_tail_light_false_positive(
        self,
        frame: np.ndarray,
        gray_frame: np.ndarray,
        bbox,
        area_ratio: float,
        normalized_class_name: str,
    ) -> bool:
        if normalized_class_name != "fire":
            return False
        if area_ratio > self.tail_light_max_area_ratio:
            return False

        x1, y1, x2, y2 = self._clip_bbox_to_frame(bbox, frame.shape[1], frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return False

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        aspect_ratio = bbox_w / max(bbox_h, 1)
        if aspect_ratio < self.tail_light_min_aspect_ratio:
            return False
        if aspect_ratio > self.tail_light_max_aspect_ratio:
            return False

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv_crop)
        red_mask = (
            ((h_channel <= 10) | (h_channel >= 170))
            & (s_channel >= 120)
            & (v_channel >= 110)
        )
        orange_mask = (
            (h_channel >= 8) & (h_channel <= 35) & (s_channel >= 70) & (v_channel >= 90)
        )
        bright_mask = v_channel >= 150

        red_ratio = float(np.count_nonzero(red_mask) / red_mask.size)
        orange_ratio = float(np.count_nonzero(orange_mask) / orange_mask.size)
        bright_ratio = float(np.count_nonzero(bright_mask) / bright_mask.size)
        value_std_ratio = float(np.std(v_channel) / 255.0)
        motion_ratio = self._compute_bbox_motion_ratio(gray_frame, (x1, y1, x2, y2))

        if red_ratio < self.tail_light_min_red_ratio:
            return False
        if orange_ratio > self.tail_light_max_orange_ratio:
            return False
        if bright_ratio < self.tail_light_min_bright_ratio:
            return False
        if value_std_ratio > self.tail_light_max_value_std:
            return False
        if motion_ratio is not None and motion_ratio > self.tail_light_max_motion_ratio:
            return False

        self.logger.info(
            "Rejected tail-light-like fire detection | bbox=%s | red=%.2f | orange=%.2f | bright=%.2f | std=%.2f | motion=%s",
            (x1, y1, x2, y2),
            red_ratio,
            orange_ratio,
            bright_ratio,
            value_std_ratio,
            f"{motion_ratio:.3f}" if motion_ratio is not None else "n/a",
        )
        return True

    def _compute_bbox_motion_ratio(
        self, gray_frame: np.ndarray, bbox
    ) -> Optional[float]:
        if self._prev_gray_frame is None:
            return None
        if self._prev_gray_frame.shape != gray_frame.shape:
            return None

        x1, y1, x2, y2 = self._clip_bbox_to_frame(
            bbox, gray_frame.shape[1], gray_frame.shape[0]
        )
        if x2 <= x1 or y2 <= y1:
            return None

        current_crop = gray_frame[y1:y2, x1:x2]
        prev_crop = self._prev_gray_frame[y1:y2, x1:x2]
        if current_crop.size == 0 or prev_crop.size == 0:
            return None

        diff = cv2.absdiff(current_crop, prev_crop)
        _, motion_mask = cv2.threshold(
            diff,
            self.tail_light_motion_diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        return float(cv2.countNonZero(motion_mask) / motion_mask.size)

    @staticmethod
    def _clip_bbox_to_frame(bbox, frame_width: int, frame_height: int):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), frame_width))
        y1 = max(0, min(int(y1), frame_height))
        x2 = max(0, min(int(x2), frame_width))
        y2 = max(0, min(int(y2), frame_height))
        return x1, y1, x2, y2

    def _resolve_track_id(self, tracker_id, bbox, class_name) -> int:
        """Gán track_id ổn định: dùng tracker_id nếu có, fallback IoU matching."""
        now = time.time()
        if tracker_id is not None:
            # Tracker thành công → lưu bbox để fallback sau
            self._last_tracked_boxes[tracker_id] = {
                "bbox": bbox,
                "class": class_name,
                "last_seen": now,
            }
            return tracker_id

        # Tracker fail → IoU matching với boxes đã biết
        best_id, best_iou = None, 0.3
        for tid, info in self._last_tracked_boxes.items():
            if info["class"] != class_name:
                continue
            iou = self._compute_iou(bbox, info["bbox"])
            if iou > best_iou:
                best_id, best_iou = tid, iou

        if best_id is not None:
            self._last_tracked_boxes[best_id]["bbox"] = bbox
            self._last_tracked_boxes[best_id]["last_seen"] = now
            return best_id

        # Không match → tạo ID mới (counter tăng dần, ổn định hơn hash)
        new_id = self._next_synthetic_id
        self._next_synthetic_id += 1
        self._last_tracked_boxes[new_id] = {
            "bbox": bbox,
            "class": class_name,
            "last_seen": now,
        }
        return new_id

    @staticmethod
    def _compute_iou(box_a, box_b) -> float:
        """Tính IoU giữa 2 bbox (x1,y1,x2,y2)."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _is_duplicate_detection(
        self,
        accepted_detections: List[DetectedObject],
        bbox: tuple,
        normalized_class_name: str,
    ) -> bool:
        merge_iou = float(getattr(self, "detection_iou_merge_threshold", 0.6))
        if merge_iou <= 0:
            return False

        for existing in accepted_detections:
            existing_bbox = existing.bbox
            if existing_bbox is None:
                continue

            existing_extra = existing.extra if isinstance(existing.extra, dict) else {}
            existing_class = str(
                existing_extra.get("normalized_class")
                or existing_extra.get("raw_class")
                or ""
            ).strip().lower()
            if existing_class != normalized_class_name:
                continue

            if self._compute_iou(existing_bbox, bbox) >= merge_iou:
                return True

        return False

    def _detect_foreground_boxes(
        self, frame: np.ndarray, roi_polys: List[np.ndarray]
    ) -> List[tuple]:
        runtime_kwargs = self._get_yolo_runtime_kwargs(
            conf=self.foreground_conf_threshold,
            iou=self.foreground_iou_threshold,
            classes=self.FOREGROUND_CLASS_IDS,
            verbose=False,
        )
        # predict() does not accept tracker / vid_stride; strip them out.
        runtime_kwargs.pop("tracker", None)
        runtime_kwargs.pop("vid_stride", None)

        try:
            results = self.aux_model.predict(source=frame, **runtime_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Foreground aux detector failed: %s", exc)
            return []

        foreground_boxes: List[tuple] = []
        for result in results:
            boxes = result.boxes
            if not boxes:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                in_any_roi = any(
                    cv2.pointPolygonTest(rp, center, False) >= 0 for rp in roi_polys
                )
                if not in_any_roi:
                    continue
                foreground_boxes.append((x1, y1, x2, y2))
        return foreground_boxes

    def _remember_foreground_boxes(
        self, foreground_boxes: List[tuple], current_time: float
    ) -> None:
        self._cleanup_recent_foreground_boxes(current_time)
        for bbox in foreground_boxes:
            self._recent_foreground_boxes.append(
                {
                    "bbox": bbox,
                    "last_seen": current_time,
                }
            )

    def _cleanup_recent_foreground_boxes(self, current_time: float) -> None:
        self._recent_foreground_boxes = [
            info
            for info in self._recent_foreground_boxes
            if current_time - info["last_seen"] <= self.recent_foreground_hold_seconds
        ]

    def _overlaps_foreground_object(
        self, detection_bbox: tuple, foreground_boxes: List[tuple]
    ) -> bool:
        det_area = self._bbox_area(detection_bbox)
        if det_area <= 0 or not foreground_boxes:
            return False

        det_center = (
            (detection_bbox[0] + detection_bbox[2]) // 2,
            (detection_bbox[1] + detection_bbox[3]) // 2,
        )

        for fg_bbox in foreground_boxes:
            if self._point_in_bbox(det_center, fg_bbox):
                return True

            intersection_area = self._intersection_area(detection_bbox, fg_bbox)
            if intersection_area <= 0:
                continue

            cover_ratio = intersection_area / det_area
            if cover_ratio >= self.foreground_cover_ratio_threshold:
                return True

            if self._bbox_iou(detection_bbox, fg_bbox) >= self.foreground_iou_threshold:
                return True

        return False

    def _has_recent_foreground_activity(
        self, detection_bbox: tuple, current_time: float
    ) -> bool:
        self._cleanup_recent_foreground_boxes(current_time)
        if not self._recent_foreground_boxes:
            return False

        expanded_bbox = self._expand_bbox(
            detection_bbox, self.foreground_proximity_ratio
        )
        det_center = (
            (detection_bbox[0] + detection_bbox[2]) // 2,
            (detection_bbox[1] + detection_bbox[3]) // 2,
        )

        for info in self._recent_foreground_boxes:
            fg_bbox = info["bbox"]
            if self._point_in_bbox(det_center, fg_bbox):
                return True
            if self._intersection_area(expanded_bbox, fg_bbox) > 0:
                return True
        return False

    @staticmethod
    def _expand_bbox(bbox: tuple, margin_ratio: float) -> tuple:
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        dx = int(width * margin_ratio)
        dy = int(height * margin_ratio)
        return x1 - dx, y1 - dy, x2 + dx, y2 + dy

    @staticmethod
    def _bbox_area(bbox: tuple) -> int:
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _intersection_area(bbox_a: tuple, bbox_b: tuple) -> int:
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _bbox_iou(cls, bbox_a: tuple, bbox_b: tuple) -> float:
        intersection_area = cls._intersection_area(bbox_a, bbox_b)
        if intersection_area <= 0:
            return 0.0
        union_area = cls._bbox_area(bbox_a) + cls._bbox_area(bbox_b) - intersection_area
        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def _point_in_bbox(point: tuple, bbox: tuple) -> bool:
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    def _cleanup_stale_pending(self):
        """Xóa các pending detection đã không xuất hiện quá _stale_timeout giây.
        Cũng dọn _last_tracked_boxes cũ hơn 3 giây."""
        now = time.time()
        stale_ids = [
            tid
            for tid, info in self._pending_detections.items()
            if now - info["last_seen"] > self._stale_timeout
        ]
        for tid in stale_ids:
            self.logger.debug(
                f"Pending expired: ID={tid} - không thấy trong {self._stale_timeout}s → xóa"
            )
            del self._pending_detections[tid]

        # Dọn fallback boxes cũ
        stale_box_ids = [
            tid
            for tid, info in self._last_tracked_boxes.items()
            if now - info["last_seen"] > 3.0
        ]
        for tid in stale_box_ids:
            del self._last_tracked_boxes[tid]

