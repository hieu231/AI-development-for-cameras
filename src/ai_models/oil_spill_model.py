"""
src/ai_models/oil_spill_model.py
"""

from typing import Any, Dict, List, Optional
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator
from src.utils.alert_levels import AlertLevel
from src.utils.roi_utils import (
    DEFAULT_NORMALIZED_ROI,
    denormalize_roi,
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)


class OilSpillModel(BaseModel):
    """Detect oil spill regions on the ground."""

    FALLBACK_GRID_SIZE = 40
    FOREGROUND_CLASS_IDS = [0, 1, 2, 3, 5, 7]

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/oil_11l_100ep.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            model_name="OilSpillModel",
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.aux_model_path = resolve_engine_path(kwargs.get(
            "aux_model_path",
            kwargs.get("person_model_path", "src/ai_models/model_weights/yolo11n.pt"),
        ), runtime_device=self.device)
        self.aux_model = YOLO(self.aux_model_path)
        if self.aux_model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.aux_model.to(self.device)

        self.conf_threshold = kwargs.get("conf_threshold", confidence_threshold)
        self.iou_threshold = kwargs.get("iou_threshold", iou_threshold)
        self.detection_cooldown = int(kwargs.get("detection_cooldown", 1800))
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

        # Post-processing tuned to reject tall or unstable false positives.
        # Defaults relaxed so elongated / long spills are not filtered out before
        # the temporal state machine gets to see them.
        self.min_bbox_area_ratio = float(kwargs.get("min_bbox_area_ratio", 0.003))
        # Large real spills can occupy most of the vertical frame in overhead /
        # oblique station cameras. 0.5 was too strict and silently rejected
        # those cases before the temporal confirmation pipeline even ran.
        # Keep a high-but-not-unbounded ceiling so obviously full-frame false
        # positives are still filtered.
        self.max_bbox_height_ratio = float(kwargs.get("max_bbox_height_ratio", 0.85))
        self.min_aspect_ratio = float(kwargs.get("min_aspect_ratio", 0.0))
        self.max_aspect_ratio = float(kwargs.get("max_aspect_ratio", 8.0))
        self.max_motion_ratio = float(kwargs.get("max_motion_ratio", 0.12))
        self.max_edge_motion_ratio = float(kwargs.get("max_edge_motion_ratio", 0.2))
        self.max_center_shift_ratio = float(kwargs.get("max_center_shift_ratio", 0.08))
        # Large amorphous spill blobs jitter more between frames than compact
        # detections. Relax these thresholds so bbox wobble does not keep
        # resetting otherwise stable large-spill candidates.
        self.max_area_change_ratio = float(kwargs.get("max_area_change_ratio", 0.8))
        self.max_area_shrink_ratio = float(kwargs.get("max_area_shrink_ratio", 0.25))
        self.min_area_growth_ratio = float(kwargs.get("min_area_growth_ratio", 0.08))
        self.motion_history_window = max(3, int(kwargs.get("motion_history_window", 5)))
        self.min_growth_steps = max(1, int(kwargs.get("min_growth_steps", 2)))
        self.min_consecutive_frames = max(
            1, int(kwargs.get("min_consecutive_frames", 3))
        )
        self.min_stable_seconds = float(kwargs.get("min_stable_seconds", 0.6))
        self.motion_diff_threshold = int(kwargs.get("motion_diff_threshold", 20))
        self.motion_blur_kernel = self._normalize_odd_kernel_size(
            kwargs.get("motion_blur_kernel", 5)
        )
        self.motion_morph_kernel = self._normalize_odd_kernel_size(
            kwargs.get("motion_morph_kernel", 5)
        )
        self.edge_band_ratio = float(kwargs.get("edge_band_ratio", 0.2))
        self._stale_timeout = float(kwargs.get("stale_timeout", 3.0))
        self._pending_detections: Dict[int, Dict[str, Any]] = {}
        self._recent_foreground_boxes: List[Dict[str, Any]] = []
        self._prev_gray_frame = None

        # All mutable per-instance state (pending detections, foreground memory,
        # prev-frame) is guarded by this lock. process_frame may be invoked from
        # the camera processing thread and from admin/debug paths; the YOLO
        # tracker itself also maintains per-instance state that must not be
        # re-entered concurrently, so scoping the lock around the frame body is
        # both correct and simple.
        self._state_lock = threading.Lock()

        # Monotonic counter for fallback detection keys (used when the YOLO
        # tracker fails to assign an ID to a spill blob). Kept in a negative
        # namespace so these synthetic keys can never collide with real
        # ByteTrack IDs.
        self._next_fallback_idx = 0
        self._fallback_iou_threshold = float(kwargs.get("fallback_iou_threshold", 0.3))

        # Per-object event dedup / burst control. We rely on ObjectTracker for
        # both the 30-minute repeat suppression and the short save cooldown;
        # there is deliberately no additional global cooldown here, because a
        # global cooldown silently swallows independent spills from other
        # tracks in multi-spill scenes.
        save_cooldown = float(kwargs.get("save_cooldown", 2.0))
        self.object_tracker = ObjectTracker(
            reset_interval=self.detection_cooldown,
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

        self.logger.info(
            "OilSpillModel loaded | Device: %s | Conf: %.2f | Area >= %.3f | "
            "Max height <= %.2f | Aspect >= %.2f | Motion <= %.2f | Edge motion <= %.2f | "
            "Growth >= %.2f | Shrink <= %.2f | Frames >= %s | Stable >= %.2fs | Cooldown: %ss",
            self.device,
            self.conf_threshold,
            self.min_bbox_area_ratio,
            self.max_bbox_height_ratio,
            self.min_aspect_ratio,
            self.max_motion_ratio,
            self.max_edge_motion_ratio,
            self.min_area_growth_ratio,
            self.max_area_shrink_ratio,
            self.min_consecutive_frames,
            self.min_stable_seconds,
            self.detection_cooldown,
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        # All per-instance mutable state is touched under this lock. The
        # Ultralytics tracker itself also keeps per-instance state that must
        # not be re-entered, so scoping the lock around the whole frame is
        # both correct and cheap (only one thread per camera instance runs
        # inference at a time in practice).
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

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
        recent_violation_deduplicator = self._get_recent_violation_deduplicator()
        h, w = frame.shape[:2]
        frame_area = w * h
        annotate = kwargs.get("annotate", True)
        current_time = time.time()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.motion_blur_kernel > 1:
            gray_frame = cv2.GaussianBlur(
                gray_frame,
                (self.motion_blur_kernel, self.motion_blur_kernel),
                0,
            )

        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)
        foreground_boxes = self._detect_foreground_boxes(frame, roi_polys)
        self._remember_foreground_boxes(foreground_boxes, current_time)

        results = self._run_yolo_track(
            self.model,
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_violations = []
        claimed_detection_keys: set[int] = set()

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
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if not is_point_in_any_roi(center, roi_polys):
                    continue

                if self._overlaps_foreground_object((x1, y1, x2, y2), foreground_boxes):
                    continue
                if self._has_recent_foreground_activity((x1, y1, x2, y2), current_time):
                    continue

                bbox_area = bbox_w * bbox_h
                area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
                if area_ratio < self.min_bbox_area_ratio:
                    continue

                height_ratio = bbox_h / h if h > 0 else 0.0
                if height_ratio > self.max_bbox_height_ratio:
                    continue

                aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0
                if aspect_ratio < self.min_aspect_ratio:
                    continue
                if self.max_aspect_ratio > 0 and aspect_ratio > self.max_aspect_ratio:
                    # Reject extremely thin vertical strips (pillars, poles).
                    continue

                motion_ratio, edge_motion_ratio = self._compute_motion_metrics(
                    (x1, y1, x2, y2), gray_frame
                )
                if motion_ratio > self.max_motion_ratio:
                    continue
                if edge_motion_ratio > self.max_edge_motion_ratio:
                    continue

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_class_name = result.names.get(cls_id, "Spill")
                class_name = "Vet loang bat thuong"

                detection_key = self._resolve_detection_key(
                    track_id,
                    (x1, y1, x2, y2),
                    claimed_detection_keys,
                )
                claimed_detection_keys.add(detection_key)
                pending = self._pending_detections.get(detection_key)
                if (
                    pending
                    and current_time - pending["last_seen"] <= self._stale_timeout
                ):
                    if self._is_stable_candidate(
                        pending,
                        center,
                        bbox_area,
                        motion_ratio,
                        edge_motion_ratio,
                        w,
                        h,
                    ):
                        pending["count"] += 1
                    else:
                        pending["count"] = 1
                        pending["first_seen"] = current_time
                    pending["last_seen"] = current_time
                    pending["previous_center"] = pending.get("center", center)
                    pending["previous_area"] = int(pending.get("area", bbox_area))
                    pending["bbox"] = (x1, y1, x2, y2)
                    pending["confidence"] = conf
                    pending["track_id"] = track_id
                    pending["center"] = center
                    pending["area"] = bbox_area
                    pending["motion_ratio"] = motion_ratio
                    pending["edge_motion_ratio"] = edge_motion_ratio
                else:
                    pending = {
                        "count": 1,
                        "first_seen": current_time,
                        "last_seen": current_time,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "track_id": track_id,
                        "initial_center": center,
                        "previous_center": center,
                        "center": center,
                        "initial_area": bbox_area,
                        "previous_area": bbox_area,
                        "area": bbox_area,
                        "motion_history": [],
                        "motion_ratio": motion_ratio,
                        "edge_motion_ratio": edge_motion_ratio,
                    }
                    self._pending_detections[detection_key] = pending

                self._record_motion_history(pending, current_time, center, bbox_area)

                if not self._has_matured_candidate(pending, current_time):
                    continue
                if not self._has_confirmation_signal(pending, bbox_area, current_time):
                    continue

                effective_track_id = track_id if track_id is not None else detection_key
                obj = DetectedObject(
                    label=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    extra={
                        "track_id": effective_track_id,
                        "class_id": cls_id,
                        "raw_class": raw_class_name,
                    },
                )
                detections.append(obj)

                # Dedup & burst control is entirely delegated to ObjectTracker:
                #   * same (track_id, "OIL_SPILL") is suppressed until the track
                #     is evicted after detection_cooldown seconds;
                #   * independent spills (different track IDs / fallback keys)
                #     are NOT blocked by any global cooldown, so concurrent
                #     spills in multi-spill scenes are all captured.
                is_recent_duplicate = recent_violation_deduplicator.is_recent_duplicate(
                    "OIL_SPILL",
                    effective_track_id,
                    (x1, y1, x2, y2),
                    current_time=current_time,
                )
                if is_recent_duplicate:
                    self.logger.debug(
                        "Suppressed recent duplicate oil spill: ID=%s",
                        effective_track_id,
                    )

                if (
                    not is_recent_duplicate
                    and self.object_tracker.should_record_event(
                    effective_track_id, "OIL_SPILL"
                    )
                ):
                    recent_violation_deduplicator.remember_event(
                        "OIL_SPILL",
                        effective_track_id,
                        (x1, y1, x2, y2),
                        current_time=current_time,
                    )
                    new_violations.append(
                        {
                            "track_id": effective_track_id,
                            "type": class_name,
                            "confidence": conf,
                            "bbox": (x1, y1, x2, y2),
                        }
                    )
                    self.logger.warning(
                        "OIL SPILL DETECTED | ID: %s | Conf: %.2f",
                        effective_track_id,
                        conf,
                    )

                if annotate and annotated_frame is not None:
                    color = (0, 100, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    label = f"VET LOANG BAT THUONG {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        annotated_frame, (x1, y1 - th - 12), (x1 + tw, y1), color, -1
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

        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys)

        self._cleanup_stale_pending(current_time)
        self._cleanup_recent_foreground_boxes(current_time)
        self._prev_gray_frame = gray_frame

        event_triggered = len(new_violations) > 0

        detection_dicts: List[Dict[str, Any]] = []
        for detection in detections:
            track_id = None
            class_id = None
            if detection.extra and isinstance(detection.extra, dict):
                track_id = detection.extra.get("track_id")
                class_id = detection.extra.get("class_id")

            detection_dicts.append(
                {
                    "class_id": class_id,
                    "class_name": detection.label,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                    "track_id": track_id,
                }
            )

        violations = [
            {
                "track_id": violation["track_id"],
                "violation_type": violation["type"],
                "confidence": violation["confidence"],
                "bbox": violation["bbox"],
            }
            for violation in new_violations
        ]

        primary_event = violations[0] if violations else None
        event_type = "Xuất hiện vết loang bất thường"
        metadata: Dict[str, Any] = {
            "type": event_type,
            "eventType": event_type,
            "title": event_type,
            "severity": "high" if event_triggered else "low",
            "description": "Phát hiện vết loang bất thường trong khu vực giám sát",
            "detections": detection_dicts,
            "violations": violations,
            "count": len(detection_dicts),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "oil_spill",
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

    def _resolve_detection_key(
        self,
        track_id: Optional[int],
        bbox: tuple,
        claimed_detection_keys: set[int],
    ) -> int:
        """
        Map a raw detection to a stable pending-slot key.

        When ByteTrack gives us a track_id we trust it directly. When tracking
        fails (common for amorphous oil-spill blobs, especially slow-spreading
        ones), we match against existing fallback slots by bbox IoU. A spill
        that drifts across a grid boundary now keeps the same key — the old
        grid-hash fallback reset the confirmation counter every time the
        centroid crossed a 40px cell edge, which quietly killed recall near
        boundaries.

        Fallback keys live in a negative namespace so they can never collide
        with real (positive) ByteTrack IDs, even across long sessions.
        """
        if track_id is not None:
            return int(track_id)

        best_key: Optional[int] = None
        best_iou = 0.0
        for key, pending in self._pending_detections.items():
            if key >= 0:
                # Don't re-associate a fallback detection with a real tracked
                # object's slot — that would corrupt its history.
                continue
            if key in claimed_detection_keys:
                # Another detection in this same frame already claimed this
                # pending slot. Reusing it here would collapse concurrent,
                # independent spills into one logical object.
                continue
            existing_bbox = pending.get("bbox")
            if not existing_bbox:
                continue
            iou = self._bbox_iou(bbox, existing_bbox)
            if iou > best_iou:
                best_iou = iou
                best_key = key

        if best_key is not None and best_iou >= self._fallback_iou_threshold:
            return best_key

        self._next_fallback_idx += 1
        return -self._next_fallback_idx

    def _cleanup_stale_pending(self, current_time: float) -> None:
        stale_keys = [
            key
            for key, info in self._pending_detections.items()
            if current_time - info["last_seen"] > self._stale_timeout
        ]
        for key in stale_keys:
            del self._pending_detections[key]

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

    def _detect_foreground_boxes(
        self, frame: np.ndarray, roi_polys: List[np.ndarray]
    ) -> List[tuple]:
        runtime_kwargs = self._get_yolo_runtime_kwargs(
            conf=self.foreground_conf_threshold,
            iou=self.foreground_iou_threshold,
            classes=self.FOREGROUND_CLASS_IDS,
            verbose=False,
        )
        runtime_kwargs.pop("tracker", None)
        runtime_kwargs.pop("vid_stride", None)

        results = self.aux_model.predict(source=frame, **runtime_kwargs)
        foreground_boxes: List[tuple] = []

        for result in results:
            boxes = result.boxes
            if not boxes:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if not is_point_in_any_roi(center, roi_polys):
                    continue
                foreground_boxes.append((x1, y1, x2, y2))

        return foreground_boxes

    def _overlaps_foreground_object(
        self, spill_bbox: tuple, foreground_boxes: List[tuple]
    ) -> bool:
        spill_area = self._bbox_area(spill_bbox)
        if spill_area <= 0:
            return False

        spill_center = (
            (spill_bbox[0] + spill_bbox[2]) // 2,
            (spill_bbox[1] + spill_bbox[3]) // 2,
        )

        for foreground_bbox in foreground_boxes:
            if self._point_in_bbox(spill_center, foreground_bbox):
                return True

            intersection_area = self._intersection_area(spill_bbox, foreground_bbox)
            if intersection_area <= 0:
                continue

            spill_cover_ratio = intersection_area / spill_area
            if spill_cover_ratio >= self.foreground_cover_ratio_threshold:
                return True

            if (
                self._bbox_iou(spill_bbox, foreground_bbox)
                >= self.foreground_iou_threshold
            ):
                return True

        return False

    def _has_recent_foreground_activity(
        self, spill_bbox: tuple, current_time: float
    ) -> bool:
        self._cleanup_recent_foreground_boxes(current_time)
        if not self._recent_foreground_boxes:
            return False

        expanded_spill_bbox = self._expand_bbox(
            spill_bbox, self.foreground_proximity_ratio
        )
        spill_center = (
            (spill_bbox[0] + spill_bbox[2]) // 2,
            (spill_bbox[1] + spill_bbox[3]) // 2,
        )

        for info in self._recent_foreground_boxes:
            foreground_bbox = info["bbox"]
            if self._point_in_bbox(spill_center, foreground_bbox):
                return True
            if self._intersection_area(expanded_spill_bbox, foreground_bbox) > 0:
                return True

        return False

    def _compute_motion_metrics(self, bbox: tuple, gray_frame: np.ndarray) -> tuple:
        if self._prev_gray_frame is None:
            return 0.0, 0.0

        x1, y1, x2, y2 = bbox
        prev_crop = self._prev_gray_frame[y1:y2, x1:x2]
        curr_crop = gray_frame[y1:y2, x1:x2]
        if (
            prev_crop.size == 0
            or curr_crop.size == 0
            or prev_crop.shape != curr_crop.shape
        ):
            return 0.0, 0.0

        # Normalize global brightness offset within the crop before diffing.
        # Camera auto-exposure and ambient lighting changes produce a near-
        # uniform shift across the whole crop that absdiff would misread as
        # motion (filling the whole bbox with "change"). Subtracting the mean
        # offset rejects that false signal while preserving actual local
        # motion (people, vehicles, real pixel changes).
        prev_mean = float(prev_crop.mean())
        curr_mean = float(curr_crop.mean())
        brightness_offset = curr_mean - prev_mean
        if abs(brightness_offset) > 1.0:
            curr_for_diff = np.clip(
                curr_crop.astype(np.int16) - int(round(brightness_offset)),
                0,
                255,
            ).astype(np.uint8)
        else:
            curr_for_diff = curr_crop

        diff = cv2.absdiff(prev_crop, curr_for_diff)
        _, motion_mask = cv2.threshold(
            diff, self.motion_diff_threshold, 255, cv2.THRESH_BINARY
        )
        if self.motion_morph_kernel > 1:
            kernel = np.ones(
                (self.motion_morph_kernel, self.motion_morph_kernel), dtype=np.uint8
            )
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        changed_pixels = cv2.countNonZero(motion_mask)
        total_pixels = motion_mask.size
        if total_pixels <= 0:
            return 0.0, 0.0
        return changed_pixels / total_pixels, self._compute_edge_motion_ratio(
            motion_mask
        )

    def _is_stable_candidate(
        self,
        pending: Dict[str, Any],
        center: tuple,
        bbox_area: int,
        motion_ratio: float,
        edge_motion_ratio: float,
        frame_w: int,
        frame_h: int,
    ) -> bool:
        history = pending.get("motion_history") or []
        initial_center = pending.get("initial_center", pending.get("center", center))
        prev_center = pending.get("previous_center", pending.get("center", center))
        prev_area = max(
            1, int(pending.get("previous_area", pending.get("area", bbox_area)))
        )
        frame_diagonal = max(1.0, float(np.hypot(frame_w, frame_h)))
        center_shift = float(
            np.hypot(center[0] - initial_center[0], center[1] - initial_center[1])
        )
        frame_shift = float(
            np.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
        )
        center_shift_ratio = center_shift / frame_diagonal
        frame_shift_ratio = frame_shift / frame_diagonal
        area_change_ratio = abs(bbox_area - prev_area) / prev_area

        if history:
            history_centers = [entry["center"] for entry in history] + [center]
            step_shifts = [
                float(np.hypot(curr[0] - prev[0], curr[1] - prev[1])) / frame_diagonal
                for prev, curr in zip(history_centers, history_centers[1:])
            ]
            if step_shifts:
                avg_step_shift_ratio = sum(step_shifts) / len(step_shifts)
                if avg_step_shift_ratio > self.max_center_shift_ratio:
                    return False

        # NOTE: growth_steps is intentionally NOT checked here. Stability means
        # "stationary + low motion", not "actively growing". Previously this
        # check reset pending["count"] to 1 on every frame for an already-
        # formed static spill, which is exactly why long/static spills were
        # never confirmed. Growth is evaluated separately in
        # _has_confirmation_signal as one of two accepted confirmation paths.

        if center_shift_ratio > self.max_center_shift_ratio:
            return False
        if frame_shift_ratio > self.max_center_shift_ratio:
            return False
        if area_change_ratio > self.max_area_change_ratio:
            return False
        if motion_ratio > self.max_motion_ratio:
            return False
        if edge_motion_ratio > self.max_edge_motion_ratio:
            return False
        return True

    def _has_confirmation_signal(
        self,
        pending: Dict[str, Any],
        bbox_area: int,
        current_time: float,
    ) -> bool:
        """
        Confirm a candidate via EITHER of two paths:

          Path A — "actively spreading": the bbox area is growing over the
            motion-history window (classic oil-spill-in-progress signature).

          Path B — "static but sustained": the bbox has been observed for
            long enough with no shrink, even if it isn't growing. This is
            the path that catches long / already-formed / dripped spills,
            which the previous growth-only gate silently dropped.

        Both paths reject an actively shrinking region up front — a real
        spill does not shrink.
        """
        history = pending.get("motion_history") or []
        initial_area = max(1, int(pending.get("initial_area", bbox_area)))
        previous_area = max(1, int(pending.get("previous_area", bbox_area)))

        history_areas = [int(entry["area"]) for entry in history]
        if history_areas:
            first_area = max(1, min(initial_area, history_areas[0]))
            last_area = max(bbox_area, history_areas[-1])
        else:
            first_area = initial_area
            last_area = bbox_area

        growth_ratio = (last_area - first_area) / first_area
        shrink_ratio = (
            (previous_area - bbox_area) / previous_area
            if bbox_area < previous_area
            else 0.0
        )

        # Hard reject for both paths: actively shrinking is never a real spill.
        if shrink_ratio > self.max_area_shrink_ratio:
            return False

        # Path A — actively growing.
        if growth_ratio >= self.min_area_growth_ratio:
            growth_steps_required = self.min_growth_steps
            if history_areas:
                growth_steps = sum(
                    1
                    for prev_area_value, curr_area_value in zip(
                        history_areas, history_areas[1:] + [bbox_area]
                    )
                    if curr_area_value > prev_area_value
                )
                if growth_steps >= growth_steps_required:
                    return True
            elif growth_steps_required <= 0:
                return True

        # Path B — sustained static presence. Uses a slightly stricter bar
        # than the base maturity gate so we don't confirm on short flickers:
        # a static region must persist ~2x longer than a growing one before
        # we trust it as a real spill.
        first_seen = float(pending.get("first_seen", current_time))
        stable_duration = max(0.0, current_time - first_seen)
        count = int(pending.get("count", 0))
        static_confirm_frames = max(
            self.min_consecutive_frames + 1,
            int(round(self.min_consecutive_frames * 1.5)),
        )
        static_confirm_seconds = max(self.min_stable_seconds * 2.0, 1.2)
        if count >= static_confirm_frames and stable_duration >= static_confirm_seconds:
            return True

        return False

    def _record_motion_history(
        self,
        pending: Dict[str, Any],
        current_time: float,
        center: tuple,
        bbox_area: int,
    ) -> None:
        history = pending.setdefault("motion_history", [])
        history.append(
            {
                "timestamp": current_time,
                "center": center,
                "area": int(bbox_area),
            }
        )
        if len(history) > self.motion_history_window:
            del history[: len(history) - self.motion_history_window]

    def _has_matured_candidate(
        self, pending: Dict[str, Any], current_time: float
    ) -> bool:
        first_seen = float(
            pending.get("first_seen", pending.get("last_seen", current_time))
        )
        stable_duration = max(0.0, current_time - first_seen)
        return (
            int(pending.get("count", 0)) >= self.min_consecutive_frames
            and stable_duration >= self.min_stable_seconds
        )

    def _compute_edge_motion_ratio(self, motion_mask: np.ndarray) -> float:
        mask_h, mask_w = motion_mask.shape[:2]
        if mask_h <= 2 or mask_w <= 2:
            total_pixels = motion_mask.size
            if total_pixels <= 0:
                return 0.0
            return cv2.countNonZero(motion_mask) / total_pixels

        band_w = max(1, int(mask_w * self.edge_band_ratio))
        band_h = max(1, int(mask_h * self.edge_band_ratio))
        edge_mask = np.ones((mask_h, mask_w), dtype=np.uint8) * 255

        inner_x1 = band_w
        inner_x2 = max(inner_x1, mask_w - band_w)
        inner_y1 = band_h
        inner_y2 = max(inner_y1, mask_h - band_h)
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            edge_mask[inner_y1:inner_y2, inner_x1:inner_x2] = 0

        edge_motion = cv2.bitwise_and(motion_mask, edge_mask)
        edge_pixels = cv2.countNonZero(edge_mask)
        if edge_pixels <= 0:
            return 0.0
        return cv2.countNonZero(edge_motion) / edge_pixels

    @staticmethod
    def _normalize_odd_kernel_size(raw_value: Any) -> int:
        kernel_size = max(1, int(raw_value))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

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
