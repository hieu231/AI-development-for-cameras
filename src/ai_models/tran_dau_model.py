"""
Rapid-response sudden-flow spill model for `tran_dau.pt`.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, DetectedObject, resolve_engine_path
from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator
from src.utils.alert_levels import AlertLevel
from src.utils.roi_utils import (
    build_roi_poly_arrays,
    draw_roi_overlays,
    is_point_in_any_roi,
)


class TranDauModel(BaseModel):
    """Detect sudden oil spill flow that requires immediate response."""

    FALLBACK_GRID_SIZE = 40
    FOREGROUND_CLASS_IDS = [0, 1, 2, 3, 5, 7]

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/tran_dau.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.3,
        **kwargs: Any,
    ):
        super().__init__(
            model_name="TranDauModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **kwargs,
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.model.to(self.device)

        self.classes_to_keep: Optional[List[int]] = None
        raw_classes = kwargs.get("classes")
        if raw_classes is not None and isinstance(raw_classes, list):
            self.classes_to_keep = [int(class_id) for class_id in raw_classes]
        else:
            model_names = getattr(self.model, "names", None)
            if isinstance(model_names, dict) and len(model_names) == 1:
                only_id = next(iter(model_names.keys()))
                try:
                    self.classes_to_keep = [int(only_id)]
                except Exception:
                    self.classes_to_keep = None

        standardized_spill_label = "Phát hiện tràn dầu"
        legacy_spill_labels = {
            "",
            "Dòng chảy tràn dầu đột ngột",
            "dong chay tran dau dot ngot",
        }
        configured_spill_label = str(kwargs.get("spill_label", standardized_spill_label))
        self.spill_label = (
            standardized_spill_label
            if configured_spill_label in legacy_spill_labels
            else configured_spill_label
        )
        configured_metadata_type = str(
            kwargs.get("metadata_type", standardized_spill_label)
        )
        self.metadata_type = (
            standardized_spill_label
            if configured_metadata_type in legacy_spill_labels
            else configured_metadata_type
        )
        self.metadata_model_type = str(kwargs.get("metadata_model_type", "tran_dau"))
        self.overlay_label = str(
            kwargs.get("overlay_label", "DONG CHAY DAU DOT NGOT")
        )
        self.alert_message = str(
            kwargs.get(
                "alert_message",
                "Phát hiện tràn dầu, yêu cầu cảnh báo nhanh và xử lý tức thời",
            )
        )
        self.monitor_message = str(
            kwargs.get("monitor_message", "Giám sát nguy cơ tràn dầu")
        )

        self.aux_model_path = resolve_engine_path(kwargs.get(
            "aux_model_path",
            kwargs.get("person_model_path", "src/ai_models/model_weights/yolo11n.pt"),
        ), runtime_device=self.device)
        self.aux_model = YOLO(self.aux_model_path)
        if self.aux_model_path.endswith(".pt") and self.device in ["cuda", "mps"]:
            self.aux_model.to(self.device)

        self.conf_threshold = float(kwargs.get("conf_threshold", confidence_threshold))
        self.iou_threshold = float(kwargs.get("iou_threshold", iou_threshold))
        self.detection_cooldown = int(kwargs.get("detection_cooldown", 60))
        self.foreground_conf_threshold = float(
            kwargs.get("foreground_conf_threshold", kwargs.get("person_conf_threshold", 0.35))
        )
        self.foreground_iou_threshold = float(
            kwargs.get("foreground_iou_threshold", kwargs.get("person_iou_threshold", 0.35))
        )
        self.foreground_cover_ratio_threshold = float(
            kwargs.get(
                "foreground_cover_ratio_threshold",
                kwargs.get("person_cover_ratio_threshold", 0.55),
            )
        )
        self.foreground_proximity_ratio = float(
            kwargs.get("foreground_proximity_ratio", 0.18)
        )
        self.recent_foreground_hold_seconds = float(
            kwargs.get("recent_foreground_hold_seconds", 0.5)
        )

        self.min_bbox_area_ratio = float(kwargs.get("min_bbox_area_ratio", 0.002))
        self.max_bbox_height_ratio = float(kwargs.get("max_bbox_height_ratio", 0.28))
        self.min_aspect_ratio = float(kwargs.get("min_aspect_ratio", 0.55))
        self.min_consecutive_frames = max(1, int(kwargs.get("min_consecutive_frames", 1)))
        self.motion_diff_threshold = int(kwargs.get("motion_diff_threshold", 20))
        self.motion_blur_kernel = self._normalize_odd_kernel_size(
            kwargs.get("motion_blur_kernel", 5)
        )
        self.motion_morph_kernel = self._normalize_odd_kernel_size(
            kwargs.get("motion_morph_kernel", 5)
        )
        self.edge_band_ratio = float(kwargs.get("edge_band_ratio", 0.2))
        self._stale_timeout = float(kwargs.get("stale_timeout", 3.0))

        self.high_severity_area_ratio = float(
            kwargs.get("high_severity_area_ratio", 0.008)
        )
        self.critical_severity_area_ratio = float(
            kwargs.get("critical_severity_area_ratio", 0.02)
        )
        if self.critical_severity_area_ratio < self.high_severity_area_ratio:
            self.critical_severity_area_ratio = self.high_severity_area_ratio

        self.global_event_cooldown = float(kwargs.get("global_event_cooldown", 3.0))
        self.continuous_event_interval = float(
            kwargs.get("continuous_event_interval", 0.5)
        )
        self.track_ongoing_incidents = bool(
            kwargs.get("track_ongoing_incidents", True)
        )
        self.debug_detection_summary = bool(
            kwargs.get("debug_detection_summary", True)
        )
        self.debug_summary_interval_seconds = float(
            kwargs.get("debug_summary_interval_seconds", 5.0)
        )
        self.incident_hold_seconds = float(
            kwargs.get("incident_hold_seconds", 10.0)
        )
        self.flow_motion_ratio_threshold = float(
            kwargs.get("flow_motion_ratio_threshold", 0.04)
        )
        self.flow_edge_motion_ratio_threshold = float(
            kwargs.get("flow_edge_motion_ratio_threshold", 0.06)
        )
        self.sudden_area_growth_ratio = float(
            kwargs.get("sudden_area_growth_ratio", 0.18)
        )
        self.min_flow_frames = max(1, int(kwargs.get("min_flow_frames", 1)))
        self.max_incident_motion_ratio = float(
            kwargs.get("max_incident_motion_ratio", 0.85)
        )
        self.max_incident_edge_motion_ratio = float(
            kwargs.get("max_incident_edge_motion_ratio", 0.95)
        )

        self._pending_detections: Dict[int, Dict[str, Any]] = {}
        self._recent_foreground_boxes: List[Dict[str, Any]] = []
        self._active_incidents: Dict[int, Dict[str, Any]] = {}
        self._prev_gray_frame = None
        self.last_global_detection = 0.0
        self.object_tracker = ObjectTracker(reset_interval=self.detection_cooldown)
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
        self._last_debug_summary_at = 0.0
        self._debug_stats_accumulator: Dict[str, int] = {}

        # Thread safety: process_frame may be called from different threads
        # (camera processor thread vs. API/reload thread). Guard all mutable
        # per-instance state with a reentrant lock.
        self._state_lock = threading.RLock()

        # Fallback detection-key counter. Lives in a negative namespace so
        # synthetic keys never collide with real (positive) ByteTrack IDs.
        self._next_fallback_idx = 0
        self._fallback_iou_threshold = float(kwargs.get("fallback_iou_threshold", 0.3))

        self.logger.info(
            "TranDauModel loaded | Device: %s | Conf: %.2f | Flow motion >= %.2f | "
            "Edge motion >= %.2f | Growth >= %.2f | Cooldown: %ss",
            self.device,
            self.conf_threshold,
            self.flow_motion_ratio_threshold,
            self.flow_edge_motion_ratio_threshold,
            self.sudden_area_growth_ratio,
            self.detection_cooldown,
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
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
        frame_stats: Dict[str, int] = {
            "boxes_total": 0,
            "reject_invalid_bbox": 0,
            "reject_outside_roi": 0,
            "reject_fg_overlap": 0,
            "reject_fg_recent": 0,
            "reject_small_area": 0,
            "reject_tall_bbox": 0,
            "reject_bad_aspect": 0,
            "reject_high_motion_noise": 0,
            "reject_not_confirmed_or_not_ongoing": 0,
            "accepted_detections": 0,
            "events_emitted": 0,
        }

        h, w = frame.shape[:2]
        frame_area = max(1, w * h)
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

        track_kwargs: Dict[str, Any] = {
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "persist": True,
            "verbose": False,
        }
        if self.classes_to_keep:
            track_kwargs["classes"] = self.classes_to_keep

        results = self._run_yolo_track(self.model, frame, **track_kwargs)

        annotated_frame = frame.copy() if annotate else None
        detections: List[DetectedObject] = []
        new_violations: List[Dict[str, Any]] = []
        max_confidence = 0.0
        highest_severity = "none"
        claimed_detection_keys: Set[int] = set()

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
                frame_stats["boxes_total"] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    frame_stats["reject_invalid_bbox"] += 1
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if not is_point_in_any_roi(center, roi_polys):
                    frame_stats["reject_outside_roi"] += 1
                    continue

                spill_bbox = (x1, y1, x2, y2)
                conf = float(box.conf[0])
                max_confidence = max(max_confidence, conf)
                cls_id = int(box.cls[0])

                detection_key = self._resolve_detection_key(
                    track_id, spill_bbox, claimed_detection_keys
                )
                claimed_detection_keys.add(detection_key)
                effective_track_id = track_id if track_id is not None else detection_key

                detections.append(
                    DetectedObject(
                        label=self.spill_label,
                        confidence=conf,
                        bbox=spill_bbox,
                        extra={
                            "track_id": effective_track_id,
                            "class_id": cls_id,
                            "severity": "high",
                            "incident_state": "sudden_flow",
                            "area_ratio": round((bbox_w * bbox_h) / frame_area, 4),
                            "duration_seconds": 0.0,
                            "response_required": True,
                        },
                    )
                )
                frame_stats["accepted_detections"] += 1

                is_recent_duplicate = recent_violation_deduplicator.is_recent_duplicate(
                    "SUDDEN_OIL_FLOW",
                    effective_track_id,
                    spill_bbox,
                    current_time=current_time,
                )
                if is_recent_duplicate:
                    self.logger.debug(
                        "Suppressed recent duplicate tran dau: ID=%s",
                        effective_track_id,
                    )

                should_emit_event = False
                if not is_recent_duplicate:
                    should_emit_event = self.object_tracker.should_record_event(
                        effective_track_id,
                        "SUDDEN_OIL_FLOW",
                        repeat_interval=self.continuous_event_interval,
                    )
                event_gate_interval = self.global_event_cooldown
                if self.continuous_event_interval > 0:
                    event_gate_interval = min(
                        self.global_event_cooldown,
                        self.continuous_event_interval,
                    )
                if (
                    should_emit_event
                    and current_time - self.last_global_detection
                    >= event_gate_interval
                ):
                    recent_violation_deduplicator.remember_event(
                        "SUDDEN_OIL_FLOW",
                        effective_track_id,
                        spill_bbox,
                        current_time=current_time,
                    )
                    new_violations.append(
                        {
                            "track_id": effective_track_id,
                            "type": self.spill_label,
                            "confidence": conf,
                            "bbox": spill_bbox,
                            "event_kind": "SUDDEN_OIL_FLOW_DETECTED",
                            "severity": "high",
                            "flow_detected": True,
                            "motion_ratio": 0.0,
                            "edge_motion_ratio": 0.0,
                            "area_growth_ratio": 0.0,
                            "total_growth_ratio": 0.0,
                            "response_required": True,
                            "response_mode": "immediate",
                        }
                    )
                    self.logger.warning(
                        "SUDDEN OIL FLOW DETECTED | ID: %s | Severity: %s | "
                        "Conf: %.2f | Motion: %.3f | Growth: %.3f",
                        effective_track_id,
                        "HIGH",
                        conf,
                        0.0,
                        0.0,
                    )
                    self.last_global_detection = current_time
                    frame_stats["events_emitted"] += 1

                if annotate and annotated_frame is not None:
                    color = (0, 140, 255)
                    thickness = 3
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    label = f"{self.overlay_label} [HIGH] {conf:.2f}"
                    if track_id is not None:
                        label += f" ID:{track_id}"

                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
                    )
                    cv2.rectangle(
                        annotated_frame, (x1, y1 - th - 12), (x1 + tw, y1), color, -1
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                    )

        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys)

        self._prev_gray_frame = gray_frame
        self._maybe_log_debug_summary(
            current_time=current_time,
            frame_stats=frame_stats,
            active_incident_count=len(detections),
            violation_count=len(new_violations),
        )

        detection_dicts: List[Dict[str, Any]] = []
        for detection in detections:
            extra = detection.extra if isinstance(detection.extra, dict) else {}
            detection_dicts.append(
                {
                    "class_id": extra.get("class_id"),
                    "class_name": detection.label,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                    "track_id": extra.get("track_id"),
                    "severity": extra.get("severity", "high"),
                    "incident_state": extra.get("incident_state", "sudden_flow"),
                    "area_ratio": extra.get("area_ratio", 0.0),
                    "duration_seconds": extra.get("duration_seconds", 0.0),
                    "flow_detected": extra.get("flow_detected", True),
                    "flow_confirmed": extra.get("flow_confirmed", True),
                    "motion_ratio": extra.get("motion_ratio", 0.0),
                    "edge_motion_ratio": extra.get("edge_motion_ratio", 0.0),
                    "area_growth_ratio": extra.get("area_growth_ratio", 0.0),
                    "total_growth_ratio": extra.get("total_growth_ratio", 0.0),
                    "response_required": extra.get("response_required", True),
                }
            )

        violations = [
            {
                "track_id": violation["track_id"],
                "violation_type": violation["type"],
                "confidence": violation["confidence"],
                "bbox": violation["bbox"],
                "event_kind": violation["event_kind"],
                "severity": violation["severity"],
                "flow_detected": violation["flow_detected"],
                "motion_ratio": violation["motion_ratio"],
                "edge_motion_ratio": violation["edge_motion_ratio"],
                "area_growth_ratio": violation["area_growth_ratio"],
                "total_growth_ratio": violation["total_growth_ratio"],
                "response_required": violation["response_required"],
                "response_mode": violation["response_mode"],
            }
            for violation in new_violations
        ]

        event_triggered = bool(violations)
        requires_immediate_response = len(detection_dicts) > 0
        metadata: Dict[str, Any] = {
            "type": self.metadata_type,
            "eventType": self.metadata_type,
            "title": self.metadata_type,
            "description": self.alert_message if requires_immediate_response else self.monitor_message,
            "detections": detection_dicts,
            "violations": violations,
            "count": len(detection_dicts),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": self.metadata_model_type,
            "incident_type": "sudden_oil_flow",
            "severity": "high" if event_triggered else "low",
            "response_mode": "immediate",
            "requires_immediate_response": requires_immediate_response,
            "active_incidents": detection_dicts if requires_immediate_response else [],
            "event_message": self.alert_message if requires_immediate_response else self.monitor_message,
            "alert": {
                "level": AlertLevel.HIGH.value,
                "message": self.alert_message if requires_immediate_response else self.monitor_message,
                "confidence": max_confidence,
                "detected_objects": detection_dicts,
            },
        }

        if violations:
            primary_event = violations[0]
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]
            metadata["event_kind"] = primary_event["event_kind"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )

    def _resolve_detection_key(
        self,
        track_id: Optional[int],
        bbox: tuple,
        claimed_detection_keys: Set[int],
    ) -> int:
        """Map a raw detection to a stable pending-slot key.

        When ByteTrack gives a track_id we trust it directly. When tracking
        fails, we match against existing fallback slots by bbox IoU so that a
        detection drifting across a grid boundary keeps the same key (the old
        grid-hash approach silently reset the confirmation counter every time
        the centroid crossed a 40 px cell edge, killing recall near boundaries).

        Fallback keys live in a negative namespace so they can never collide
        with real (positive) ByteTrack IDs, even across long sessions.
        """
        if track_id is not None:
            return int(track_id)

        best_key: Optional[int] = None
        best_iou = 0.0
        for key, pending in self._pending_detections.items():
            if key >= 0:
                continue
            if key in claimed_detection_keys:
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

    def _maybe_log_debug_summary(
        self,
        current_time: float,
        frame_stats: Dict[str, int],
        active_incident_count: int,
        violation_count: int,
    ) -> None:
        if not self.debug_detection_summary:
            return

        for key, value in frame_stats.items():
            self._debug_stats_accumulator[key] = self._debug_stats_accumulator.get(key, 0) + int(value)

        self._debug_stats_accumulator["frames"] = self._debug_stats_accumulator.get("frames", 0) + 1
        self._debug_stats_accumulator["last_active_incidents"] = int(active_incident_count)
        self._debug_stats_accumulator["last_violations"] = int(violation_count)

        interval = max(0.5, float(self.debug_summary_interval_seconds))
        if current_time - self._last_debug_summary_at < interval:
            return

        self._last_debug_summary_at = current_time
        stats = self._debug_stats_accumulator
        self._debug_stats_accumulator = {}
        self.logger.info(
            "TranDau debug summary | frames=%s boxes=%s accept=%s events=%s "
            "rej[roi=%s fg_overlap=%s fg_recent=%s area=%s h=%s ar=%s motion=%s gate=%s] "
            "last_active=%s last_violations=%s",
            stats.get("frames", 0),
            stats.get("boxes_total", 0),
            stats.get("accepted_detections", 0),
            stats.get("events_emitted", 0),
            stats.get("reject_outside_roi", 0),
            stats.get("reject_fg_overlap", 0),
            stats.get("reject_fg_recent", 0),
            stats.get("reject_small_area", 0),
            stats.get("reject_tall_bbox", 0),
            stats.get("reject_bad_aspect", 0),
            stats.get("reject_high_motion_noise", 0),
            stats.get("reject_not_confirmed_or_not_ongoing", 0),
            stats.get("last_active_incidents", 0),
            stats.get("last_violations", 0),
        )

    def _classify_incident_severity(
        self,
        area_ratio: float,
        total_growth_ratio: float,
    ) -> str:
        if (
            area_ratio >= self.critical_severity_area_ratio
            or total_growth_ratio >= max(self.sudden_area_growth_ratio * 2, 0.4)
        ):
            return "critical"
        return "high"

    @staticmethod
    def _merge_severity(current: str, incoming: str) -> str:
        rank = {"none": 0, "high": 1, "critical": 2}
        return incoming if rank.get(incoming, 0) > rank.get(current, 0) else current

    def _remember_active_incident(
        self,
        track_id: int,
        bbox: tuple,
        confidence: float,
        area_px: int,
        area_ratio: float,
        severity: str,
        current_time: float,
        motion_ratio: float,
        edge_motion_ratio: float,
        area_growth_ratio: float,
        total_growth_ratio: float,
    ) -> Dict[str, Any]:
        incident = self._active_incidents.get(track_id)
        if incident is None:
            incident = {
                "track_id": track_id,
                "first_seen": current_time,
            }
            self._active_incidents[track_id] = incident

        incident["last_seen"] = current_time
        incident["bbox"] = bbox
        incident["confidence"] = confidence
        incident["area_px"] = area_px
        incident["area_ratio"] = area_ratio
        incident["severity"] = severity
        incident["motion_ratio"] = motion_ratio
        incident["edge_motion_ratio"] = edge_motion_ratio
        incident["area_growth_ratio"] = area_growth_ratio
        incident["total_growth_ratio"] = total_growth_ratio
        return incident

    def _cleanup_stale_incidents(self, current_time: float) -> None:
        stale_ids = [
            track_id
            for track_id, incident in self._active_incidents.items()
            if current_time - incident.get("last_seen", current_time)
            > self.incident_hold_seconds
        ]
        for track_id in stale_ids:
            del self._active_incidents[track_id]

    def _cleanup_stale_pending(self, current_time: float) -> None:
        stale_keys = [
            key
            for key, info in self._pending_detections.items()
            if current_time - info["last_seen"] > self._stale_timeout
        ]
        for key in stale_keys:
            del self._pending_detections[key]

    def _remember_foreground_boxes(
        self,
        foreground_boxes: List[tuple],
        current_time: float,
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
        self,
        frame: np.ndarray,
        roi_polys: List[np.ndarray],
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
        self,
        spill_bbox: tuple,
        foreground_boxes: List[tuple],
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

            if self._bbox_iou(spill_bbox, foreground_bbox) >= self.foreground_iou_threshold:
                return True

        return False

    def _has_recent_foreground_activity(
        self,
        spill_bbox: tuple,
        current_time: float,
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

    def _compute_motion_metrics(
        self,
        bbox: tuple,
        gray_frame: np.ndarray,
    ) -> Tuple[float, float]:
        if self._prev_gray_frame is None:
            return 0.0, 0.0

        x1, y1, x2, y2 = bbox
        prev_crop = self._prev_gray_frame[y1:y2, x1:x2]
        curr_crop = gray_frame[y1:y2, x1:x2]
        if prev_crop.size == 0 or curr_crop.size == 0 or prev_crop.shape != curr_crop.shape:
            return 0.0, 0.0

        diff = cv2.absdiff(prev_crop, curr_crop)
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
        return changed_pixels / total_pixels, self._compute_edge_motion_ratio(motion_mask)

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
