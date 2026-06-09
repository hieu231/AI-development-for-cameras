"""
People Control Model - Detects people entering warehouse/restricted areas
Supports ROI (rectangle or polygon) with normalized coordinates
"""
from typing import Dict, Any, Optional, List, Tuple
import threading
import numpy as np
import cv2
import time
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, resolve_engine_path
from src.core.object_tracker import RecentViolationDeduplicator
from src.utils.alert_levels import AlertLevel
from src.utils.roi_utils import denormalize_roi


class PeopleControlModel(BaseModel):
    """Model for detecting people in restricted areas with ROI support"""

    def __init__(self, model_path: str = "src/ai_models/model_weights/yolo11m.pt", **kwargs):
        """
        Initialize People Control Model

        Args:
            model_path: Path to YOLO model weights
            **kwargs: Additional parameters
                - conf_threshold: Confidence threshold (default: 0.4)
                - detection_cooldown: Seconds between repeated alerts for same track in ROI (default: 300)
                - global_event_cooldown: Minimum seconds between any 2 alerts globally (default: 300)
                - bbox_growth_window_sec: Time window to evaluate bbox expansion (default: 2.0)
                - bbox_growth_ratio_threshold: Required area growth ratio in window (default: 1.15)
                - bbox_min_samples: Minimum samples in window before evaluating growth (default: 3)
                - track_lost_timeout_sec: Keep track alive when temporarily lost (default: 5.0)
                - max_roi_polygons: Maximum number of ROI polygons (default: 1, max: 5)
        """
        super().__init__(
            model_name="PeopleControlModel",
            default_alert_level=AlertLevel.HIGH,
            model_path=model_path,
            **kwargs
        )

        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        # Only move to device for PyTorch models, not TensorRT/ONNX
        if self.device and model_path.endswith('.pt'):
            self.model.to(self.device)

        # Only detect person class (ID 0 in COCO)
        self.target_class = 0
        self.conf_threshold = kwargs.get('conf_threshold', 0.4)

        # Detection parameters
        self.detection_cooldown = kwargs.get('detection_cooldown', 300)  # 5 minutes
        self.global_event_cooldown = kwargs.get('global_event_cooldown', 300)  # 5 minutes
        self.last_global_event_time = 0.0
        self.bbox_growth_window_sec = kwargs.get('bbox_growth_window_sec', 2.0)
        self.bbox_growth_ratio_threshold = kwargs.get('bbox_growth_ratio_threshold', 1.15)
        self.bbox_min_samples = kwargs.get('bbox_min_samples', 3)
        self.track_lost_timeout_sec = kwargs.get('track_lost_timeout_sec', 5.0)
        
        # Multi-ROI support
        self.max_roi_polygons = max(1, min(5, int(kwargs.get('max_roi_polygons', 1))))
        self._event_dedup_window_seconds = float(
            kwargs.get("event_dedup_window_seconds", self.track_lost_timeout_sec)
        )
        self._event_dedup_iou_threshold = float(
            kwargs.get("event_dedup_iou_threshold", 0.5)
        )
        self.recent_violation_deduplicator = RecentViolationDeduplicator(
            window_seconds=self._event_dedup_window_seconds,
            iou_threshold=self._event_dedup_iou_threshold,
        )

        # Guards track_status and related mutable state.
        self._state_lock = threading.Lock()

        # Tracking state
        self.track_status = {}
        # track_id -> {
        #   in_roi, roi_index, first_enter_time, last_event_time,
        #   pending_enter_event, bbox_area_history[(timestamp, area)]
        # }

        # Colors - palette for different ROIs
        self.roi_colors = [
            (0, 255, 0),    # ROI 0: Green
            (255, 165, 0),  # ROI 1: Orange
            (0, 165, 255),  # ROI 2: Red-Orange
            (255, 0, 255),  # ROI 3: Magenta
            (0, 255, 255),  # ROI 4: Cyan
        ]
        self.person_color = (255, 0, 0)
        self.detection_color = (0, 0, 255)

        self.logger.info(f"PeopleControlModel initialized on {self.device} (max_roi_polygons: {self.max_roi_polygons})")

    def _build_roi_polygons(
        self,
        normalized_roi: Any,
        frame_width: int,
        frame_height: int
    ) -> List[List[Tuple[int, int]]]:
        """
        Build list of ROI polygons from normalized ROI input
        
        Input can be:
        - Single polygon: [[x1,y1], [x2,y2], ...]
        - Polygon set: [[[x1,y1], [...]], [[...], [...]], ...]
        
        Returns:
            List of pixel-coordinate ROI polygons, max length = max_roi_polygons
        """
        roi_polygons = []
        
        if not normalized_roi:
            # Default: full frame
            return [denormalize_roi([[0, 0], [1, 0], [1, 1], [0, 1]], frame_width, frame_height)]
        
        # Check if it's a polygon set (list of polygons)
        if isinstance(normalized_roi, list) and len(normalized_roi) > 0:
            first_elem = normalized_roi[0]
            
            # Check if first element is a polygon (not a coordinate pair)
            if isinstance(first_elem, (list, tuple)) and len(first_elem) > 0:
                first_of_first = first_elem[0]
                
                # If first of first is a list/tuple, it's a polygon set
                if isinstance(first_of_first, (list, tuple)) and len(first_of_first) >= 2:
                    # Polygon set format: [polygon1, polygon2, ...]
                    for i, polygon in enumerate(normalized_roi):
                        if i >= self.max_roi_polygons:
                            break
                        if isinstance(polygon, (list, tuple)) and len(polygon) >= 3:
                            denorm = denormalize_roi(polygon, frame_width, frame_height)
                            if denorm:
                                roi_polygons.append(denorm)
                else:
                    # Single polygon format: [[x1,y1], [x2,y2], ...]
                    denorm = denormalize_roi(normalized_roi, frame_width, frame_height)
                    if denorm:
                        roi_polygons.append(denorm)
        
        # If no valid polygons extracted, use full frame as default
        if not roi_polygons:
            roi_polygons = [denormalize_roi([[0, 0], [1, 0], [1, 1], [0, 1]], frame_width, frame_height)]
        
        return roi_polygons
    
    def _find_matching_roi_index(
        self,
        point: Tuple[int, int],
        roi_polygons: List[List[Tuple[int, int]]]
    ) -> Optional[int]:
        """
        Find which ROI (if any) contains the point
        
        Returns:
            ROI index (0-based) or None if not in any ROI
        """
        for roi_idx, roi_polygon in enumerate(roi_polygons):
            if roi_polygon is not None:
                roi_poly_np = np.array(roi_polygon, dtype=np.int32)
                if cv2.pointPolygonTest(roi_poly_np, (float(point[0]), float(point[1])), False) >= 0:
                    return roi_idx
        return None

    def _is_point_in_roi(
        self,
        point: Tuple[int, int],
        roi_polygon: Optional[List[Tuple[int, int]]],
        roi_rect: Optional[Tuple[int, int, int, int]]
    ) -> bool:
        """Check if point is inside ROI (single polygon, deprecated in favor of _find_matching_roi_index)"""
        if roi_polygon is not None:
            roi_poly_np = np.array(roi_polygon, dtype=np.int32)
            return cv2.pointPolygonTest(roi_poly_np, (float(point[0]), float(point[1])), False) >= 0
        elif roi_rect is not None:
            x1, y1, x2, y2 = roi_rect
            return x1 <= point[0] <= x2 and y1 <= point[1] <= y2
        return False

    def _draw_rois(
        self,
        frame: np.ndarray,
        roi_polygons: List[List[Tuple[int, int]]]
    ) -> np.ndarray:
        """
        Draw multiple ROI polygons on frame with different colors
        
        Args:
            frame: Input frame
            roi_polygons: List of pixel-coordinate ROI polygons
            
        Returns:
            Annotated frame
        """
        overlay = frame.copy()
        
        for roi_idx, roi_polygon in enumerate(roi_polygons):
            if roi_polygon is None or len(roi_polygon) < 3:
                continue
                
            roi_color = self.roi_colors[roi_idx % len(self.roi_colors)]
            roi_poly_np = np.array(roi_polygon, dtype=np.int32)
            
            # Fill overlay
            cv2.fillPoly(overlay, [roi_poly_np], (roi_color[0] // 4, roi_color[1] // 4, roi_color[2] // 4))
            
            # Draw border
            cv2.polylines(frame, [roi_poly_np.reshape((-1, 1, 2))], True, roi_color, 2)
            
            # Add label
            label = f'ROI {roi_idx + 1}'
            cv2.putText(frame, label, (roi_poly_np[0][0] + 5, roi_poly_np[0][1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2, cv2.LINE_AA)
        
        # Blend overlay with frame
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        return frame

    def _draw_roi(
        self,
        frame: np.ndarray,
        roi_polygon: Optional[List[Tuple[int, int]]],
        roi_rect: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """Draw single ROI on frame (deprecated, use _draw_rois for multiple ROIs)"""
        overlay = frame.copy()

        if roi_polygon is not None:
            roi_poly_np = np.array(roi_polygon, dtype=np.int32)
            cv2.fillPoly(overlay, [roi_poly_np], (0, 50, 0))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.polylines(frame, [roi_poly_np.reshape((-1, 1, 2))], True, self.roi_colors[0], 2)
            cv2.putText(frame, 'KHU VUC KHO', (roi_poly_np[0][0] + 5, roi_poly_np[0][1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.roi_colors[0], 2, cv2.LINE_AA)
        elif roi_rect is not None:
            x1, y1, x2, y2 = roi_rect
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 50, 0), -1)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.roi_colors[0], 2)
            cv2.putText(frame, 'KHU VUC KHO', (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.roi_colors[0], 2)

        return frame

    def _is_bbox_expansion_confirmed(self, track_id: Any, current_time: float) -> bool:
        """Check whether bbox area has expanded enough in configured time window."""
        status = self.track_status.get(track_id)
        if status is None:
            return False

        history: List[Tuple[float, float]] = status.get('bbox_area_history', [])
        cutoff_time = current_time - self.bbox_growth_window_sec
        filtered_history = [(ts, area) for ts, area in history if ts >= cutoff_time]
        status['bbox_area_history'] = filtered_history

        if len(filtered_history) < self.bbox_min_samples:
            return False

        areas = [area for _, area in filtered_history if area > 0]
        if len(areas) < self.bbox_min_samples:
            return False

        min_area = min(areas)
        max_area = max(areas)
        if min_area <= 0:
            return False

        growth_ratio = max_area / min_area
        return growth_ratio >= self.bbox_growth_ratio_threshold

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        """
        Process frame and detect people in ROI(s)

        Args:
            frame: Input frame (BGR)
            **kwargs: Additional parameters
                - roi: Normalized ROI - can be single polygon [[x1,y1], [x2,y2], ...]
                        or polygon set [polygon1, polygon2, ...] (values in [0,1])
                - max_roi_polygons: Max ROI polygons to use (1-5, overrides init value)
                - annotate: Whether to annotate frame (default: True)

        Returns:
            DetectionResult with standardized format
        """
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def _get_recent_violation_deduplicator(self) -> RecentViolationDeduplicator:
        deduplicator = getattr(self, "recent_violation_deduplicator", None)
        if deduplicator is None:
            deduplicator = RecentViolationDeduplicator(
                window_seconds=float(
                    getattr(self, "_event_dedup_window_seconds", self.track_lost_timeout_sec)
                ),
                iou_threshold=float(
                    getattr(self, "_event_dedup_iou_threshold", 0.5)
                ),
            )
            self.recent_violation_deduplicator = deduplicator
        return deduplicator

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        recent_violation_deduplicator = self._get_recent_violation_deduplicator()
        current_time = time.time()
        annotate = kwargs.get('annotate', True)

        frame_height, frame_width = frame.shape[:2]

        # Update max_roi_polygons if provided
        if 'max_roi_polygons' in kwargs:
            self.max_roi_polygons = max(1, min(5, int(kwargs.get('max_roi_polygons', 1))))

        # Get ROI input (can be single polygon or polygon set)
        normalized_roi = kwargs.get('roi', [[0, 0], [1, 0], [1, 1], [0, 1]])

        # Build list of ROI polygons (pixel coordinates)
        roi_polygons = self._build_roi_polygons(normalized_roi, frame_width, frame_height)

        # Keep a clean copy for YOLO inference (without annotations from previous models)
        clean_frame = frame
        annotated_frame = frame.copy() if annotate else None

        # Draw ROIs
        if annotate and annotated_frame is not None:
            annotated_frame = self._draw_rois(annotated_frame, roi_polygons)

        # Run tracking on CLEAN frame (no previous annotations)
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            persist=True,
            conf=self.conf_threshold,
            classes=[self.target_class],
            verbose=False,
        )

        detection_result = None
        event_triggered = False
        current_track_ids = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            current_track_ids = set(track_ids)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Synthetic track_id fallback khi YOLO tracker không gán được ID
                # (thường xảy ra với VSS HTTP-FLV streams)
                if track_id is None:
                    gx, gy = center_x // 50, center_y // 50
                    track_id = (hash(("person", gx, gy)) & 0x7FFFFFFF)
                    current_track_ids.add(track_id)

                # Find which ROI (if any) the person is in
                matching_roi_index = self._find_matching_roi_index((center_x, center_y), roi_polygons)

                # Initialize tracking state
                if track_id not in self.track_status:
                    self.track_status[track_id] = {
                        'in_roi': False,
                        'roi_index': None,
                        'first_enter_time': None,
                        'last_event_time': 0.0,
                        'pending_enter_event': False,
                        'bbox_area_history': [],
                        'last_seen_time': 0.0
                    }

                status = self.track_status[track_id]
                status['last_seen_time'] = current_time

                # Process detection
                if matching_roi_index is not None:
                    # Person is in a ROI
                    is_first_enter = not status['in_roi'] or status['roi_index'] != matching_roi_index

                    if is_first_enter:
                        status['in_roi'] = True
                        status['roi_index'] = matching_roi_index
                        status['first_enter_time'] = current_time
                        status['pending_enter_event'] = True
                        status['bbox_area_history'] = []

                    status['bbox_area_history'].append((current_time, float(box_area)))

                    # Per-track cooldown only. A person just entering the ROI
                    # always fires (pending_enter_event), then the same track
                    # is re-suppressed until detection_cooldown has passed.
                    # No global cooldown: independent people entering
                    # different ROIs all fire independently.
                    time_since_track_event = current_time - status['last_event_time']
                    should_trigger = (
                        status['pending_enter_event']
                        or time_since_track_event >= self.detection_cooldown
                    )

                    if should_trigger:
                        violation_key = f"PEOPLE_CONTROL:ROI:{matching_roi_index}"
                        if recent_violation_deduplicator.is_recent_duplicate(
                            violation_key,
                            track_id,
                            [x1, y1, x2, y2],
                            current_time=current_time,
                        ):
                            status['last_event_time'] = current_time
                            status['pending_enter_event'] = False
                            if annotate and annotated_frame is not None:
                                cv2.rectangle(
                                    annotated_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    self.person_color,
                                    2,
                                )
                            continue

                        # Record event
                        status['last_event_time'] = current_time
                        status['pending_enter_event'] = False
                        self.last_global_event_time = current_time
                        event_triggered = True
                        recent_violation_deduplicator.remember_event(
                            violation_key,
                            track_id,
                            [x1, y1, x2, y2],
                            current_time=current_time,
                        )

                        detection_result = {
                            'track_id': track_id,
                            'roi_index': matching_roi_index,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y]
                        }

                        reason = "first_enter" if is_first_enter else "still_in_roi_after_cooldown"
                        self.logger.info(
                            f"Phát hiện người vào ROI {matching_roi_index + 1} (ID: {track_id}, conf: {conf:.2f}, "
                            f"Lý do: {reason})"
                        )

                        # Draw in red when detected
                        if annotate and annotated_frame is not None:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.detection_color, 3)
                    else:
                        # Draw normal bounding box
                        if annotate and annotated_frame is not None:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.person_color, 2)
                else:
                    # Person left all ROIs
                    if status['in_roi']:
                        status['in_roi'] = False
                        status['roi_index'] = None
                        status['first_enter_time'] = None
                        status['pending_enter_event'] = False
                        status['bbox_area_history'] = []

                    if annotate and annotated_frame is not None:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.person_color, 2)

        # Cleanup old tracking IDs (with timeout) even when current frame has no detections
        old_track_ids = set(self.track_status.keys())
        for old_id in old_track_ids - current_track_ids:
            old_status = self.track_status.get(old_id)
            if old_status is None:
                continue

            last_seen_time = old_status.get('last_seen_time', 0.0)
            if (current_time - last_seen_time) >= self.track_lost_timeout_sec:
                if old_status.get('in_roi'):
                    roi_idx = old_status.get('roi_index', 'unknown')
                    self.logger.info(
                        f"Track {old_id} rời ROI {roi_idx} hoặc mất dấu quá timeout ({self.track_lost_timeout_sec}s)"
                    )
                del self.track_status[old_id]

        # Build detections
        all_detections: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Check which ROI (if any) contains person
                roi_index = self._find_matching_roi_index((center_x, center_y), roi_polygons)
                
                if roi_index is not None:
                    detection_info = {
                        "class_id": 0,  # Person class = 0 in COCO
                        "class_name": "Person",
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                        "roi_index": roi_index
                    }
                    all_detections.append(detection_info)
        
        # Build violations from detection_result
        if detection_result:
            violations.append({
                "track_id": detection_result.get('track_id'),
                "roi_index": detection_result.get('roi_index'),
                "violation_type": "Người vào khu vực cấm",
                "confidence": detection_result.get('confidence', 0.0),
                "bbox": detection_result.get('bbox', []),
            })
        
        event_type = "Xâm nhập trái phép khu vực"
        for violation in violations:
            violation["violation_type"] = event_type
            violation["event_type"] = event_type
            violation["description"] = "Phát hiện xâm nhập trái phép khu vực trong khu vực giám sát"

        primary_event = violations[0] if violations else None
        
        # Metadata
        metadata: Dict[str, Any] = {
            "type": "Người vào khu vực cấm",
            "eventType": "Người vào khu vực cấm",
            "severity": "high" if event_triggered else "low",
            "description": f"Phát hiện người vào khu vực cấm (sử dụng {len(roi_polygons)} ROI)",
            "detections": all_detections,
            "violations": violations,
            "count": len(all_detections),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "people_control",
            "roi_count": len(roi_polygons),
            "max_roi_polygons": self.max_roi_polygons,
        }
        
        metadata["type"] = event_type
        metadata["eventType"] = event_type
        metadata["title"] = event_type
        metadata["description"] = f"Phát hiện xâm nhập trái phép khu vực (sử dụng {len(roi_polygons)} ROI)"

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]
            metadata["roi_index"] = primary_event.get("roi_index")

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata
        )

