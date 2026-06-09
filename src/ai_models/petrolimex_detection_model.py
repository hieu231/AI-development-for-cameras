"""
src/ai_models/petrolimex_detection_model.py
Detect vest / no-vest violations using Petrolimex PPE model
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, resolve_engine_path
from src.core.object_tracker import ObjectTracker
from src.utils.roi_utils import (
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)
from src.utils.alert_levels import AlertLevel


class PetrolimexDetectionModel(BaseModel):
    """Model detect PPE violations: novest and no_hardhat."""

    VIOLATION_DISPLAY_MAPPING = {
        "novest": "Không mặc áo bảo hộ",
        "no_hardhat": "Không đội mũ bảo hộ",
    }

    EVENT_TYPE = "Không tuân thủ bảo hộ lao động"
    EVENT_DESCRIPTION = "Phát hiện vi phạm bảo hộ lao động trong khu vực giám sát"

    LABEL_ALIASES = {
        "vest": "vest",
        "novest": "novest",
        "no_vest": "novest",
        "no-vest": "novest",
        "without_vest": "novest",
        "hardhat": "hardhat",
        "hard_hat": "hardhat",
        "helmet": "hardhat",
        "no_hardhat": "no_hardhat",
        "no-hardhat": "no_hardhat",
        "no hardhat": "no_hardhat",
        "without_hardhat": "no_hardhat",
        "without_helmet": "no_hardhat",
        "nohelmet": "no_hardhat",
        "no_helmet": "no_hardhat",
        "person": "person",
        "people": "person",
    }

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/petrolimex.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        person_conf_threshold: float = 0.5,
        detection_cooldown: int = 300,
        **kwargs,
    ):
        super().__init__(
            model_name="PetrolimexDetectionModel",
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
        self.novest_conf_threshold = float(
            kwargs.get("novest_conf_threshold", 0.35)
        )
        self.person_conf_threshold = kwargs.get(
            "person_conf_threshold", person_conf_threshold
        )
        self.no_hardhat_model_path = resolve_engine_path(kwargs.get(
            "no_hardhat_model_path", "src/ai_models/model_weights/atld_92.pt"
        ), runtime_device=self.device)
        self.no_hardhat_conf_threshold = float(
            kwargs.get("no_hardhat_conf_threshold", self.person_conf_threshold)
        )
        self.enable_people_aux_detection = bool(
            kwargs.get("enable_people_aux_detection", True)
        )
        self.people_model_path = resolve_engine_path(kwargs.get(
            "people_model_path", "src/ai_models/model_weights/yolo11m.pt"
        ), runtime_device=self.device)
        self.people_conf_threshold = float(
            kwargs.get("people_conf_threshold", self.person_conf_threshold)
        )
        self.people_match_iou_threshold = float(
            kwargs.get("people_match_iou_threshold", 0.05)
        )
        # Fraction of person bbox height (from top) considered "head region".
        # A no_hardhat detection whose centre falls outside this region is a
        # floating false positive and is discarded.
        self._head_region_fraction = float(kwargs.get("head_region_fraction", 0.5))

        self.no_hardhat_model = YOLO(self.no_hardhat_model_path)
        if self.no_hardhat_model_path.endswith(".pt") and self.device in [
            "cuda",
            "mps",
        ]:
            self.no_hardhat_model.to(self.device)

        self.people_model = None
        if self.enable_people_aux_detection:
            self.people_model = YOLO(self.people_model_path)
            if self.people_model_path.endswith(".pt") and self.device in [
                "cuda",
                "mps",
            ]:
                self.people_model.to(self.device)

        self.class_names: Dict[int, str] = {}
        model_names = getattr(self.model, "names", {})
        if isinstance(model_names, dict):
            for class_id, name in model_names.items():
                normalized = self._normalize_label(str(name))
                if normalized in {"vest", "novest"}:
                    self.class_names[int(class_id)] = normalized

        self.no_hardhat_class_names: Dict[int, str] = {}
        self.no_hardhat_classes_to_keep: List[int] = []
        self._hardhat_model_person_class_ids: List[int] = []
        hardhat_model_names = getattr(self.no_hardhat_model, "names", {})
        if isinstance(hardhat_model_names, dict):
            for class_id, name in hardhat_model_names.items():
                normalized = self._normalize_label(str(name))
                if normalized == "no_hardhat":
                    self.no_hardhat_class_names[int(class_id)] = normalized
                elif str(name).strip().lower() == "person":
                    self._hardhat_model_person_class_ids.append(int(class_id))
        self.no_hardhat_classes_to_keep = list(self.no_hardhat_class_names.keys())

        self.people_class_names: Dict[int, str] = {}
        self._people_model_person_class_ids: List[int] = []
        if self.people_model is not None:
            people_model_names = getattr(self.people_model, "names", {})
            if isinstance(people_model_names, dict):
                for class_id, name in people_model_names.items():
                    normalized = self._normalize_label(str(name))
                    raw_name = str(name).strip().lower()
                    if normalized == "person" or raw_name == "person":
                        self.people_class_names[int(class_id)] = "person"
                        self._people_model_person_class_ids.append(int(class_id))
            if not self._people_model_person_class_ids:
                self.people_class_names[0] = "person"
                self._people_model_person_class_ids = [0]

        save_cooldown = float(kwargs.get("save_cooldown", 2.0))
        self.object_tracker = ObjectTracker(
            reset_interval=int(kwargs.get("detection_cooldown", detection_cooldown)),
            save_cooldown=save_cooldown,
        )
        self._track_match_iou_threshold = float(
            kwargs.get("track_match_iou_threshold", 0.5)
        )
        self._track_stale_timeout = float(kwargs.get("track_stale_timeout", 3.0))
        self._event_dedup_window_seconds = float(
            kwargs.get("event_dedup_window_seconds", 2.0)
        )
        self._event_dedup_iou_threshold = float(
            kwargs.get("event_dedup_iou_threshold", 0.5)
        )

        self._vest_stable_tracks: Dict[int, Dict[str, Any]] = {}
        self._hardhat_stable_tracks: Dict[int, Dict[str, Any]] = {}
        self._people_stable_tracks: Dict[int, Dict[str, Any]] = {}
        self._recent_emitted_violations: Dict[str, List[Dict[str, Any]]] = {}
        self._next_synthetic_track_id = 900000

        self._state_lock = threading.Lock()

        self.logger.info(
            f"PetrolimexDetectionModel loaded | Device: {self.device} | "
            f"Conf: {self.conf_threshold} | IoU: {self.iou_threshold} | "
            f"NovConf: {self.novest_conf_threshold} | "
            f"PeopleAux: {self.enable_people_aux_detection} | "
            f"PeopleConf: {self.people_conf_threshold} | "
            f"Classes: {self.class_names if self.class_names else 'auto'} | "
            f"NoHardhat classes: {self.no_hardhat_class_names if self.no_hardhat_class_names else 'auto'} | "
            f"HeadRegion: {self._head_region_fraction:.0%} | "
            f"EventDedup: {self._event_dedup_window_seconds:.1f}s"
        )

    @staticmethod
    def _normalize_label(raw_label: str) -> str:
        normalized = raw_label.strip().lower().replace(" ", "_")
        return PetrolimexDetectionModel.LABEL_ALIASES.get(normalized, "")

    def _resolve_track_id(
        self,
        tracker_id: Optional[int],
        bbox: tuple[int, int, int, int],
        stable_tracks: Dict[int, Dict[str, Any]],
    ) -> int:
        """
        Stabilize track IDs using the provided namespace dict.

        Each model (vest, hardhat, people) passes its own stable_tracks dict so
        that YOLO's internally assigned IDs never cross-contaminate each other.
        """
        now = time.time()
        self._cleanup_stale_tracks(now, stable_tracks)

        if tracker_id is not None and tracker_id in stable_tracks:
            stable_tracks[tracker_id]["bbox"] = bbox
            stable_tracks[tracker_id]["last_seen"] = now
            return tracker_id

        best_id: Optional[int] = None
        best_iou = self._track_match_iou_threshold
        for stable_id, info in stable_tracks.items():
            iou = self._compute_iou(bbox, info["bbox"])
            if iou > best_iou:
                best_id = stable_id
                best_iou = iou

        if best_id is not None:
            stable_tracks[best_id]["bbox"] = bbox
            stable_tracks[best_id]["last_seen"] = now
            return best_id

        stable_id = (
            int(tracker_id) if tracker_id is not None else self._next_synthetic_track_id
        )
        if tracker_id is None:
            self._next_synthetic_track_id += 1

        stable_tracks[stable_id] = {"bbox": bbox, "last_seen": now}
        return stable_id

    def _cleanup_stale_tracks(
        self, now: float, stable_tracks: Dict[int, Dict[str, Any]]
    ) -> None:
        stale_ids = [
            stable_id
            for stable_id, info in stable_tracks.items()
            if now - float(info["last_seen"]) > self._track_stale_timeout
        ]
        for stable_id in stale_ids:
            del stable_tracks[stable_id]

    def _cleanup_recent_emitted_violations(self, now: float) -> None:
        if self._event_dedup_window_seconds <= 0:
            self._recent_emitted_violations.clear()
            return

        violation_types_to_remove: List[str] = []
        for violation_type, entries in self._recent_emitted_violations.items():
            fresh_entries = [
                entry
                for entry in entries
                if now - float(entry["seen_at"]) < self._event_dedup_window_seconds
            ]
            if fresh_entries:
                self._recent_emitted_violations[violation_type] = fresh_entries
            else:
                violation_types_to_remove.append(violation_type)

        for violation_type in violation_types_to_remove:
            del self._recent_emitted_violations[violation_type]

    def _match_recent_emitted_violation(
        self,
        violation_type: str,
        track_id: int,
        bbox: tuple[int, int, int, int],
        now: float,
    ) -> bool:
        if self._event_dedup_window_seconds <= 0:
            return False

        entries = self._recent_emitted_violations.get(violation_type, [])
        matched_entry: Optional[Dict[str, Any]] = None
        best_iou = self._event_dedup_iou_threshold

        for entry in entries:
            if int(entry["track_id"]) == int(track_id):
                matched_entry = entry
                break

            iou = self._compute_iou(bbox, entry["bbox"])
            if iou >= best_iou:
                matched_entry = entry
                best_iou = iou

        if matched_entry is None:
            return False

        matched_entry["track_id"] = int(track_id)
        matched_entry["bbox"] = bbox
        matched_entry["seen_at"] = now
        return True

    def _remember_emitted_violation(
        self,
        violation_type: str,
        track_id: int,
        bbox: tuple[int, int, int, int],
        now: float,
    ) -> None:
        entries = self._recent_emitted_violations.setdefault(violation_type, [])
        entries.append(
            {
                "track_id": int(track_id),
                "bbox": bbox,
                "seen_at": now,
            }
        )

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

    @staticmethod
    def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _is_in_head_region(
        self,
        ppe_box: tuple[int, int, int, int],
        person_boxes: List[tuple[int, int, int, int]],
    ) -> bool:
        """
        Return True if the centre of ppe_box falls within the upper
        `_head_region_fraction` of any person bounding box.
        """
        cx, cy = self._bbox_center(ppe_box)
        for px1, py1, px2, py2 in person_boxes:
            head_bottom = py1 + (py2 - py1) * self._head_region_fraction
            if px1 <= cx <= px2 and py1 <= cy <= head_bottom:
                return True
        return False

    def _build_people_match_score(
        self,
        person_bbox: tuple[int, int, int, int],
        detection_bbox: tuple[int, int, int, int],
        detection_confidence: float,
    ) -> Optional[Tuple[int, float, float]]:
        dcx, dcy = self._bbox_center(detection_bbox)
        contains_center = (
            person_bbox[0] <= dcx <= person_bbox[2]
            and person_bbox[1] <= dcy <= person_bbox[3]
        )
        overlap = self._compute_iou(person_bbox, detection_bbox)

        if not contains_center and overlap < self.people_match_iou_threshold:
            return None

        return (1 if contains_center else 0, overlap, detection_confidence)

    def _assign_people_to_detections(
        self,
        people_candidates: List[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        target_classes: Set[str],
    ) -> Dict[int, int]:
        candidate_pairs: List[Tuple[Tuple[int, float, float], int, int]] = []

        for person_index, person in enumerate(people_candidates):
            person_bbox = tuple(int(v) for v in person.get("bbox", [])[:4])
            if len(person_bbox) != 4:
                continue

            for detection_index, detection in enumerate(detections):
                if detection.get("class_name") not in target_classes:
                    continue

                raw_bbox = detection.get("bbox") or []
                if len(raw_bbox) < 4:
                    continue

                detection_bbox = tuple(int(v) for v in raw_bbox[:4])
                score = self._build_people_match_score(
                    person_bbox,
                    detection_bbox,
                    float(detection.get("confidence", 0.0)),
                )
                if score is None:
                    continue

                candidate_pairs.append((score, person_index, detection_index))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        assignments: Dict[int, int] = {}
        used_people: Set[int] = set()
        used_detections: Set[int] = set()

        for score, person_index, detection_index in candidate_pairs:
            if person_index in used_people or detection_index in used_detections:
                continue
            assignments[person_index] = detection_index
            used_people.add(person_index)
            used_detections.add(detection_index)

        return assignments

    def _find_best_person_for_detection(
        self,
        detection: Dict[str, Any],
        people_candidates: List[Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[Tuple[int, float, float]]]:
        raw_bbox = detection.get("bbox") or []
        if len(raw_bbox) < 4:
            return None, None

        detection_bbox = tuple(int(v) for v in raw_bbox[:4])
        detection_confidence = float(detection.get("confidence", 0.0))

        best_person_index: Optional[int] = None
        best_score: Optional[Tuple[int, float, float]] = None

        for person_index, person in enumerate(people_candidates):
            person_bbox = tuple(int(v) for v in person.get("bbox", [])[:4])
            if len(person_bbox) != 4:
                continue

            score = self._build_people_match_score(
                person_bbox,
                detection_bbox,
                detection_confidence,
            )
            if score is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_person_index = person_index

        return best_person_index, best_score

    @staticmethod
    def _record_violation(
        violations_to_record: Dict[Tuple[int, str], Dict[str, Any]],
        track_id: int,
        violation_type: str,
        confidence: float,
        bbox: List[int],
    ) -> None:
        key = (int(track_id), violation_type)
        existing = violations_to_record.get(key)
        if existing is None or confidence > float(existing["confidence"]):
            violations_to_record[key] = {
                "track_id": int(track_id),
                "violation_type": violation_type,
                "confidence": confidence,
                "bbox": bbox,
            }

    def _draw_detection(
        self,
        annotated_frame: np.ndarray,
        detection: Dict[str, Any],
        violation_classes: Set[str],
    ) -> None:
        raw_bbox = detection.get("bbox") or []
        if len(raw_bbox) < 4:
            return

        x1, y1, x2, y2 = [int(v) for v in raw_bbox[:4]]
        class_name = str(detection.get("class_name", "object"))
        display_name = str(detection.get("display_name") or class_name)
        confidence = float(detection.get("confidence", 0.0))
        track_id = detection.get("track_id")
        is_violation = bool(
            detection.get("is_violation") or class_name in violation_classes
        )
        color = (0, 0, 255) if is_violation else (0, 255, 0)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{display_name} {confidence:.2f}"
        if track_id is not None:
            label += f" ID:{track_id}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - th - 10)
        cv2.rectangle(annotated_frame, (x1, top), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        with self._state_lock:
            return self._process_frame_locked(frame, **kwargs)

    def _process_frame_locked(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)
        current_time = time.time()
        self._cleanup_recent_emitted_violations(current_time)

        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)
        clean_frame = frame

        yolo_conf = min(self.conf_threshold, self.novest_conf_threshold)
        ppe_results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=yolo_conf,
            iou=self.iou_threshold,
            persist=True,
            verbose=False,
        )

        annotated_frame: Optional[np.ndarray] = frame.copy() if annotate else None
        all_detections: List[Dict[str, Any]] = []
        people_statuses: List[Dict[str, Any]] = []
        ppe_detections: List[Dict[str, Any]] = []
        violations_to_record: Dict[Tuple[int, str], Dict[str, Any]] = {}
        violation_classes = {"novest", "no_hardhat"}

        for result in ppe_results:
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

                raw_class_name = result.names.get(cls_id, f"class_{cls_id}")
                class_name = self.class_names.get(
                    cls_id, self._normalize_label(str(raw_class_name))
                )
                if class_name not in {"vest", "novest"}:
                    continue

                min_conf = (
                    self.novest_conf_threshold
                    if class_name == "novest"
                    else self.conf_threshold
                )
                if conf < min_conf:
                    continue

                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = self._bbox_center(bbox)
                if not is_point_in_any_roi((float(cx), float(cy)), roi_polys):
                    continue

                raw_track_id = track_ids[i] if i < len(track_ids) else None
                effective_track_id = self._resolve_track_id(
                    raw_track_id,
                    bbox,
                    self._vest_stable_tracks,
                )

                detection = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": list(bbox),
                    "track_id": effective_track_id,
                    "source_track_id": raw_track_id,
                    "source_model": "petrolimex",
                    "is_violation": class_name in violation_classes,
                }
                ppe_detections.append(detection)
                all_detections.append(detection)

        person_candidates: List[Dict[str, Any]] = []

        if self.enable_people_aux_detection and self.people_model is not None:
            people_track_kwargs: Dict[str, Any] = {
                "conf": self.people_conf_threshold,
                "persist": True,
                "verbose": False,
            }
            if self._people_model_person_class_ids:
                people_track_kwargs["classes"] = self._people_model_person_class_ids

            people_results = self._run_yolo_track(
                self.people_model,
                clean_frame,
                **people_track_kwargs,
            )

            for result in people_results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                track_ids = (
                    boxes.id.int().cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(boxes)
                )

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    raw_class_name = result.names.get(cls_id, f"class_{cls_id}")
                    normalized_class_name = self.people_class_names.get(
                        cls_id, self._normalize_label(str(raw_class_name))
                    )
                    if normalized_class_name != "person":
                        continue

                    conf = float(box.conf[0])
                    if conf < self.people_conf_threshold:
                        continue

                    bbox = tuple(int(v) for v in box.xyxy[0].tolist())
                    cx, cy = self._bbox_center(bbox)
                    if not is_point_in_any_roi((float(cx), float(cy)), roi_polys):
                        continue

                    raw_track_id = track_ids[i] if i < len(track_ids) else None
                    effective_track_id = self._resolve_track_id(
                        raw_track_id,
                        bbox,
                        self._people_stable_tracks,
                    )

                    person_candidates.append(
                        {
                            "class_id": cls_id,
                            "class_name": "person",
                            "confidence": conf,
                            "bbox": list(bbox),
                            "track_id": effective_track_id,
                            "source_track_id": raw_track_id,
                            "source_model": "people",
                        }
                    )

        if person_candidates:
            vest_assignments = self._assign_people_to_detections(
                person_candidates,
                ppe_detections,
                {"vest"},
            )
            positive_people_indices = set(vest_assignments.keys())
            best_novest_by_person: Dict[
                int, Tuple[Tuple[int, float, float], int]
            ] = {}

            for detection_index, detection in enumerate(ppe_detections):
                if detection.get("class_name") != "novest":
                    continue

                matched_person_index, matched_score = self._find_best_person_for_detection(
                    detection,
                    person_candidates,
                )
                if matched_person_index is None or matched_score is None:
                    continue

                detection["hidden_by_people_aux"] = True

                if matched_person_index in positive_people_indices:
                    continue

                current_best = best_novest_by_person.get(matched_person_index)
                if current_best is None or matched_score > current_best[0]:
                    best_novest_by_person[matched_person_index] = (
                        matched_score,
                        detection_index,
                    )

            for person_index, person in enumerate(person_candidates):
                matched_vest = None
                if person_index in vest_assignments:
                    matched_vest = ppe_detections[vest_assignments[person_index]]

                matched_novest = None
                best_novest_entry = best_novest_by_person.get(person_index)
                if best_novest_entry is not None:
                    matched_novest = ppe_detections[best_novest_entry[1]]

                ppe_status = "positive" if matched_vest is not None else "negative"
                status_detection = {
                    **person,
                    "display_name": f"person_{ppe_status}",
                    "ppe_status": ppe_status,
                    "matched_vest_track_id": (
                        matched_vest.get("track_id") if matched_vest else None
                    ),
                    "matched_novest_track_id": (
                        matched_novest.get("track_id") if matched_novest else None
                    ),
                    "is_violation": ppe_status == "negative",
                }
                people_statuses.append(status_detection)

                if ppe_status == "negative":
                    violation_confidence = float(
                        matched_novest.get("confidence", person["confidence"])
                        if matched_novest is not None
                        else person["confidence"]
                    )
                    self._record_violation(
                        violations_to_record,
                        int(person["track_id"]),
                        "novest",
                        violation_confidence,
                        list(person["bbox"]),
                    )

        all_detections = [
            detection
            for detection in all_detections
            if not detection.get("hidden_by_people_aux")
        ]

        for detection in ppe_detections:
            if detection["class_name"] != "novest":
                continue
            if detection.get("hidden_by_people_aux"):
                continue

            self._record_violation(
                violations_to_record,
                int(detection["track_id"]),
                "novest",
                float(detection["confidence"]),
                list(detection["bbox"]),
            )

        hardhat_results = self._run_yolo_track(
            self.no_hardhat_model,
            clean_frame,
            conf=self.no_hardhat_conf_threshold,
            iou=self.iou_threshold,
            persist=True,
            verbose=False,
        )

        person_boxes: List[tuple[int, int, int, int]] = []
        for result in hardhat_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self._hardhat_model_person_class_ids:
                    px1, py1, px2, py2 = box.xyxy[0].tolist()
                    person_boxes.append((int(px1), int(py1), int(px2), int(py2)))

        for result in hardhat_results:
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

                raw_class_name = result.names.get(cls_id, f"class_{cls_id}")
                class_name = self.no_hardhat_class_names.get(
                    cls_id, self._normalize_label(str(raw_class_name))
                )
                if class_name != "no_hardhat":
                    continue
                if conf < self.no_hardhat_conf_threshold:
                    continue

                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = self._bbox_center(bbox)
                if not is_point_in_any_roi((float(cx), float(cy)), roi_polys):
                    continue

                if person_boxes and not self._is_in_head_region(bbox, person_boxes):
                    continue

                raw_track_id = track_ids[i] if i < len(track_ids) else None
                effective_track_id = self._resolve_track_id(
                    raw_track_id,
                    bbox,
                    self._hardhat_stable_tracks,
                )

                all_detections.append(
                    {
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": list(bbox),
                        "track_id": effective_track_id,
                        "source_track_id": raw_track_id,
                        "source_model": "atld_92",
                        "is_violation": True,
                    }
                )
                self._record_violation(
                    violations_to_record,
                    effective_track_id,
                    "no_hardhat",
                    conf,
                    list(bbox),
                )

        if annotate and annotated_frame is not None:
            for detection in all_detections:
                self._draw_detection(annotated_frame, detection, violation_classes)
            for detection in people_statuses:
                self._draw_detection(annotated_frame, detection, violation_classes)
            draw_roi_overlays(annotated_frame, roi_polys, color=(0, 0, 255))

        events_to_create: List[Dict[str, Any]] = []
        for violation in violations_to_record.values():
            track_id = int(violation["track_id"])
            violation_type = str(violation["violation_type"])
            conf = float(violation["confidence"])
            bbox = list(violation["bbox"])
            bbox_tuple = tuple(int(v) for v in bbox[:4])

            if self._match_recent_emitted_violation(
                violation_type,
                track_id,
                bbox_tuple,
                current_time,
            ):
                self.logger.debug(
                    "Suppressed recent duplicate PPE violation: %s | ID: %s",
                    violation_type,
                    track_id,
                )
                continue

            if not self.object_tracker.should_record_event(track_id, violation_type):
                continue

            self._remember_emitted_violation(
                violation_type,
                track_id,
                bbox_tuple,
                current_time,
            )
            display_violation_type = self.VIOLATION_DISPLAY_MAPPING.get(
                violation_type, violation_type
            )
            events_to_create.append(
                {
                    "track_id": track_id,
                    "violation_type": violation_type,
                    "violation_display_type": display_violation_type,
                    "raw_violation_type": violation_type,
                    "event_type": self.EVENT_TYPE,
                    "description": self.EVENT_DESCRIPTION,
                    "confidence": conf,
                    "bbox": bbox,
                }
            )
            self.logger.info(
                f"Ghi nhan vi pham: {violation_type} | ID: {track_id} | Conf: {conf:.2f}"
            )

        event_triggered = len(events_to_create) > 0
        primary_event = events_to_create[0] if event_triggered else None

        metadata: Dict[str, Any] = {
            "type": "Nhan vien dau khi",
            "eventType": "Nhan vien dau khi",
            "severity": "high" if event_triggered else "low",
            "description": "Vi pham an toan lao dong",
            "detections": all_detections,
            "people_statuses": people_statuses,
            "violations": events_to_create,
            "count": len(all_detections),
            "people_count": len(people_statuses),
            "people_positive_count": sum(
                1
                for detection in people_statuses
                if detection.get("ppe_status") == "positive"
            ),
            "people_negative_count": sum(
                1
                for detection in people_statuses
                if detection.get("ppe_status") == "negative"
            ),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "Nhan vien dau khi model",
        }
        metadata["type"] = self.EVENT_TYPE
        metadata["eventType"] = self.EVENT_TYPE
        metadata["title"] = self.EVENT_TYPE
        metadata["description"] = self.EVENT_DESCRIPTION
        metadata["model_type"] = "petrolimex_detection_model"

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["violation_display"] = primary_event.get(
                "violation_display_type",
                primary_event["violation_type"],
            )
            metadata["confidence"] = primary_event["confidence"]
            metadata["track_id"] = primary_event["track_id"]

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )
