"""
Workspace Monitor Model - Giám sát khu vực làm việc
Kết hợp 3 chức năng:
  1. Đếm người trong vùng làm việc (ROI) → cảnh báo nếu vượt quá max_workers
  2. Phát hiện vi phạm PPE (NO-Vest) → cảnh báo nếu không mặc trang phục bảo hộ
  3. Nhận diện khuôn mặt → cảnh báo nếu người lạ (không thuộc danh sách đăng ký)
"""

import os
from typing import Dict, Any, Optional, List, Tuple, Set
import numpy as np
import cv2
import time
from datetime import datetime

from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult
from src.core.object_tracker import ObjectTracker
from src.utils.roi_utils import (
    DEFAULT_NORMALIZED_ROI,
    denormalize_roi,
    get_effective_roi,
    validate_normalized_roi_input,
)
from src.utils.alert_levels import AlertLevel
from src.database import SessionLocal
from src.models.face_profile import FaceProfile
from src.ai_models.auto_roi.reflective_wire_detector import ReflectiveWireDetector


class WorkspaceMonitorModel(BaseModel):
    PPE_LABEL_ALIASES = {
        "vest": "vest",
        "ppe_vest": "vest",
        "safety_vest": "vest",
        "reflective_vest": "vest",
        "no_vest": "no_vest",
        "novest": "no_vest",
        "without_vest": "no_vest",
    }

    __doc__ = """
    Model giám sát khu vực làm việc:
    - Đếm người → cảnh báo nếu vượt max_workers
    - PPE detection → cảnh báo nếu không mặc trang phục bảo hộ
    - Face recognition → cảnh báo nếu người lạ vào khu vực
    """

    def __init__(
        self,
        model_path: str = "src/ai_models/model_weights/yolo11m.pt",
        **kwargs: Any,
    ):
        """
        Initialize Workspace Monitor Model

        Args:
            model_path: Path to YOLO person detection model weights
            **kwargs: Additional parameters
                - ppe_model_path: Path to PPE detection model (default: ppe_100ep.pt)
                - max_workers: Max allowed people in workspace (default: 5)
                - conf_threshold: Person detection confidence (default: 0.4)
                - ppe_conf_threshold: PPE detection confidence (default: 0.45)
                - face_similarity_threshold: Face match threshold (default: 0.45)
                - detection_cooldown: Cooldown between same-type alerts in seconds (default: 300)
                - registered_employee_ids: List of employee IDs registered for this session
                - registered_worker_count: Registered worker slots for this camera/session
        """
        super().__init__(
            model_name="WorkspaceMonitorModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=kwargs.get("conf_threshold", 0.4),
            model_path=model_path,
            **kwargs,
        )

        # ── Person detection model (YOLO) ──────────────────────────
        self.person_model = YOLO(model_path)
        if self.device and model_path.endswith(".pt"):
            self.person_model.to(self.device)
        self.person_class = 0  # COCO person class
        self.conf_threshold: float = kwargs.get("conf_threshold", 0.4)

        # ── PPE detection model (YOLO) ─────────────────────────────
        ppe_model_path = kwargs.get(
            "ppe_model_path",
            "src/ai_models/model_weights/ppe_100ep.pt",
        )
        self.ppe_model = YOLO(ppe_model_path)
        if self.device and ppe_model_path.endswith(".pt"):
            self.ppe_model.to(self.device)
        self.ppe_conf_threshold: float = kwargs.get("ppe_conf_threshold", 0.45)
        self.ppe_conflict_margin: float = float(kwargs.get("ppe_conflict_margin", 0.1))
        self.ppe_conflict_iou_threshold: float = float(
            kwargs.get("ppe_conflict_iou_threshold", 0.1)
        )

        # PPE class mappings are resolved from model metadata to avoid hardcoded drift.
        self.ppe_class_names: Dict[int, str] = {}
        self.ppe_violation_classes: Dict[int, str] = {}
        self.ppe_safe_classes: Dict[int, str] = {}
        self.ppe_classes_to_track: List[int] = []
        self._configure_ppe_classes()

        # ── Face recognition engine ────────────────────────────────
        self.face_engine = None
        self.face_enabled = False
        self.enable_face_processing: bool = kwargs.get(
            "enable_face_processing",
            os.getenv("WORKSPACE_ENABLE_FACE_RECOGNITION", "false").lower() == "true",
        )

        if not self.enable_face_processing:
            self.logger.info(
                "Face processing is temporarily disabled for workspace monitor"
            )

        try:
            from src.face_recognition.face_engine import get_face_engine

            if self.enable_face_processing:
                self.face_engine = get_face_engine()
                if self.face_engine is not None:
                    self.face_enabled = True
                    self.logger.info(
                        "Face recognition engine loaded for workspace monitor"
                    )
                else:
                    self.logger.warning(
                        "Face recognition engine not available – "
                        "stranger detection will be disabled"
                    )
        except Exception as e:
            self.logger.warning(f"Failed to load face recognition engine: {e}")

        self.face_similarity_threshold: float = kwargs.get(
            "face_similarity_threshold",
            float(os.getenv("DEFAULT_FACE_SIMILARITY_THRESHOLD", "0.45")),
        )
        self.face_profiles_refresh_interval_sec: int = max(
            5,
            int(
                kwargs.get(
                    "face_profiles_refresh_interval_sec",
                    os.getenv("WORKSPACE_FACE_CACHE_REFRESH_SEC", "30"),
                )
            ),
        )
        self._face_profiles_cache: List[Dict[str, Any]] = []
        self._face_profiles_cache_ts: float = 0.0

        # ── Workspace parameters ───────────────────────────────────
        self.max_workers: int = kwargs.get("max_workers", 5)
        self.registered_employee_ids: List[str] = kwargs.get(
            "registered_employee_ids", []
        )
        self.registered_worker_count: int = max(
            0,
            int(kwargs.get("registered_worker_count", 0) or 0),
        )
        self.detection_cooldown: float = kwargs.get("detection_cooldown", 300)

        self._frame_counter: int = 0
        self.ppe_process_every_n_frames: int = max(
            1,
            int(
                kwargs.get(
                    "ppe_process_every_n_frames",
                    os.getenv("WORKSPACE_PPE_EVERY_N_FRAMES", "1"),
                )
            ),
        )
        self.face_process_every_n_frames: int = max(
            1,
            int(
                kwargs.get(
                    "face_process_every_n_frames",
                    os.getenv("WORKSPACE_FACE_EVERY_N_FRAMES", "5"),
                )
            ),
        )

        # ── Tracking & cooldowns ───────────────────────────────────
        # Auto ROI (reflective wire) MVP
        self.auto_roi_enabled: bool = bool(kwargs.get("auto_roi_enabled", False))
        self.auto_roi_source: str = str(
            kwargs.get("auto_roi_source", "reflective_wire")
        ).strip().lower()
        self.auto_roi_min_conf: float = float(kwargs.get("auto_roi_min_conf", 0.35))
        self.auto_roi_update_interval: int = max(
            1, int(kwargs.get("auto_roi_update_interval", 15) or 1)
        )
        self.auto_roi_stable_frames: int = max(
            1, int(kwargs.get("auto_roi_stable_frames", 3) or 1)
        )
        self.auto_roi_lock: bool = bool(kwargs.get("auto_roi_lock", False))
        self._auto_roi_similarity_threshold: float = float(
            kwargs.get("auto_roi_similarity_threshold", 0.08)
        )
        self._auto_roi_detector: Optional[ReflectiveWireDetector] = None
        if self.auto_roi_enabled and self.auto_roi_source == "reflective_wire":
            try:
                self._auto_roi_detector = ReflectiveWireDetector()
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize reflective-wire auto ROI detector: {e}"
                )
                self._auto_roi_detector = None
        self._auto_roi_last: Optional[List[List[float]]] = None
        self._auto_roi_last_confidence: float = 0.0
        self._auto_roi_candidate: Optional[List[List[float]]] = None
        self._auto_roi_candidate_count: int = 0
        self._auto_roi_locked: bool = False

        self.object_tracker = ObjectTracker(reset_interval=1800)  # 30 min
        self.last_overcrowd_event: float = 0.0  # timestamp of last overcrowd alert
        self.last_unknown_face_event: Dict[str, float] = {}  # emb_hash → timestamp
        self.face_scanned_tracks: Dict[int, Dict[str, Any]] = {}

        # EVN-style visual persistence (track_id -> state), giảm nhấp nháy bbox
        self.visual_absence_timeout_sec: float = float(
            kwargs.get(
                "visual_absence_timeout_sec",
                os.getenv("WORKSPACE_VISUAL_ABSENCE_TIMEOUT_SEC", "1.0"),
            )
        )
        self._person_track_state: Dict[int, Dict[str, Any]] = {}

        # ── Colors ─────────────────────────────────────────────────
        self.roi_color = (255, 200, 0)  # cyan-ish for workspace zone
        self.person_color = (0, 255, 0)  # green – normal
        self.violation_color = (0, 0, 255)  # red – violation
        self.stranger_color = (0, 0, 255)  # red – stranger

        self.logger.info(
            f"WorkspaceMonitorModel initialized | device={self.device} | "
            f"max_workers={self.max_workers} | "
            f"registered_worker_count={self.registered_worker_count} | "
            f"registered_employees={self.registered_employee_ids} | "
            f"face_enabled={self.face_enabled} | "
            f"auto_roi_enabled={self.auto_roi_enabled} | "
            f"auto_roi_source={self.auto_roi_source}"
        )

        if self.face_enabled:
            self._refresh_face_profiles_cache(force=True)

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _is_point_in_roi(
        self,
        point: Tuple[int, int],
        roi_poly_np: np.ndarray,
    ) -> bool:
        """Check if a point is inside the ROI polygon."""
        if roi_poly_np is None or len(roi_poly_np) < 3:
            return False

        try:
            return (
                cv2.pointPolygonTest(
                    roi_poly_np, (float(point[0]), float(point[1])), False
                )
                >= 0
            )
        except Exception:
            return False

    def _safe_build_roi(
        self,
        roi_input: Any,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[List[List[float]], np.ndarray, bool, Optional[str]]:
        """Validate ROI input and build a safe polygon with full-frame fallback."""
        default_roi: List[List[float]] = DEFAULT_NORMALIZED_ROI

        candidate_roi = get_effective_roi(roi_input)
        if candidate_roi is None:
            candidate_roi = default_roi

        is_valid, error_msg = validate_normalized_roi_input(candidate_roi)
        roi_fallback_used = not is_valid

        normalized_roi = default_roi if roi_fallback_used else candidate_roi
        if roi_fallback_used:
            self.logger.warning(
                f"Invalid ROI input for workspace monitor, fallback to full-frame: {error_msg}"
            )

        roi_polygon = denormalize_roi(normalized_roi, frame_w, frame_h)
        roi_poly_np = np.array(roi_polygon, dtype=np.int32)

        if len(roi_poly_np) < 3:
            normalized_roi = default_roi
            roi_polygon = denormalize_roi(normalized_roi, frame_w, frame_h)
            roi_poly_np = np.array(roi_polygon, dtype=np.int32)
            roi_fallback_used = True
            error_msg = "ROI denormalization produced < 3 points"
            self.logger.warning(
                "ROI denormalization invalid, fallback to full-frame polygon"
            )

        return normalized_roi, roi_poly_np, roi_fallback_used, error_msg

    @staticmethod
    def _roi_mean_point_distance(
        roi_a: Optional[List[List[float]]],
        roi_b: Optional[List[List[float]]],
    ) -> float:
        """Compute mean L1 point distance between two normalized polygons."""
        if not roi_a or not roi_b:
            return float("inf")
        if len(roi_a) != len(roi_b):
            return float("inf")

        distances: List[float] = []
        for point_a, point_b in zip(roi_a, roi_b):
            if len(point_a) != 2 or len(point_b) != 2:
                return float("inf")
            distances.append(
                abs(float(point_a[0]) - float(point_b[0]))
                + abs(float(point_a[1]) - float(point_b[1]))
            )
        if not distances:
            return float("inf")
        return float(np.mean(distances))

    def _resolve_workspace_roi_input(
        self,
        frame: np.ndarray,
        kwargs: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """Resolve ROI input by combining manual ROI and optional auto ROI."""
        manual_roi_input: Any = kwargs.get("roi", DEFAULT_NORMALIZED_ROI)

        auto_roi_enabled = bool(kwargs.get("auto_roi_enabled", self.auto_roi_enabled))
        auto_roi_source = str(
            kwargs.get("auto_roi_source", self.auto_roi_source)
        ).strip().lower()
        try:
            auto_roi_min_conf = float(
                kwargs.get("auto_roi_min_conf", self.auto_roi_min_conf)
            )
        except (TypeError, ValueError):
            auto_roi_min_conf = self.auto_roi_min_conf
        auto_roi_min_conf = max(0.0, min(1.0, auto_roi_min_conf))

        try:
            auto_roi_update_interval = max(
                1,
                int(kwargs.get("auto_roi_update_interval", self.auto_roi_update_interval)),
            )
        except (TypeError, ValueError):
            auto_roi_update_interval = self.auto_roi_update_interval

        try:
            auto_roi_stable_frames = max(
                1, int(kwargs.get("auto_roi_stable_frames", self.auto_roi_stable_frames))
            )
        except (TypeError, ValueError):
            auto_roi_stable_frames = self.auto_roi_stable_frames

        auto_roi_lock = bool(kwargs.get("auto_roi_lock", self.auto_roi_lock))
        roi_edit_mode = bool(kwargs.get("roi_edit_mode", False))

        auto_meta: Dict[str, Any] = {
            "roi_source": "manual",
            "auto_roi_enabled": auto_roi_enabled,
            "auto_roi_used": False,
            "auto_roi_source": auto_roi_source,
            "auto_roi_status": "disabled",
            "auto_roi_confidence": 0.0,
            "auto_roi_points": None,
            "auto_roi_min_conf": auto_roi_min_conf,
            "auto_roi_update_interval": auto_roi_update_interval,
            "auto_roi_stable_frames": auto_roi_stable_frames,
            "auto_roi_lock": auto_roi_lock,
        }

        if not auto_roi_enabled:
            return manual_roi_input, auto_meta

        if roi_edit_mode:
            auto_meta["auto_roi_status"] = "bypassed_roi_edit_mode"
            return manual_roi_input, auto_meta

        if auto_roi_source != "reflective_wire":
            auto_meta["auto_roi_status"] = "unsupported_source"
            return manual_roi_input, auto_meta

        if auto_roi_lock and self._auto_roi_locked and self._auto_roi_last is not None:
            auto_meta["roi_source"] = "auto_reflective_wire"
            auto_meta["auto_roi_used"] = True
            auto_meta["auto_roi_status"] = "locked"
            auto_meta["auto_roi_confidence"] = self._auto_roi_last_confidence
            auto_meta["auto_roi_points"] = self._auto_roi_last
            return self._auto_roi_last, auto_meta

        auto_meta["auto_roi_status"] = "waiting_update_window"
        should_run_detection = self._frame_counter % auto_roi_update_interval == 0
        if should_run_detection:
            if self._auto_roi_detector is None:
                try:
                    self._auto_roi_detector = ReflectiveWireDetector()
                except Exception as e:
                    self.logger.warning(
                        f"Auto ROI detector lazy init failed: {e}"
                    )
                    self._auto_roi_detector = None
            if self._auto_roi_detector is None:
                auto_meta["auto_roi_status"] = "detector_unavailable"
            else:
                try:
                    detection_result = self._auto_roi_detector.detect(frame)
                except Exception as e:
                    self.logger.warning(f"Auto ROI detection failed: {e}")
                    detection_result = None
                    auto_meta["auto_roi_status"] = "detector_error"

                if detection_result is not None:
                    candidate_roi = detection_result.roi
                    candidate_confidence = float(detection_result.confidence)
                    candidate_valid, _ = validate_normalized_roi_input(candidate_roi)
                    if (
                        candidate_valid
                        and candidate_confidence >= auto_roi_min_conf
                        and candidate_roi is not None
                    ):
                        distance = self._roi_mean_point_distance(
                            self._auto_roi_candidate, candidate_roi
                        )
                        if distance <= self._auto_roi_similarity_threshold:
                            self._auto_roi_candidate_count += 1
                        else:
                            self._auto_roi_candidate = candidate_roi
                            self._auto_roi_candidate_count = 1

                        if self._auto_roi_candidate_count >= auto_roi_stable_frames:
                            self._auto_roi_last = candidate_roi
                            self._auto_roi_last_confidence = candidate_confidence
                            self._auto_roi_candidate = None
                            self._auto_roi_candidate_count = 0
                            if auto_roi_lock:
                                self._auto_roi_locked = True
                            auto_meta["auto_roi_status"] = (
                                "updated_locked"
                                if auto_roi_lock and self._auto_roi_locked
                                else "updated"
                            )
                        else:
                            auto_meta["auto_roi_status"] = (
                                "stabilizing_"
                                f"{self._auto_roi_candidate_count}/{auto_roi_stable_frames}"
                            )
                    else:
                        self._auto_roi_candidate = None
                        self._auto_roi_candidate_count = 0
                        if candidate_roi is None:
                            auto_meta["auto_roi_status"] = "no_candidate"
                        else:
                            auto_meta["auto_roi_status"] = "low_confidence"

        if self._auto_roi_last is not None:
            auto_meta["roi_source"] = "auto_reflective_wire"
            auto_meta["auto_roi_used"] = True
            auto_meta["auto_roi_confidence"] = self._auto_roi_last_confidence
            auto_meta["auto_roi_points"] = self._auto_roi_last
            if auto_meta["auto_roi_status"] in {
                "waiting_update_window",
                "detector_unavailable",
                "no_candidate",
                "low_confidence",
            }:
                auto_meta["auto_roi_status"] = "using_cached"
            return self._auto_roi_last, auto_meta

        return manual_roi_input, auto_meta

    def _draw_roi(
        self,
        frame: np.ndarray,
        roi_poly_np: np.ndarray,
        show_points: bool = False,
    ) -> np.ndarray:
        """Draw ROI overlay on frame.

        Args:
            frame: Input frame (BGR)
            roi_poly_np: ROI polygon as numpy array of int32 points
            show_points: If True, draw numbered vertex circles for editing
        """
        if roi_poly_np is None or len(roi_poly_np) < 3:
            return frame

        cv2.polylines(
            frame,
            [roi_poly_np.reshape((-1, 1, 2))],
            True,
            self.roi_color,
            2,
        )
        cv2.putText(
            frame,
            "KHU VUC LAM VIEC",
            (roi_poly_np[0][0] + 5, roi_poly_np[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.roi_color,
            2,
            cv2.LINE_AA,
        )

        if show_points:
            for i, pt in enumerate(roi_poly_np):
                x, y = int(pt[0]), int(pt[1])
                # Outer circle (white border)
                cv2.circle(frame, (x, y), 14, (255, 255, 255), 2, cv2.LINE_AA)
                # Inner circle (filled cyan)
                cv2.circle(frame, (x, y), 12, self.roi_color, -1, cv2.LINE_AA)
                # Point number label
                label = str(i + 1)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.putText(
                    frame,
                    label,
                    (x - tw // 2, y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        return frame

    @classmethod
    def _normalize_ppe_label(cls, raw_label: str) -> str:
        """Normalize PPE labels from YOLO metadata into internal names."""
        normalized = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
        return cls.PPE_LABEL_ALIASES.get(normalized, "")

    def _configure_ppe_classes(self) -> None:
        """Resolve PPE class IDs from the loaded model names."""
        self.ppe_class_names = {}
        self.ppe_violation_classes = {}
        self.ppe_safe_classes = {}

        model_names = getattr(self.ppe_model, "names", {})
        if isinstance(model_names, list):
            model_names = {idx: name for idx, name in enumerate(model_names)}

        if isinstance(model_names, dict):
            for class_id, raw_label in model_names.items():
                normalized_label = self._normalize_ppe_label(str(raw_label))
                if normalized_label == "no_vest":
                    self.ppe_class_names[int(class_id)] = "No Vest"
                    self.ppe_violation_classes[int(class_id)] = "No Vest"
                elif normalized_label == "vest":
                    self.ppe_class_names[int(class_id)] = "Vest"
                    self.ppe_safe_classes[int(class_id)] = "Vest"

        if not self.ppe_violation_classes:
            self.ppe_class_names[2] = "No Vest"
            self.ppe_violation_classes = {2: "No Vest"}
            self.logger.warning(
                "Unable to resolve NO-Vest class from PPE model metadata, fallback to class 2"
            )

        self.ppe_classes_to_track = sorted(self.ppe_class_names.keys())
        self.logger.info(
            "Workspace PPE classes resolved | "
            f"tracked={self.ppe_classes_to_track} | "
            f"violations={self.ppe_violation_classes} | "
            f"safe={self.ppe_safe_classes}"
        )

    @staticmethod
    def _compute_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two xyxy boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _get_embedding_hash(self, embedding: np.ndarray) -> str:
        """Generate a short hash for an unknown face embedding (for cooldown tracking)."""
        return str(tuple(np.round(embedding[:8], 2)))

    def _should_trigger_unknown_face(self, embedding: np.ndarray) -> bool:
        """Check cooldown for a specific unknown face."""
        emb_hash = self._get_embedding_hash(embedding)
        now = time.time()
        last = self.last_unknown_face_event.get(emb_hash, 0.0)
        if now - last < self.detection_cooldown:
            return False
        self.last_unknown_face_event[emb_hash] = now
        return True

    def _refresh_face_profiles_cache(self, force: bool = False) -> None:
        """Refresh active face profiles cache from DB on interval."""
        now = time.time()
        if (
            not force
            and self._face_profiles_cache_ts > 0
            and now - self._face_profiles_cache_ts
            < self.face_profiles_refresh_interval_sec
        ):
            return

        db = SessionLocal()
        try:
            profiles = (
                db.query(FaceProfile)
                .filter(
                    FaceProfile.is_active == True,
                    FaceProfile.avg_face_embedding.isnot(None),
                )
                .all()
            )

            cached: List[Dict[str, Any]] = []
            for profile in profiles:
                emb = np.asarray(profile.avg_face_embedding, dtype=np.float32)
                if emb.size == 0:
                    continue
                norm = float(np.linalg.norm(emb))
                if norm <= 0:
                    continue
                emb = emb / norm
                cached.append(
                    {
                        "employee_id": profile.employee_id,
                        "employee_name": profile.employee_name,
                        "embedding": emb,
                    }
                )

            self._face_profiles_cache = cached
            self._face_profiles_cache_ts = now
        except Exception as e:
            self.logger.error(f"Failed to refresh face cache: {e}", exc_info=True)
        finally:
            db.close()

    def _find_best_face_match_cached(
        self, query_embedding: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """Find best face match from in-memory cache using cosine similarity."""
        if not self._face_profiles_cache:
            return None, 0.0

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm <= 0:
            return None, 0.0
        query = query / query_norm

        best_profile: Optional[Dict[str, Any]] = None
        best_similarity = -1.0

        for profile in self._face_profiles_cache:
            similarity = float(np.dot(query, profile["embedding"]))
            if similarity > best_similarity:
                best_similarity = similarity
                best_profile = profile

        if best_profile is None or best_similarity < self.face_similarity_threshold:
            return None, 0.0

        return best_profile, best_similarity

    def _cleanup_stale_track_state(self, current_time: float) -> None:
        """EVN-style cleanup: xóa track không thấy trong một khoảng thời gian."""
        expired_person = [
            tid
            for tid, state in self._person_track_state.items()
            if current_time - float(state.get("last_seen", 0.0))
            >= self.visual_absence_timeout_sec
        ]
        for tid in expired_person:
            self._person_track_state.pop(tid, None)
            self.face_scanned_tracks.pop(tid, None)

    def _match_face_to_person_track(
        self,
        face_bbox: Optional[List[float]],
        tracked_people: List[Dict[str, Any]],
        allowed_track_ids: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """Match a face box to the most likely person track in ROI."""
        if face_bbox is None or len(face_bbox) < 4:
            return None

        fx1, fy1, fx2, fy2 = [int(v) for v in face_bbox[:4]]
        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2

        candidates: List[Tuple[float, int]] = []
        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue

            tid = int(track_id)
            if allowed_track_ids is not None and tid not in allowed_track_ids:
                continue

            x1, y1, x2, y2 = person["bbox"]
            if x1 <= fcx <= x2 and y1 <= fcy <= y2:
                # Prefer smaller distance between centers.
                pcx, pcy = (x1 + x2) // 2, (y1 + y2) // 2
                dist2 = float((pcx - fcx) ** 2 + (pcy - fcy) ** 2)
                candidates.append((dist2, tid))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _match_ppe_to_person_track(
        self,
        ppe_bbox: Optional[List[int]],
        tracked_people: List[Dict[str, Any]],
    ) -> Optional[int]:
        """Match a PPE box to the most likely person track."""
        if ppe_bbox is None or len(ppe_bbox) < 4:
            return None

        px1, py1, px2, py2 = [int(v) for v in ppe_bbox[:4]]
        pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2

        containing_candidates: List[Tuple[float, int]] = []
        best_overlap_track_id: Optional[int] = None
        best_overlap = 0.0

        for person in tracked_people:
            track_id = person.get("track_id")
            if track_id is None:
                continue

            tid = int(track_id)
            x1, y1, x2, y2 = person["bbox"]
            if x1 <= pcx <= x2 and y1 <= pcy <= y2:
                person_cx, person_cy = (x1 + x2) // 2, (y1 + y2) // 2
                dist2 = float((person_cx - pcx) ** 2 + (person_cy - pcy) ** 2)
                containing_candidates.append((dist2, tid))
                continue

            overlap = self._compute_iou((px1, py1, px2, py2), (x1, y1, x2, y2))
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_track_id = tid

        if containing_candidates:
            containing_candidates.sort(key=lambda item: item[0])
            return containing_candidates[0][1]

        if best_overlap_track_id is not None and best_overlap >= 0.05:
            return best_overlap_track_id

        return None

    def _should_suppress_no_vest(
        self,
        no_vest_detection: Dict[str, Any],
        vest_detection: Optional[Dict[str, Any]],
    ) -> bool:
        """Suppress NO-Vest when the same person also has a plausible Vest detection."""
        if vest_detection is None:
            return False

        no_vest_bbox = no_vest_detection.get("bbox")
        vest_bbox = vest_detection.get("bbox")
        if no_vest_bbox is None or vest_bbox is None:
            return False

        no_vest_conf = float(no_vest_detection.get("confidence", 0.0))
        vest_conf = float(vest_detection.get("confidence", 0.0))
        if vest_conf <= 0:
            return False

        no_vest_person_track = no_vest_detection.get("person_track_id")
        vest_person_track = vest_detection.get("person_track_id")
        same_person_track = (
            no_vest_person_track is not None
            and no_vest_person_track == vest_person_track
        )

        overlap = self._compute_iou(
            tuple(int(v) for v in no_vest_bbox[:4]),
            tuple(int(v) for v in vest_bbox[:4]),
        )
        if not same_person_track and overlap < self.ppe_conflict_iou_threshold:
            return False

        return vest_conf + self.ppe_conflict_margin >= no_vest_conf

    # ──────────────────────────────────────────────────────────────
    # Main processing
    # ──────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, **kwargs: Any) -> DetectionResult:
        """
        Process a single frame for workspace monitoring.

        Args:
            frame: Input frame (BGR)
            **kwargs: Additional parameters
                - roi: Normalized ROI polygon [[x1,y1], ...] (values 0‒1).
                       Tối thiểu 3 điểm, mặc định 4 điểm xác định vùng làm việc.
                - roi_edit_mode: Bật chế độ chỉnh sửa ROI trên frame.
                       Khi True, frame chỉ hiển thị các điểm ROI để chỉnh sửa,
                       không chạy inference AI.
                - registered_employee_ids: Override session employee list
                - registered_worker_count: Override registered worker slots for this call
                - max_workers: Override max workers for this call
                - annotate: Whether to draw annotations (default: True)

        Returns:
            DetectionResult with event=True if any violation detected
        """
        current_time = time.time()
        self._frame_counter += 1
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)

        self._cleanup_stale_track_state(current_time)

        max_workers = kwargs.get("max_workers", self.max_workers)
        registered_ids = kwargs.get(
            "registered_employee_ids", self.registered_employee_ids
        )
        registered_worker_count_raw = kwargs.get(
            "registered_worker_count", self.registered_worker_count
        )

        try:
            registered_worker_count = max(0, int(registered_worker_count_raw or 0))
        except (TypeError, ValueError):
            registered_worker_count = 0

        if not isinstance(registered_ids, list):
            registered_ids = []

        if registered_worker_count > 0:
            effective_worker_limit = registered_worker_count
            worker_limit_source = "registered_worker_count"
        elif len(registered_ids) > 0:
            effective_worker_limit = len(registered_ids)
            worker_limit_source = "registered_employee_ids"
        else:
            effective_worker_limit = max_workers
            worker_limit_source = "max_workers"

        roi_input, auto_roi_meta = self._resolve_workspace_roi_input(frame, kwargs)
        normalized_roi_input = roi_input
        (
            normalized_roi,
            roi_poly_np,
            roi_fallback_used,
            roi_validation_error,
        ) = self._safe_build_roi(normalized_roi_input, w, h)

        roi_edit_mode = kwargs.get("roi_edit_mode", False)
        if roi_edit_mode:
            return self._process_roi_edit_mode(
                frame,
                roi_poly_np,
                normalized_roi,
                w,
                h,
                roi_fallback_used=roi_fallback_used,
                roi_validation_error=roi_validation_error,
            )

        clean_frame = frame
        annotated_frame = frame.copy() if annotate else None
        if annotate and annotated_frame is not None:
            annotated_frame = self._draw_roi(annotated_frame, roi_poly_np)

        all_detections: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []

        person_results = self._run_yolo_track(
            self.person_model,
            clean_frame,
            persist=True,
            conf=self.conf_threshold,
            classes=[self.person_class],
            verbose=False,
        )

        people_in_roi: List[Dict[str, Any]] = []
        current_person_track_ids: Set[int] = set()

        if person_results and person_results[0].boxes is not None:
            boxes = person_results[0].boxes
            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if not self._is_point_in_roi((cx, cy), roi_poly_np):
                    if annotate and annotated_frame is not None:
                        cv2.rectangle(
                            annotated_frame, (x1, y1), (x2, y2), self.person_color, 1
                        )
                    continue

                person_info = {
                    "class_id": 0,
                    "class_name": "Person",
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                    "center": [cx, cy],
                }
                all_detections.append(person_info)
                people_in_roi.append(person_info)

                if track_id is not None:
                    tid = int(track_id)
                    current_person_track_ids.add(tid)
                    self._person_track_state[tid] = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "last_seen": current_time,
                    }

                if annotate and annotated_frame is not None:
                    cv2.rectangle(
                        annotated_frame, (x1, y1), (x2, y2), self.person_color, 2
                    )
                    label = f"ID:{track_id}" if track_id is not None else "Person"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.person_color,
                        1,
                    )

        if annotate and annotated_frame is not None:
            for tid, state in self._person_track_state.items():
                if tid in current_person_track_ids:
                    continue
                x1, y1, x2, y2 = state["bbox"]
                conf = float(state.get("confidence", 0.0))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.person_color, 1)
                cv2.putText(
                    annotated_frame,
                    f"ID:{tid} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    self.person_color,
                    1,
                )

        people_count = len(people_in_roi)
        if people_count > effective_worker_limit:
            time_since_last = current_time - self.last_overcrowd_event
            if time_since_last >= self.detection_cooldown:
                self.last_overcrowd_event = current_time
                violations.append(
                    {
                        "violation_type": "Vượt quá số lượng người",
                        "detail": (
                            f"Phát hiện {people_count}/{effective_worker_limit} người trong khu vực "
                            f"(nguồn ngưỡng: {worker_limit_source})"
                        ),
                        "people_count": people_count,
                        "max_workers": max_workers,
                        "effective_worker_limit": effective_worker_limit,
                        "worker_limit_source": worker_limit_source,
                        "confidence": 1.0,
                    }
                )

        if annotate and annotated_frame is not None:
            count_color = (
                self.violation_color
                if people_count > effective_worker_limit
                else self.person_color
            )
            cv2.putText(
                annotated_frame,
                f"So nguoi: {people_count}/{effective_worker_limit}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                count_color,
                2,
            )

        ppe_results = self._run_yolo_track(
            self.ppe_model,
            clean_frame,
            conf=self.ppe_conf_threshold,
            classes=self.ppe_classes_to_track,
            persist=True,
            verbose=False,
        )

        ppe_detections: List[Dict[str, Any]] = []
        suppressed_ppe_detections: List[Dict[str, Any]] = []
        for result in ppe_results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            ppe_track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )

            for box, track_id in zip(boxes, ppe_track_ids):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.ppe_class_names.get(cls)
                if class_name is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if not self._is_point_in_roi((cx, cy), roi_poly_np):
                    continue

                bbox = [x1, y1, x2, y2]
                # Synthetic track_id fallback khi YOLO tracker không gán được ID
                person_track_id = self._match_ppe_to_person_track(bbox, people_in_roi)
                effective_track_id = (
                    int(person_track_id) if person_track_id is not None else None
                )
                if effective_track_id is None and track_id is not None:
                    effective_track_id = int(track_id)
                if effective_track_id is None:
                    gx, gy = cx // 50, cy // 50
                    effective_track_id = hash((class_name, gx, gy)) & 0x7FFFFFFF
                ppe_detections.append(
                    {
                        "class_id": cls,
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": bbox,
                        "track_id": effective_track_id,
                        "source_track_id": track_id,
                        "person_track_id": person_track_id,
                    }
                )

        ppe_detections_by_track: Dict[int, List[Dict[str, Any]]] = {}
        for detection in ppe_detections:
            effective_track_id = int(detection["track_id"])
            ppe_detections_by_track.setdefault(effective_track_id, []).append(detection)

        for effective_track_id, track_detections in ppe_detections_by_track.items():
            best_no_vest = max(
                (
                    detection
                    for detection in track_detections
                    if detection["class_name"] == "No Vest"
                ),
                key=lambda item: item["confidence"],
                default=None,
            )
            if best_no_vest is None:
                continue

            best_vest = max(
                (
                    detection
                    for detection in track_detections
                    if detection["class_name"] == "Vest"
                ),
                key=lambda item: item["confidence"],
                default=None,
            )
            if self._should_suppress_no_vest(best_no_vest, best_vest):
                suppressed_ppe_detections.append(best_no_vest)
                continue

            violation_key = f"PPE_{best_no_vest['class_name']}"
            if self.object_tracker.should_record_event(
                effective_track_id, violation_key
            ):
                violations.append(
                    {
                        "violation_type": best_no_vest["class_name"],
                        "track_id": effective_track_id,
                        "confidence": best_no_vest["confidence"],
                        "bbox": best_no_vest["bbox"],
                    }
                )

            if annotate and annotated_frame is not None:
                x1, y1, x2, y2 = best_no_vest["bbox"]
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2), self.violation_color, 3
                )
                label = f"{best_no_vest['class_name']} {best_no_vest['confidence']:.2f}"
                label += f" ID:{effective_track_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - th - 10),
                    (x1 + tw + 5, y1),
                    self.violation_color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        if self.face_enabled and self.face_engine is not None:
            try:
                unscanned_track_ids = {
                    int(tid)
                    for tid in current_person_track_ids
                    if int(tid) not in self.face_scanned_tracks
                }

                if unscanned_track_ids:
                    self._refresh_face_profiles_cache()
                    # FaceDetector.detect() now returns a 3-tuple
                    # (tensors, confidences, bboxes). Use the bbox from
                    # the public API instead of poking at the underlying
                    # MTCNN-only `.detector.detector.detect(rgb)` (which
                    # crashes on the new RetinaFace backend).
                    _detect_out = self.face_engine.detector.detect(clean_frame)
                    if isinstance(_detect_out, tuple) and len(_detect_out) == 3:
                        face_tensors, confidences, face_boxes = _detect_out
                    else:
                        face_tensors, confidences = _detect_out
                        face_boxes = None

                    if face_tensors is not None and len(face_tensors) > 0:
                        embeddings = self.face_engine.embedder.embed(face_tensors)
                        scan_count = min(len(embeddings), len(confidences))

                        for i in range(scan_count):
                            embedding = embeddings[i]
                            face_conf = confidences[i]
                            face_bbox: Optional[List[float]] = None

                            if face_boxes is not None and i < len(face_boxes):
                                box_values = face_boxes[i]
                                if box_values is not None and len(box_values) >= 4:
                                    face_bbox = [
                                        float(box_values[0]),
                                        float(box_values[1]),
                                        float(box_values[2]),
                                        float(box_values[3]),
                                    ]

                            matched_track_id = self._match_face_to_person_track(
                                face_bbox=face_bbox,
                                tracked_people=people_in_roi,
                                allowed_track_ids=unscanned_track_ids,
                            )
                            if (
                                matched_track_id is None
                                or matched_track_id in self.face_scanned_tracks
                            ):
                                continue

                            matched_profile, similarity = (
                                self._find_best_face_match_cached(embedding)
                            )
                            scan_result: Dict[str, Any] = {
                                "track_id": matched_track_id,
                                "scanned_at": current_time,
                                "confidence": float(face_conf),
                            }
                            if face_bbox is not None:
                                scan_result["face_bbox"] = [int(v) for v in face_bbox]

                            if matched_profile is not None:
                                employee_id = matched_profile["employee_id"]
                                employee_name = matched_profile["employee_name"]
                                scan_result["employee_id"] = employee_id
                                scan_result["employee_name"] = employee_name
                                scan_result["similarity"] = float(similarity)

                                if (
                                    len(registered_ids) > 0
                                    and employee_id not in registered_ids
                                ):
                                    scan_result["status"] = "known_unregistered"
                                    violations.append(
                                        {
                                            "violation_type": "Người không đăng ký",
                                            "track_id": matched_track_id,
                                            "employee_id": employee_id,
                                            "employee_name": employee_name,
                                            "similarity": float(similarity),
                                            "confidence": float(face_conf),
                                        }
                                    )
                                else:
                                    scan_result["status"] = "registered"
                            else:
                                scan_result["status"] = "unknown"
                                violations.append(
                                    {
                                        "violation_type": "Người lạ trong khu vực",
                                        "track_id": matched_track_id,
                                        "employee_id": None,
                                        "employee_name": None,
                                        "similarity": 0.0,
                                        "confidence": float(face_conf),
                                    }
                                )

                            self.face_scanned_tracks[matched_track_id] = scan_result
                            unscanned_track_ids.discard(matched_track_id)

                if annotate and annotated_frame is not None:
                    for person in people_in_roi:
                        track_id = person.get("track_id")
                        if track_id is None:
                            continue
                        tid = int(track_id)
                        if tid not in self.face_scanned_tracks:
                            continue

                        status_data = self.face_scanned_tracks[tid]
                        x1, y1, _, _ = person["bbox"]
                        status = status_data.get("status")

                        if status == "registered":
                            text = f"ID:{tid} FACE:OK"
                            color = self.person_color
                        elif status == "known_unregistered":
                            name = status_data.get("employee_name") or "KHONG DANG KY"
                            text = f"ID:{tid} {name}"
                            color = self.stranger_color
                        elif status == "unknown":
                            text = f"ID:{tid} NGUOI LA"
                            color = self.stranger_color
                        else:
                            text = f"ID:{tid} FACE:SCANNED"
                            color = self.person_color

                        cv2.putText(
                            annotated_frame,
                            text,
                            (x1, max(20, y1 - 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

            except Exception as e:
                self.logger.error(f"Face recognition error: {e}", exc_info=True)

        event_triggered = len(violations) > 0
        primary_event = violations[0] if violations else None

        metadata: Dict[str, Any] = {
            "type": "Giám sát khu vực làm việc",
            "severity": "high" if event_triggered else "low",
            "detections": all_detections,
            "violations": violations,
            "roi": normalized_roi,
            "roi_source": auto_roi_meta.get("roi_source", "manual"),
            "roi_fallback_used": roi_fallback_used,
            "people_count": people_count,
            "max_workers": max_workers,
            "registered_worker_count": registered_worker_count,
            "effective_worker_limit": effective_worker_limit,
            "worker_limit_source": worker_limit_source,
            "registered_employees": registered_ids,
            "count": len(all_detections),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_type": "workspace_monitor",
            "ppe_process_every_n_frames": self.ppe_process_every_n_frames,
            "ppe_classes_to_track": self.ppe_classes_to_track,
            "ppe_detections": ppe_detections,
            "ppe_suppressed_count": len(suppressed_ppe_detections),
            "face_process_every_n_frames": self.face_process_every_n_frames,
            "ppe_ran_this_frame": True,
            "face_ran_this_frame": True,
            "face_cache_profiles": len(self._face_profiles_cache),
            "face_scanned_track_count": len(self.face_scanned_tracks),
            "visual_absence_timeout_sec": self.visual_absence_timeout_sec,
            "person_track_state_count": len(self._person_track_state),
            "auto_roi_enabled": auto_roi_meta.get("auto_roi_enabled", False),
            "auto_roi_used": auto_roi_meta.get("auto_roi_used", False),
            "auto_roi_source": auto_roi_meta.get("auto_roi_source", "reflective_wire"),
            "auto_roi_status": auto_roi_meta.get("auto_roi_status", "disabled"),
            "auto_roi_confidence": auto_roi_meta.get("auto_roi_confidence", 0.0),
            "auto_roi_points": auto_roi_meta.get("auto_roi_points"),
            "auto_roi_min_conf": auto_roi_meta.get("auto_roi_min_conf", 0.0),
            "auto_roi_update_interval": auto_roi_meta.get(
                "auto_roi_update_interval", self.auto_roi_update_interval
            ),
            "auto_roi_stable_frames": auto_roi_meta.get(
                "auto_roi_stable_frames", self.auto_roi_stable_frames
            ),
            "auto_roi_lock": auto_roi_meta.get("auto_roi_lock", self.auto_roi_lock),
        }

        if roi_validation_error:
            metadata["roi_validation_error"] = roi_validation_error

        if primary_event:
            metadata["violation"] = primary_event["violation_type"]
            metadata["confidence"] = primary_event.get("confidence", 0.0)
            if "track_id" in primary_event:
                metadata["track_id"] = primary_event["track_id"]

        return DetectionResult(
            frame=annotated_frame, event=event_triggered, metadata=metadata
        )

    # ──────────────────────────────────────────────────────────────
    # ROI Edit Mode
    # ──────────────────────────────────────────────────────────────

    def _process_roi_edit_mode(
        self,
        frame: np.ndarray,
        roi_poly_np: np.ndarray,
        normalized_roi: list,
        frame_w: int,
        frame_h: int,
        roi_fallback_used: bool = False,
        roi_validation_error: Optional[str] = None,
    ) -> DetectionResult:
        """
        Render ROI editing overlay — no AI inference.

        Hiển thị các điểm ROI đánh số thứ tự trên frame để người dùng
        xem và chỉnh sửa ROI thông qua API.
        """
        edit_frame = frame.copy()

        # Draw ROI polygon with numbered vertex points
        edit_frame = self._draw_roi(edit_frame, roi_poly_np, show_points=True)

        # Guide text at top of frame
        num_points = len(roi_poly_np)
        guide_text = f"CHE DO CHINH SUA ROI - {num_points} diem (toi thieu 3)"
        cv2.putText(
            edit_frame,
            guide_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Show coordinate info for each point
        for i, pt in enumerate(normalized_roi):
            info = f"P{i + 1}: ({pt[0]:.3f}, {pt[1]:.3f})"
            cv2.putText(
                edit_frame,
                info,
                (10, 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return DetectionResult(
            frame=edit_frame,
            event=False,
            metadata={
                "type": "Chỉnh sửa khu vực làm việc",
                "severity": "low",
                "roi_edit_mode": True,
                "roi_points": normalized_roi,
                "roi_fallback_used": roi_fallback_used,
                "num_points": num_points,
                "frame_size": [frame_w, frame_h],
                "model_type": "workspace_monitor",
                "timestamp": time.strftime("%Y%m%d%H%M%S"),
                "roi_validation_error": roi_validation_error,
            },
        )

    # ──────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────

    def cleanup(self):
        """Release all resources."""
        if hasattr(self, "last_unknown_face_event"):
            self.last_unknown_face_event.clear()
        if hasattr(self, "_person_track_state"):
            self._person_track_state.clear()
        if hasattr(self, "face_scanned_tracks"):
            self.face_scanned_tracks.clear()
        if hasattr(self, "_auto_roi_candidate"):
            self._auto_roi_candidate = None
            self._auto_roi_candidate_count = 0
        if hasattr(self, "_auto_roi_last"):
            self._auto_roi_last = None
            self._auto_roi_last_confidence = 0.0
            self._auto_roi_locked = False
        if hasattr(self, "object_tracker"):
            self.object_tracker.reset()
        super().cleanup()
