"""
Face Recognition Model
Detects and recognizes faces in video streams.

Emits enter/exit events identical to the vehicle tracker pattern:
  - One ENTER event when a face (known or unknown) first appears.
  - One EXIT event when that face has been missing for >present_timeout
    seconds.
  - No per-frame spam in between.

Known faces are tracked by `profile_id`. Unknown faces are clustered by
cosine similarity (default 0.55) so the same stranger across frames is
treated as one identity rather than a new "Unknown" every few frames.
"""

import os
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
import time
from datetime import datetime, date, timedelta

from src.ai_models.base_model import BaseModel, DetectionResult
from src.face_recognition.face_engine import get_face_engine
from src.face_recognition.repository import FaceProfileRepository
from src.database import get_db
from src.utils.roi_utils import denormalize_roi, get_effective_roi


class FaceRecognitionModel(BaseModel):
    """Model for face detection and recognition in video streams"""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize Face Recognition Model

        Args:
            model_path: Not used (kept for compatibility)
            **kwargs: Additional parameters
                - face_similarity_threshold: Similarity threshold (default: 0.45)
                - roi_polygon: ROI as polygon [(x1,y1), (x2,y2), ...]
                - roi_rect: ROI as rectangle (x1, y1, x2, y2)
        """
        super().__init__(model_path, **kwargs)

        # Get face recognition engine
        self.engine = get_face_engine()
        if self.engine is None:
            raise RuntimeError("Face recognition is disabled or failed to initialize")

        # Parameters
        self.face_similarity_threshold = kwargs.get(
            "face_similarity_threshold",
            float(os.getenv("DEFAULT_FACE_SIMILARITY_THRESHOLD", "0.45")),
        )

        # Cooldown periods (in seconds)
        self.known_cooldown = int(
            os.getenv("FACE_RECOGNITION_COOLDOWN_KNOWN", "300")
        )  # 5 minutes
        self.unknown_cooldown = int(
            os.getenv("FACE_RECOGNITION_COOLDOWN_UNKNOWN", "60")
        )  # 1 minute

        # ── Enter/Exit tracker (matches the vehicle tracker pattern) ────
        # active_known_tracks  : profile_id -> dict(last_seen, employee_name, ...)
        # active_unknown_tracks: list of dicts with fields
        #                        embedding, last_seen, internal_id, first_seen
        # `present_timeout` is the gap (seconds) after which we consider a
        # face "left the scene" and emit an EXIT event. Defaults to 10s
        # to match the vehicle tracker.
        self.active_known_tracks: Dict[str, dict] = {}
        self.active_unknown_tracks: List[dict] = []
        self.present_timeout = int(
            os.getenv("FACE_PRESENT_TIMEOUT_SEC", "10")
        )
        # Cosine-similarity floor for clustering an unknown face across
        # frames as the SAME stranger. Without this every frame would
        # produce a "new" Unknown enter event because of embedding noise.
        self.unknown_dedup_similarity = float(
            os.getenv("FACE_RECOGNITION_UNKNOWN_DEDUP_SIM", "0.55")
        )
        # Counter for assigning internal IDs to unknown faces — gives each
        # stranger a stable identifier across frames + in events ("Người lạ #3").
        self._next_unknown_id = 1
        # Daily counters (reset at local midnight) so the dashboard can
        # show "N known + M unknown faces today" without table scans.
        self.daily_known_count = 0
        self.daily_unknown_count = 0
        self.daily_count_date = date.today()
        # Per-day dedup so the same person leaving + re-entering many times
        # increments the counter only on first sight of the day.
        self._counted_known_today: set = set()
        self._counted_unknown_today: set = set()

        # Legacy aliases — kept so any external code reading these still
        # works during the rollout. Will be removed in a follow-up.
        self.last_detected_known = self.active_known_tracks
        self.last_detected_unknown = self.active_unknown_tracks

        # Frame skip for performance (higher = more bbox updates per second)
        self.fps_limit = int(os.getenv("FPS_LIMIT", "30"))
        self.frame_delay = 1.0 / self.fps_limit if self.fps_limit > 0 else 0
        self.last_process_time = 0

        self.logger.info(
            f"FaceRecognitionModel initialized with threshold={self.face_similarity_threshold}"
        )

    def _should_process_frame(self) -> bool:
        """Check if enough time has passed to process next frame"""
        current_time = time.time()
        time_since_last = current_time - self.last_process_time

        if time_since_last >= self.frame_delay:
            self.last_process_time = current_time
            return True
        return False

    def _apply_roi(
        self,
        frame: np.ndarray,
        roi_rect: Optional[tuple] = None,
        roi_polygon: Optional[list] = None,
    ) -> np.ndarray:
        """
        Apply ROI mask to frame

        Args:
            frame: Input frame
            roi_rect: ROI as rectangle (x1, y1, x2, y2)
            roi_polygon: ROI as polygon [(x1,y1), (x2,y2), ...]

        Returns:
            Masked frame (areas outside ROI are blacked out)
        """
        h, w = frame.shape[:2]

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)

        if (
            roi_polygon is not None
            and isinstance(roi_polygon, (list, tuple))
            and len(roi_polygon) >= 3
        ):
            # Use polygon ROI
            polygon_pts = []
            for px, py in roi_polygon:
                ix = int(max(0, min(w - 1, px)))
                iy = int(max(0, min(h - 1, py)))
                polygon_pts.append((ix, iy))
            roi_poly_np = np.array(polygon_pts, dtype=np.int32)
            cv2.fillPoly(mask, [roi_poly_np], 255)

        elif roi_rect is not None:
            # Use rectangle ROI
            rx1, ry1, rx2, ry2 = roi_rect
            roi_x1 = int(max(0, min(w - 1, rx1)))
            roi_y1 = int(max(0, min(h - 1, ry1)))
            roi_x2 = int(max(0, min(w - 1, rx2)))
            roi_y2 = int(max(0, min(h - 1, ry2)))

            if roi_x2 < roi_x1:
                roi_x1, roi_x2 = roi_x2, roi_x1
            if roi_y2 < roi_y1:
                roi_y1, roi_y2 = roi_y2, roi_y1

            mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        else:
            # No ROI, use full frame
            mask[:] = 255

        # Apply mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

    def _prune_old_unknown_entries(self, current_time: datetime) -> None:
        """Legacy stub kept for any code path still calling it. Enter/exit
        tracker prunes via `_collect_exit_events` based on present_timeout.
        """
        return None

    def _maybe_reset_daily_face_counter(self) -> None:
        """Roll the daily known/unknown counters at local midnight."""
        today = date.today()
        if today != self.daily_count_date:
            self.daily_known_count = 0
            self.daily_unknown_count = 0
            self.daily_count_date = today
            self._counted_known_today.clear()
            self._counted_unknown_today.clear()
            self.logger.info(
                "Daily face counters reset for %s", today.isoformat()
            )

    def _collect_exit_events(self, now_ts: float) -> list:
        """Emit one EXIT violation per track that has been missing for
        longer than `present_timeout`. Removes the track from
        `active_*_tracks` so a re-appearance later this day fires a new
        ENTER (matching the vehicle tracker semantics).
        """
        exits: list = []

        # Known tracks
        expired_known: list = []
        for pid, track in list(self.active_known_tracks.items()):
            last_seen = track.get("last_seen", 0.0)
            if now_ts - last_seen <= self.present_timeout:
                continue
            expired_known.append(pid)
            dwell = max(
                0.0,
                last_seen - track.get("first_seen", last_seen),
            )
            exits.append(
                {
                    "violation_type": "face_exit",
                    "event_type": f"{track.get('employee_name')} rời khu vực",
                    "type": f"{track.get('employee_name')} rời khu vực",
                    "title": f"{track.get('employee_name')} rời khu vực",
                    "description": (
                        f"{track.get('employee_name')} rời khu vực "
                        f"(dwell: {dwell:.1f}s, max sim: "
                        f"{track.get('max_similarity', 0):.2f})"
                    ),
                    "person_kind": "known",
                    "employee_id": track.get("employee_id"),
                    "employee_name": track.get("employee_name"),
                    "similarity": float(track.get("max_similarity", 0.0)),
                    "bbox": None,
                    "direction": "exit",
                    "dwell_seconds": round(dwell, 2),
                }
            )
            self.logger.info(
                "Face EXIT (known): %s dwell=%.1fs",
                track.get("employee_name"),
                dwell,
            )
        for pid in expired_known:
            self.active_known_tracks.pop(pid, None)

        # Unknown tracks
        survivors: list = []
        for track in self.active_unknown_tracks:
            last_seen = track.get("last_seen", 0.0)
            if now_ts - last_seen <= self.present_timeout:
                survivors.append(track)
                continue
            dwell = max(
                0.0,
                last_seen - track.get("first_seen", last_seen),
            )
            internal_id = track.get("internal_id")
            closest_name = track.get("closest_employee_name")
            closest_sim = float(track.get("closest_similarity") or 0.0)
            exits.append(
                {
                    "violation_type": "face_exit",
                    "event_type": "Người lạ rời khu vực",
                    "type": "Người lạ rời khu vực",
                    "title": "Người lạ rời khu vực",
                    "description": (
                        f"Người lạ #{internal_id} rời khu vực "
                        f"(dwell: {dwell:.1f}s"
                        + (
                            f", gần nhất: {closest_name} @ {closest_sim:.2f})"
                            if closest_name
                            else ")"
                        )
                    ),
                    "person_kind": "unknown",
                    "internal_id": internal_id,
                    "employee_id": None,
                    "employee_name": None,
                    "closest_employee_name": closest_name,
                    "closest_similarity": closest_sim,
                    "bbox": None,
                    "direction": "exit",
                    "dwell_seconds": round(dwell, 2),
                }
            )
            self.logger.info(
                "Face EXIT (unknown #%s): dwell=%.1fs",
                internal_id,
                dwell,
            )
        self.active_unknown_tracks = survivors

        return exits

    def _should_trigger_event(
        self, profile_id: Optional[str], embedding: Optional[np.ndarray]
    ) -> bool:
        """
        Check if event should be triggered based on cooldown

        Args:
            profile_id: Profile ID if known face, None if unknown
            embedding: Face embedding (used for unknown faces)

        Returns:
            True if event should be triggered
        """
        current_time = datetime.now()

        if profile_id is not None:
            # Known face
            if profile_id in self.last_detected_known:
                last_time = self.last_detected_known[profile_id]
                if (current_time - last_time).total_seconds() < self.known_cooldown:
                    return False

            # Update last detection time
            self.last_detected_known[profile_id] = current_time
            return True

        else:
            # Unknown face — dedup against recently-seen unknown embeddings
            # using cosine similarity instead of a rounding-based hash. The
            # old hash flipped between adjacent frames because tiny
            # embedding noise crossed the rounding boundary, so every
            # frame ended up "new" and the cooldown never engaged.
            if embedding is None:
                return True

            self._prune_old_unknown_entries(current_time)
            emb = np.asarray(embedding, dtype=np.float32)
            for cached_emb, _ in self.last_detected_unknown:
                similarity = float(np.dot(emb, cached_emb))
                if similarity >= self.unknown_dedup_similarity:
                    # Same person we already emitted an event for within
                    # the cooldown window — refresh its last_seen and skip.
                    return False

            self.last_detected_unknown.append((emb, current_time))
            return True

        return True

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        """
        Process frame for face detection and recognition

        Args:
            frame: Input frame (BGR)
            **kwargs: Additional parameters
                - roi: ROI as normalized polygon with 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                       Each coordinate is in range [0,1]. Default: full frame [[0,0], [1,0], [1,1], [0,1]]
                - annotate: Whether to annotate frame (default: True)

        Returns:
            DetectionResult with standardized format
        """
        h, w = frame.shape[:2]

        # Get ROI - support both normalized and old pixel format for backward compatibility
        normalized_roi = get_effective_roi(kwargs.get("roi"))
        roi_polygon = None

        if normalized_roi:
            # New format: denormalize from [0,1] to pixel coordinates
            roi_polygon = denormalize_roi(normalized_roi, w, h)
        else:
            # Legacy format for backward compatibility
            roi_polygon = kwargs.get("roi_polygon", None)

        roi_rect = kwargs.get("roi_rect", None)
        annotate = kwargs.get("annotate", True)

        # Per-frame override from the camera's additional_parameters so
        # operators can tune via the UI / DB without restarting the
        # container. Falls back to the value captured at __init__ time
        # (which itself falls back to the env var).
        try:
            similarity_override = float(
                kwargs.get("face_similarity_threshold", self.face_similarity_threshold)
            )
            if 0.0 < similarity_override <= 1.0:
                self.face_similarity_threshold = similarity_override
        except (TypeError, ValueError):
            pass

        try:
            known_cd_override = int(
                kwargs.get("known_cooldown", self.known_cooldown)
            )
            if known_cd_override >= 0:
                self.known_cooldown = known_cd_override
        except (TypeError, ValueError):
            pass

        try:
            unknown_cd_override = int(
                kwargs.get("unknown_cooldown", self.unknown_cooldown)
            )
            if unknown_cd_override >= 0:
                self.unknown_cooldown = unknown_cd_override
        except (TypeError, ValueError):
            pass

        # Check if should process frame (FPS limit)
        if not self._should_process_frame():
            # Return frame=None so previous annotations stay in the buffer
            # (returning frame would overwrite annotated buffer with raw frame,
            #  causing bbox to flicker on/off on WebRTC)
            return DetectionResult(
                frame=None,
                event=False,
                metadata={"skipped": True, "reason": "FPS limit", "severity": "low"},
            )

        # Apply ROI
        masked_frame = self._apply_roi(frame, roi_rect, roi_polygon)

        # Detect and align faces using MTCNN
        # Returns aligned face tensors [3, 160, 160] ready for embedding
        # plus the source bboxes so the WebRTC overlay can draw rectangles.
        face_tensors, confidences, face_bboxes = self.engine.detector.detect(masked_frame)

        if face_tensors is None or len(face_tensors) == 0:
            # No faces detected
            annotated_frame = frame.copy() if annotate else None

            # Draw ROI if annotating
            if annotate and roi_polygon is not None:
                h, w = frame.shape[:2]
                polygon_pts = []
                for px, py in roi_polygon:
                    ix = int(max(0, min(w - 1, px)))
                    iy = int(max(0, min(h - 1, py)))
                    polygon_pts.append((ix, iy))
                roi_poly_np = np.array(polygon_pts, dtype=np.int32)
                cv2.polylines(annotated_frame, [roi_poly_np], True, (255, 255, 0), 2)

            return DetectionResult(
                frame=annotated_frame,
                event=False,
                metadata={"num_faces": 0, "severity": "low"},
            )

        # Generate embeddings from aligned face tensors
        embeddings = self.engine.embedder.embed(face_tensors)

        # Search database for matches
        db = next(get_db())
        repo = FaceProfileRepository(db)

        detected_faces: list = []
        face_violations: list = []  # ENTER + EXIT events emitted this frame
        now = datetime.now()
        now_ts = time.time()
        self._maybe_reset_daily_face_counter()

        for i, (embedding, confidence) in enumerate(zip(embeddings, confidences)):
            # Always retrieve the closest profile (threshold=0) so we can
            # surface the nearest known identity even when the cosine
            # similarity is below the match threshold. Operators want to
            # see "closest: Hieu 0.38" on the overlay so they can tune the
            # threshold from real data instead of guessing.
            top_candidates = repo.search_similar_faces(
                embedding.tolist(),
                threshold=0.0,
                limit=1,
                active_only=True,
            )
            closest_profile = top_candidates[0][0] if top_candidates else None
            closest_similarity = (
                float(top_candidates[0][1]) if top_candidates else 0.0
            )

            match_result = (
                (closest_profile, closest_similarity)
                if closest_profile is not None
                and closest_similarity >= self.face_similarity_threshold
                else None
            )

            bbox = face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None

            if match_result is not None:
                # ── Known face: track by profile_id ─────────────────────
                profile, similarity = match_result
                pid = str(profile.id)
                track = self.active_known_tracks.get(pid)
                is_new = track is None
                if is_new:
                    track = {
                        "profile_id": pid,
                        "employee_id": profile.employee_id,
                        "employee_name": profile.employee_name,
                        "first_seen": now_ts,
                        "max_similarity": float(similarity),
                    }
                    self.active_known_tracks[pid] = track
                    if pid not in self._counted_known_today:
                        self._counted_known_today.add(pid)
                        self.daily_known_count += 1
                    face_violations.append(
                        {
                            "violation_type": "face_enter",
                            "event_type": f"{profile.employee_name} vào khu vực",
                            "type": f"{profile.employee_name} vào khu vực",
                            "title": f"{profile.employee_name} vào khu vực",
                            "description": (
                                f"Nhận diện {profile.employee_name} "
                                f"(similarity {similarity:.2f})"
                            ),
                            "person_kind": "known",
                            "employee_id": profile.employee_id,
                            "employee_name": profile.employee_name,
                            "similarity": float(similarity),
                            "confidence": float(confidence),
                            "bbox": bbox,
                            "direction": "enter",
                            "known_today": self.daily_known_count,
                        }
                    )
                    self.logger.info(
                        "Face ENTER (known): %s sim=%.2f known_today=%d",
                        profile.employee_name,
                        similarity,
                        self.daily_known_count,
                    )
                track["last_seen"] = now_ts
                track["max_similarity"] = max(
                    track.get("max_similarity", 0.0), float(similarity),
                )
                detected_faces.append(
                    {
                        "type": "known",
                        "employee_id": profile.employee_id,
                        "employee_name": profile.employee_name,
                        "similarity": float(similarity),
                        "confidence": float(confidence),
                        "face_index": i,
                        "bbox": bbox,
                    }
                )

            else:
                # ── Unknown face (người lạ): cluster by cosine similarity ─
                emb = np.asarray(embedding, dtype=np.float32)
                matched_track = None
                best_sim = -1.0
                for track in self.active_unknown_tracks:
                    cached = track.get("embedding")
                    if cached is None:
                        continue
                    sim = float(np.dot(emb, cached))
                    if sim > best_sim and sim >= self.unknown_dedup_similarity:
                        best_sim = sim
                        matched_track = track

                if matched_track is None:
                    internal_id = self._next_unknown_id
                    self._next_unknown_id += 1
                    matched_track = {
                        "internal_id": internal_id,
                        "embedding": emb,
                        "first_seen": now_ts,
                        "closest_employee_name": (
                            closest_profile.employee_name
                            if closest_profile is not None
                            else None
                        ),
                        "closest_similarity": closest_similarity,
                    }
                    self.active_unknown_tracks.append(matched_track)
                    self.daily_unknown_count += 1
                    self._counted_unknown_today.add(internal_id)
                    face_violations.append(
                        {
                            "violation_type": "face_enter",
                            "event_type": "Người lạ vào khu vực",
                            "type": "Người lạ vào khu vực",
                            "title": "Người lạ vào khu vực",
                            "description": (
                                f"Người lạ #{internal_id} vào khu vực"
                                + (
                                    f" (gần nhất: {closest_profile.employee_name} "
                                    f"@ {closest_similarity:.2f})"
                                    if closest_profile is not None
                                    else ""
                                )
                            ),
                            "person_kind": "unknown",
                            "employee_id": None,
                            "employee_name": None,
                            "internal_id": internal_id,
                            "confidence": float(confidence),
                            "closest_employee_name": (
                                closest_profile.employee_name
                                if closest_profile is not None
                                else None
                            ),
                            "closest_similarity": closest_similarity,
                            "bbox": bbox,
                            "direction": "enter",
                            "unknown_today": self.daily_unknown_count,
                        }
                    )
                    self.logger.info(
                        "Face ENTER (unknown #%d): closest=%s @ %.3f, "
                        "unknown_today=%d",
                        internal_id,
                        closest_profile.employee_name if closest_profile else "-",
                        closest_similarity,
                        self.daily_unknown_count,
                    )

                matched_track["last_seen"] = now_ts
                # Refresh embedding with running average so the track
                # doesn't drift if the person rotates / changes lighting.
                cached = matched_track.get("embedding")
                if cached is not None:
                    matched_track["embedding"] = (
                        cached * 0.9 + emb * 0.1
                    ).astype(np.float32)

                # Refresh closest-profile metadata each frame so the
                # overlay shows THIS frame's similarity instead of the
                # value cached at track creation (which sticks at 0.0
                # if the first sighting was a bad angle). Also remember
                # the best similarity ever seen on this track for the
                # eventual EXIT event.
                if closest_profile is not None:
                    matched_track["closest_employee_name"] = (
                        closest_profile.employee_name
                    )
                matched_track["closest_similarity"] = closest_similarity
                matched_track["max_closest_similarity"] = max(
                    float(matched_track.get("max_closest_similarity", 0.0)),
                    closest_similarity,
                )

                detected_faces.append(
                    {
                        "type": "unknown",
                        "employee_id": None,
                        "employee_name": None,
                        "similarity": 0.0,
                        "confidence": float(confidence),
                        "face_index": i,
                        "bbox": bbox,
                        "internal_id": matched_track["internal_id"],
                        # Use THIS frame's values so the overlay reflects
                        # the live similarity, not the stale cached one.
                        "closest_employee_name": (
                            closest_profile.employee_name
                            if closest_profile is not None
                            else matched_track.get("closest_employee_name")
                        ),
                        "closest_similarity": closest_similarity,
                    }
                )

        # ── EXIT events: prune tracks that haven't been seen for >timeout ─
        face_violations.extend(self._collect_exit_events(now_ts))

        event_triggered = bool(face_violations)

        # Annotate frame if requested
        annotated_frame = None
        if annotate:
            annotated_frame = frame.copy()

            # Draw ROI
            if roi_polygon is not None:
                h, w = frame.shape[:2]
                polygon_pts = []
                for px, py in roi_polygon:
                    ix = int(max(0, min(w - 1, px)))
                    iy = int(max(0, min(h - 1, py)))
                    polygon_pts.append((ix, iy))
                roi_poly_np = np.array(polygon_pts, dtype=np.int32)
                cv2.polylines(annotated_frame, [roi_poly_np], True, (255, 255, 0), 2)

            # Draw faces — rectangle around each detected bbox plus a label
            # with the employee name (or "Unknown") and similarity score.
            # The "closest profile" hint only shows when the similarity is
            # genuinely near the match threshold (within 0.05) — otherwise
            # operators see "?Hieu 0.05" on every random face just because
            # Hieu happens to be the only registered profile.
            CLOSE_MATCH_HINT_BAND = 0.05  # surface hint within this of threshold

            for face_data in detected_faces:
                bbox = face_data.get("bbox")
                if face_data["type"] == "known":
                    label = (
                        f"{face_data['employee_name']} "
                        f"({face_data['similarity']:.2f})"
                    )
                    color = (0, 255, 0)
                else:
                    # Overlay labels use ASCII only — cv2.putText uses
                    # Hershey fonts which can't render Vietnamese
                    # diacritics, so "Người lạ" came out as "Ng?????i
                    # l???". The DB event metadata still uses Vietnamese
                    # strings (dashboard handles them fine with proper
                    # fonts); only the live video overlay is ASCII.
                    closest_name = face_data.get("closest_employee_name")
                    closest_sim = float(face_data.get("closest_similarity") or 0.0)
                    internal_id = face_data.get("internal_id")
                    hint_floor = max(
                        0.0,
                        self.face_similarity_threshold - CLOSE_MATCH_HINT_BAND,
                    )
                    id_suffix = f"#{internal_id}" if internal_id is not None else ""
                    if closest_name and closest_sim >= hint_floor:
                        # Yellow: nearest profile is just under the match
                        # threshold.
                        label = f"?{closest_name} {closest_sim:.2f}"
                        color = (0, 255, 255)
                    elif closest_name and closest_sim > 0.0:
                        label = (
                            f"Unknown {id_suffix} "
                            f"(~{closest_name} {closest_sim:.2f})"
                        )
                        color = (0, 0, 255)
                    else:
                        label = f"Unknown {id_suffix}".rstrip()
                        color = (0, 0, 255)

                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
                    )
                    y_label = max(th + 4, y1)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y_label - th - 4),
                        (x1 + tw + 6, y_label),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 3, y_label - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                else:
                    # Defensive: keep the legacy top-left text if a bbox is
                    # missing for any reason (shouldn't happen).
                    cv2.putText(
                        annotated_frame,
                        label,
                        (10, 30 + face_data["face_index"] * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            # Live counter overlay — ASCII only (cv2.putText Hershey font
            # can't render "Người lạ"). Dashboard shows the Vietnamese
            # version via event metadata.
            counter_label = (
                f"Known: {self.daily_known_count}  |  "
                f"Unknown: {self.daily_unknown_count}"
            )
            h_img = annotated_frame.shape[0]
            cv2.putText(
                annotated_frame,
                counter_label,
                (10, h_img - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        db.close()

        # Build a `detections` list shaped like every other model so
        # SingleThreadProcessor.compose_annotations can redraw the boxes
        # on the WebRTC overlay every frame, not just on event-trigger
        # frames. Each entry needs `bbox` (x1,y1,x2,y2), `label`,
        # `class_name`, `confidence`.
        overlay_detections = []
        for face_data in detected_faces:
            bbox = face_data.get("bbox")
            if not bbox:
                continue
            if face_data["type"] == "known":
                lbl = (
                    f"{face_data.get('employee_name','?')} "
                    f"({face_data.get('similarity',0):.2f})"
                )
            else:
                lbl = "Unknown"
            overlay_detections.append(
                {
                    "bbox": bbox,
                    "label": lbl,
                    "class_name": lbl,
                    "confidence": float(face_data.get("confidence", 0.0)),
                }
            )

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata={
                "num_faces": len(detected_faces),
                "severity": "high" if event_triggered else "low",
                "model_type": "face_recognition",
                "detected_faces": detected_faces,
                "detections": overlay_detections,
                # Enter/exit events go through SingleThreadProcessor's
                # batch-save path so each appearance / departure becomes
                # ONE DB row — no per-frame spam, identical to the
                # vehicle tracker pattern.
                "violations": face_violations,
                "known_today": self.daily_known_count,
                "unknown_today": self.daily_unknown_count,
                "active_known_count": len(self.active_known_tracks),
                "active_unknown_count": len(self.active_unknown_tracks),
                "timestamp": datetime.now().isoformat(),
            },
        )

    def cleanup(self):
        """Clean up resources"""
        # Clear enter/exit tracker caches
        self.active_known_tracks.clear()
        self.active_unknown_tracks = []
        self._counted_known_today.clear()
        self._counted_unknown_today.clear()
        # Legacy aliases (kept pointed at the live structures elsewhere)
        # are now empty since active_* were cleared above.
        super().cleanup()
