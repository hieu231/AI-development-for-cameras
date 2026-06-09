# src/core/object_tracker.py
"""
Object tracking and event deduplication helpers.

- Prevents repeating the same violation for the same tracked object until the
  track expires from memory.
- Prevents bursty saves by enforcing a short per-object save cooldown.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """State for a tracked object."""

    track_id: int
    violations: Set[str] = field(default_factory=set)
    violation_last_saved: Dict[str, float] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_saved_event: float = 0.0


class ObjectTracker:
    """
    Track objects and decide whether a new event should be recorded.

    Rules:
    - The same `(track_id, violation_type)` is recorded once until the track is
      evicted after `reset_interval` seconds without being seen.
    - Different violations for the same track are still throttled by
      `save_cooldown` to avoid burst saves for the same object.
    """

    def __init__(self, reset_interval: int = 1800, save_cooldown: float = 2.0):
        self.reset_interval = reset_interval
        self.save_cooldown = float(save_cooldown)
        self.tracked_objects: Dict[int, TrackedObject] = {}

        logger.info(
            "ObjectTracker initialized | reset_interval=%ss | save_cooldown=%ss",
            reset_interval,
            self.save_cooldown,
        )

    def should_record_event(
        self,
        track_id: Optional[int],
        violation_type: str,
        force_save: bool = False,
        repeat_interval: Optional[float] = None,
    ) -> bool:
        """
        Return True if a detection event should be persisted.

        `force_save` bypasses only the short save cooldown. It does not bypass
        duplicate suppression for the same violation on the same tracked object.
        """
        if track_id is None:
            return True

        current_time = time.time()
        self._cleanup_expired_objects(current_time)

        if track_id not in self.tracked_objects:
            self.tracked_objects[track_id] = TrackedObject(track_id=track_id)
            logger.info("New tracked object: ID=%s", track_id)

        obj = self.tracked_objects[track_id]
        obj.last_seen = current_time

        if violation_type in obj.violations:
            last_saved_for_violation = obj.violation_last_saved.get(violation_type)
            if repeat_interval is not None:
                if repeat_interval <= 0:
                    obj.last_saved_event = current_time
                    obj.violation_last_saved[violation_type] = current_time
                    logger.info(
                        "Recorded repeated violation for track %s: %s",
                        track_id,
                        violation_type,
                    )
                    return True

            if repeat_interval is not None and last_saved_for_violation is not None:
                time_since_last_violation = current_time - last_saved_for_violation
                if time_since_last_violation >= repeat_interval:
                    obj.last_saved_event = current_time
                    obj.violation_last_saved[violation_type] = current_time
                    logger.info(
                        "Recorded repeated violation for track %s: %s",
                        track_id,
                        violation_type,
                    )
                    return True

            logger.debug(
                "Suppressed duplicate violation for track %s: %s",
                track_id,
                violation_type,
            )
            return False

        if not force_save:
            time_since_last_save = current_time - obj.last_saved_event
            if time_since_last_save < self.save_cooldown:
                logger.debug(
                    "Suppressed burst save for track %s: %.2fs remaining",
                    track_id,
                    self.save_cooldown - time_since_last_save,
                )
                return False

        obj.violations.add(violation_type)
        obj.last_saved_event = current_time
        obj.violation_last_saved[violation_type] = current_time
        logger.info("Recorded violation for track %s: %s", track_id, violation_type)
        return True

    def _cleanup_expired_objects(self, current_time: float) -> None:
        """Remove tracks that have not been seen within reset_interval."""
        expired = [
            track_id
            for track_id, obj in self.tracked_objects.items()
            if current_time - obj.last_seen >= self.reset_interval
        ]
        for track_id in expired:
            logger.info("Removing expired tracked object: ID=%s", track_id)
            del self.tracked_objects[track_id]

    def reset(self) -> None:
        count = len(self.tracked_objects)
        self.tracked_objects.clear()
        logger.info("ObjectTracker reset | cleared=%s", count)

    def get_tracked_count(self) -> int:
        return len(self.tracked_objects)

    def get_stats(self) -> dict:
        now = time.time()
        return {
            "total_tracked": len(self.tracked_objects),
            "save_cooldown_seconds": self.save_cooldown,
            "reset_interval_seconds": self.reset_interval,
            "active_objects": [
                {
                    "track_id": track_id,
                    "violations": list(obj.violations),
                    "last_saved_ago": round(now - obj.last_saved_event, 1),
                    "last_seen_ago": round(now - obj.last_seen, 1),
                }
                for track_id, obj in self.tracked_objects.items()
            ],
        }


class RecentViolationDeduplicator:
    """Short-lived dedup for tracker-ID jitter across adjacent frames."""

    def __init__(self, window_seconds: float = 2.0, iou_threshold: float = 0.5):
        self.window_seconds = float(window_seconds)
        self.iou_threshold = float(iou_threshold)
        self._recent: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize_bbox(
        bbox: Optional[Tuple[int, int, int, int] | List[int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None or len(bbox) < 4:
            return None
        try:
            return tuple(int(round(float(value))) for value in bbox[:4])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _compute_iou(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
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

    def _cleanup_locked(self, current_time: float) -> None:
        if self.window_seconds <= 0:
            self._recent.clear()
            return

        expired_types: List[str] = []
        for violation_type, entries in self._recent.items():
            fresh_entries = [
                entry
                for entry in entries
                if current_time - float(entry["seen_at"]) < self.window_seconds
            ]
            if fresh_entries:
                self._recent[violation_type] = fresh_entries
            else:
                expired_types.append(violation_type)

        for violation_type in expired_types:
            del self._recent[violation_type]

    def _find_match_locked(
        self,
        violation_type: str,
        track_id: Optional[int],
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Dict[str, Any]]:
        entries = self._recent.get(violation_type, [])
        matched_entry: Optional[Dict[str, Any]] = None
        best_iou = self.iou_threshold

        for entry in entries:
            entry_track_id = entry.get("track_id")
            if (
                track_id is not None
                and entry_track_id is not None
                and int(entry_track_id) == int(track_id)
            ):
                matched_entry = entry
                break

            entry_bbox = entry.get("bbox")
            if bbox is None or entry_bbox is None:
                continue

            iou = self._compute_iou(bbox, entry_bbox)
            if iou >= best_iou:
                matched_entry = entry
                best_iou = iou

        return matched_entry

    def is_recent_duplicate(
        self,
        violation_type: str,
        track_id: Optional[int],
        bbox: Optional[Tuple[int, int, int, int] | List[int]],
        current_time: Optional[float] = None,
    ) -> bool:
        if self.window_seconds <= 0:
            return False

        normalized_track_id = None if track_id is None else int(track_id)
        normalized_bbox = self._normalize_bbox(bbox)
        if normalized_track_id is None and normalized_bbox is None:
            return False

        now = time.time() if current_time is None else float(current_time)
        with self._lock:
            self._cleanup_locked(now)
            matched_entry = self._find_match_locked(
                violation_type,
                normalized_track_id,
                normalized_bbox,
            )
            if matched_entry is None:
                return False

            matched_entry["track_id"] = normalized_track_id
            matched_entry["bbox"] = normalized_bbox
            matched_entry["seen_at"] = now
            return True

    def remember_event(
        self,
        violation_type: str,
        track_id: Optional[int],
        bbox: Optional[Tuple[int, int, int, int] | List[int]],
        current_time: Optional[float] = None,
    ) -> None:
        if self.window_seconds <= 0:
            return

        normalized_track_id = None if track_id is None else int(track_id)
        normalized_bbox = self._normalize_bbox(bbox)
        if normalized_track_id is None and normalized_bbox is None:
            return

        now = time.time() if current_time is None else float(current_time)
        with self._lock:
            self._cleanup_locked(now)
            matched_entry = self._find_match_locked(
                violation_type,
                normalized_track_id,
                normalized_bbox,
            )
            if matched_entry is not None:
                matched_entry["track_id"] = normalized_track_id
                matched_entry["bbox"] = normalized_bbox
                matched_entry["seen_at"] = now
                return

            self._recent.setdefault(violation_type, []).append(
                {
                    "track_id": normalized_track_id,
                    "bbox": normalized_bbox,
                    "seen_at": now,
                }
            )
