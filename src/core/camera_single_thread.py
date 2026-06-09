"""
Single Thread Processor - AI inference processor for camera frames
Uses standardized DetectionResult format
"""
import logging
import threading
import time
import cv2
import numpy as np
import asyncio
from uuid import UUID
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from src.database import SessionLocal
from src.models.camera_model import CameraModel
from src.models.camera import Camera
from src.models.ai_model import AiModel
from src.models.event import Event
from src.core.model_factory import ModelFactory
from src.ai_models.base_model import BaseModel, DetectionResult
from src.core.event_storage import event_storage_service
from src.core.websocket_manager import websocket_manager
from src.utils.parameter_defaults import get_default_additional_parameters
from src.utils.alert_levels import AlertLevel
from src.utils.datetime_utils import serialize_utc_datetime

logger = logging.getLogger(__name__)


class SingleThreadProcessor:
    """Process AI inference for frames from a camera"""

    # Per-model-type color for the composite annotation overlay drawn by the
    # orchestrator. Keeps the stream frame colored by which model produced
    # the detection, without trusting any individual model to draw correctly.
    _MODEL_TYPE_COLORS: Dict[str, Tuple[int, int, int]] = {
        'helmet_detection': (0, 165, 255),       # orange
        'smoke_fire': (0, 0, 255),               # red
        'oil_spill': (255, 128, 0),              # blue
        'oil_dumping': (255, 0, 128),            # purple
        'oil_cap_detection': (255, 200, 0),      # light blue
        'tran_dau': (0, 140, 255),               # amber
        'people_control': (0, 255, 255),         # yellow
        'face_recognition': (255, 0, 255),       # magenta
        'alpr': (0, 255, 128),                   # light green
        'workspace_monitor': (255, 255, 0),      # cyan
        'smoking_behavior': (0, 215, 255),       # amber
        'evn_smartech': (128, 255, 0),           # lime
        'yolov8': (200, 200, 200),               # gray
        'yolov11': (200, 200, 200),              # gray
        'object_detection': (200, 200, 200),     # gray
    }
    _DEFAULT_OVERLAY_COLOR: Tuple[int, int, int] = (0, 255, 0)
    _EVENT_DUPLICATE_WINDOW_SECONDS: float = 2.0
    _EVENT_DUPLICATE_CACHE_TTL_SECONDS: float = 30.0
    _EVENT_DUPLICATE_BBOX_QUANTIZATION_PX: int = 32

    def __init__(self, camera_id: UUID):
        self.camera_id = camera_id
        self.models: Dict[str, Dict[str, Any]] = {}
        # Guards self.models against concurrent reload (FastAPI / MQTT threads)
        # vs. processor-thread inference. Use RLock so the same thread can
        # recurse (e.g. cleanup called from load_models while iterating).
        self._models_lock = threading.RLock()
        self._recent_event_lock = threading.Lock()
        self._recent_event_cache: Dict[Tuple[Any, ...], float] = {}
        self.load_models()

    def load_models(self):
        """Load enabled models for this camera from database.

        Thread-safe: holds ``_models_lock`` for the entire swap so that
        ``process_frame`` running on the processor thread never observes a
        half-populated dict and never calls ``process_frame`` on a model
        whose ``cleanup()`` is concurrently being executed from an API
        worker thread.
        """
        db = SessionLocal()
        try:
            camera_models = db.query(CameraModel).filter(
                CameraModel.camera_id == self.camera_id,
                CameraModel.is_enabled == True
            ).all()

            new_models: Dict[str, Dict[str, Any]] = {}

            # Build the new set of models FIRST (outside the lock) so that
            # slow YOLO weight loading does not block the processor thread.
            for camera_model in camera_models:
                ai_model = camera_model.ai_model
                if ai_model is None:
                    logger.warning(
                        "Camera %s has an enabled camera_model %s without a linked ai_model; skipping",
                        self.camera_id,
                        camera_model.id,
                    )
                    continue
                if not ai_model.is_active:
                    continue

                default_additional_params = get_default_additional_parameters(ai_model.model_type)
                configured_additional_params = camera_model.additional_parameters or {}
                merged_additional_params = {
                    **default_additional_params,
                    **configured_additional_params,
                }

                # Prepare model info
                model_info = {
                    'model_id': ai_model.id,
                    'model_name': ai_model.name,
                    'model_type': ai_model.model_type,
                    'model_path': ai_model.model_path,
                    'parameters': ai_model.parameters or {},
                    'additional_params': merged_additional_params
                }

                # Create model using factory
                model_instance = ModelFactory.create_model(model_info)

                if model_instance:
                    new_models[str(ai_model.id)] = {
                        'model_instance': model_instance,
                        'model_id': ai_model.id,
                        'model_name': ai_model.name,
                        'model_type': ai_model.model_type,
                        'parameters': ai_model.parameters or {},
                        'additional_params': merged_additional_params,
                    }
                    logger.info(
                        f"Loaded model {ai_model.name} ({ai_model.model_type}) for camera {self.camera_id}"
                    )
                else:
                    logger.error(f"Failed to create model {ai_model.name}")

            # Atomically swap the active set. The cleanup of old model
            # instances MUST stay inside the lock: ``process_frame`` snapshots
            # ``self.models`` under the same lock, and if we dropped the lock
            # before cleanup an in-flight snapshot could still be calling
            # ``process_frame`` on an instance whose underlying YOLO weights
            # and Ultralytics tracker threads were being torn down from
            # another thread. That race was also a thread-leak source because
            # a half-cleaned-up YOLO instance would leave its tracker workers
            # dangling.
            with self._models_lock:
                old_models = self.models
                self.models = new_models

                for model_key, model_info in old_models.items():
                    instance = model_info.get('model_instance')
                    if instance is None:
                        continue
                    try:
                        instance.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up model {model_key}: {e}")

        finally:
            db.close()

    def process_frame(self, frame: np.ndarray) -> Dict[str, DetectionResult]:
        """
        Process frame with all loaded models.

        Isolation rules enforced by this orchestrator:

        1. **Private frame per model.** Each model receives its OWN deep copy
           of the clean, un-annotated input frame. Two of the existing models
           (``people_control_model``, ``workspace_monitor_model``) mutate the
           input frame in place; without this copy, the first model to draw
           corrupts inference for every model that runs after it.

        2. **No frame chaining.** We do NOT feed one model's annotated output
           to the next model. The old behaviour (``current_frame = result.frame``)
           was the primary reason multi-model camera threads "failed
           completely" — subsequent models had to run inference on pixels
           already covered in rectangles and text from earlier models.

        3. **Locked snapshot.** We take a tuple snapshot under ``_models_lock``
           so that an API-triggered ``reload_models()`` cannot tear down a
           model instance while it is mid-inference on this thread.

        4. **Per-model error containment.** An exception in one model produces
           a sentinel error result and does NOT prevent other models from
           running on the same frame or from emitting their own events.

        Args:
            frame: Input frame (numpy array, BGR). Not mutated by this method.

        Returns:
            Dictionary of model_key -> DetectionResult (one entry per loaded
            model, including error sentinels for models that raised).
        """
        results: Dict[str, DetectionResult] = {}

        # Snapshot under the lock so reload_models() cannot swap the dict or
        # call cleanup() on a model we're about to invoke.
        with self._models_lock:
            models_snapshot: List[Tuple[str, Dict[str, Any]]] = list(self.models.items())

        if not models_snapshot:
            return results

        for model_key, model_info in models_snapshot:
            model_name = model_info.get('model_name', model_key)
            try:
                model_instance: BaseModel = model_info['model_instance']
                additional_params = model_info.get('additional_params', {}) or {}
                process_params = dict(additional_params)
                if (
                    str(model_info.get('model_type') or '').lower() == 'smoke_fire'
                    and 'inference_frame' not in process_params
                ):
                    process_params['inference_frame'] = frame

                # Private, clean copy for this model ONLY. Required because
                # some models draw on the input in-place (see class docstring).
                model_input = frame.copy()

                detection_result: DetectionResult = model_instance.process_frame(
                    model_input, **process_params
                )

                if detection_result is None:
                    # Defensive: a model must return a DetectionResult. Treat
                    # None as a no-op so downstream compositing / broadcast
                    # still have a consistent slot for this model.
                    detection_result = DetectionResult(
                        frame=None,
                        event=False,
                        metadata={
                            'model': model_name,
                            'model_type': model_info.get('model_type'),
                            'detections': [],
                            'error': 'model returned None',
                        },
                    )

                results[model_key] = detection_result

                # Auto-save event if triggered. Each model's event is handled
                # independently — one model failing to emit does NOT block
                # the next model's event.
                if detection_result.event:
                    # Save this model's OWN annotated frame as evidence, if it
                    # produced one. Falls back to the clean input so we never
                    # save a frame contaminated by other models' drawings.
                    frame_to_save = (
                        detection_result.frame
                        if detection_result.frame is not None
                        else frame
                    )

                    try:
                        self._emit_model_events(
                            model_info=model_info,
                            detection_result=detection_result,
                            frame_to_save=frame_to_save,
                        )
                    except Exception as emit_exc:
                        logger.error(
                            "Camera %s: Model %s event emission failed - %s",
                            self.camera_id,
                            model_name,
                            emit_exc,
                            exc_info=True,
                        )

            except Exception as e:
                logger.error(
                    f"Camera {self.camera_id}: Model {model_name} error - {e}",
                    exc_info=True,
                )
                results[model_key] = DetectionResult(
                    frame=None,
                    event=False,
                    metadata={
                        'model': model_name,
                        'model_type': model_info.get('model_type'),
                        'detections': [],
                        'error': str(e),
                    },
                )

        return results

    @staticmethod
    def _normalize_event_bbox(
        bbox: Any,
    ) -> Optional[Tuple[int, int, int, int]]:
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            return tuple(int(round(float(value))) for value in bbox[:4])
        except (TypeError, ValueError):
            return None

    def _quantize_event_bbox(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if bbox is None:
            return None

        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        step = max(1, int(self._EVENT_DUPLICATE_BBOX_QUANTIZATION_PX))

        def _q(value: float) -> int:
            return int(round(value / step) * step)

        return (_q(cx), _q(cy), _q(w), _q(h))

    def _build_event_dedup_key(
        self,
        model_id: UUID,
        metadata: Dict[str, Any],
    ) -> Optional[Tuple[Any, ...]]:
        event_type = str(metadata.get('eventType') or metadata.get('type') or '').strip()
        violation = str(
            metadata.get('violation')
            or metadata.get('raw_violation_type')
            or metadata.get('violation_type')
            or ''
        ).strip()
        track_id = metadata.get('track_id')
        track_key = None if track_id is None else str(track_id).strip()
        if track_key == '':
            track_key = None
        bbox_key = self._quantize_event_bbox(
            self._normalize_event_bbox(metadata.get('bbox'))
        )
        plate_key = str(
            metadata.get('plate_number')
            or metadata.get('license_plate')
            or ''
        ).strip()
        person_key = str(
            metadata.get('person_id')
            or metadata.get('employee_id')
            or ''
        ).strip()

        identity_kind: Optional[str] = None
        identity_value: Any = None

        if plate_key:
            identity_kind = 'plate'
            identity_value = plate_key
        elif person_key:
            identity_kind = 'person'
            identity_value = person_key
        elif track_key is not None:
            # Track ID is more stable than per-frame bbox coordinates.
            identity_kind = 'track'
            identity_value = track_key
        elif bbox_key is not None:
            identity_kind = 'bbox'
            identity_value = bbox_key
        else:
            # Last-resort protection for models that omit track/bbox/person/plate.
            coarse_hint = str(
                metadata.get('title')
                or metadata.get('description')
                or ''
            ).strip()
            if not event_type and not violation and not coarse_hint:
                return None
            identity_kind = 'coarse'
            identity_value = coarse_hint

        return (
            str(self.camera_id),
            str(model_id),
            event_type,
            violation,
            identity_kind,
            identity_value,
        )

    def _prune_recent_event_cache_locked(self, current_time: float) -> None:
        expired_keys = [
            key
            for key, saved_at in self._recent_event_cache.items()
            if current_time - saved_at >= self._EVENT_DUPLICATE_CACHE_TTL_SECONDS
        ]
        for key in expired_keys:
            self._recent_event_cache.pop(key, None)

    def _reserve_recent_event_slot(
        self,
        model_id: UUID,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, Optional[Tuple[Any, ...]]]:
        dedup_key = self._build_event_dedup_key(model_id, metadata)
        if dedup_key is None:
            return True, None

        current_time = time.monotonic()
        with self._recent_event_lock:
            self._prune_recent_event_cache_locked(current_time)
            last_saved_at = self._recent_event_cache.get(dedup_key)
            if (
                last_saved_at is not None
                and current_time - last_saved_at < self._EVENT_DUPLICATE_WINDOW_SECONDS
            ):
                return False, dedup_key
            self._recent_event_cache[dedup_key] = current_time

        return True, dedup_key

    def _release_recent_event_slot(
        self,
        dedup_key: Optional[Tuple[Any, ...]],
    ) -> None:
        if dedup_key is None:
            return
        with self._recent_event_lock:
            self._recent_event_cache.pop(dedup_key, None)

    def _emit_model_events(
        self,
        model_info: Dict[str, Any],
        detection_result: DetectionResult,
        frame_to_save: np.ndarray,
    ) -> None:
        """Emit one-or-more events for a single model's DetectionResult.

        Split out from ``process_frame`` so that an emission failure on one
        model is isolated to that model (see caller's try/except).
        """
        metadata = detection_result.metadata or {}
        violations = metadata.get('violations', []) or []
        model_name = model_info.get('model_name', 'unknown')

        if len(violations) > 1:
            # Multiple violations — create one DB event per violation so the
            # evidence trail shows each one separately. Helmet model relies
            # on this for multi-person scenes.
            emitted_violation_keys = set()
            for violation in violations:
                violation_metadata = dict(metadata)
                violation_type = (
                    violation.get('violation_type')
                    or violation.get('type')
                    or violation.get('raw_type')
                    or 'violation'
                )
                violation_metadata['violation'] = violation_type
                violation_metadata['confidence'] = violation.get('confidence')
                violation_metadata['track_id'] = violation.get('track_id')
                violation_metadata['bbox'] = violation.get('bbox')
                violation_event_type = violation.get('event_type')
                if violation_event_type:
                    violation_metadata['type'] = violation_event_type
                    violation_metadata['eventType'] = violation_event_type
                    violation_metadata['title'] = violation_event_type

                violation_description = violation.get('description')
                if violation_description:
                    violation_metadata['description'] = violation_description

                violation_dedup_key = self._build_event_dedup_key(
                    model_info['model_id'],
                    violation_metadata,
                )
                if (
                    violation_dedup_key is not None
                    and violation_dedup_key in emitted_violation_keys
                ):
                    logger.info(
                        f"Camera {self.camera_id}: Suppressed duplicate violation in batch "
                        f"for {model_name}, violation={violation_type}"
                    )
                    continue
                if violation_dedup_key is not None:
                    emitted_violation_keys.add(violation_dedup_key)

                self.save_event(
                    model_id=model_info['model_id'],
                    metadata=violation_metadata,
                    image_frame=frame_to_save,
                )
                logger.info(
                    f"Camera {self.camera_id}: Event saved for {model_name}, "
                    f"violation={violation_type}"
                )
        else:
            self.save_event(
                model_id=model_info['model_id'],
                metadata=metadata,
                image_frame=frame_to_save,
            )
            logger.info(
                f"Camera {self.camera_id}: Event saved for {model_name}, "
                f"type={metadata.get('type')}"
            )

    def compose_annotations(
        self,
        base_frame: np.ndarray,
        results: Dict[str, DetectionResult],
    ) -> np.ndarray:
        """Draw every model's detections onto a fresh copy of the clean frame.

        We redraw from ``metadata["detections"]`` instead of layering each
        model's returned ``result.frame``, because:

        - Each model now infers on its own private copy, so their annotated
          frames cannot be directly composited pixel-wise.
        - Redrawing from metadata is cheap and gives the orchestrator full
          control over per-model-type color so operators can visually tell
          which model fired.

        Models that don't expose a ``detections`` list in metadata are
        simply skipped by the overlay — their events still fire normally.
        """
        composite = base_frame.copy()

        for model_key, result in results.items():
            if result is None:
                continue
            metadata = result.metadata or {}
            if metadata.get('error'):
                # Surface the failing model to operators without faking detections.
                cv2.putText(
                    composite,
                    f"{metadata.get('model', model_key)}: ERROR",
                    (10, 25 + 25 * (list(results.keys()).index(model_key))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                continue

            model_type = metadata.get('model_type') or metadata.get('model') or model_key
            color = self._MODEL_TYPE_COLORS.get(
                str(model_type).lower(),
                self._DEFAULT_OVERLAY_COLOR,
            )

            for detection in metadata.get('detections', []) or []:
                bbox = detection.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                except (TypeError, ValueError):
                    continue

                cv2.rectangle(composite, (x1, y1), (x2, y2), color, 2)

                label = str(
                    detection.get('class_name')
                    or detection.get('label')
                    or model_type
                )
                conf = detection.get('confidence')
                if conf is not None:
                    try:
                        label += f" {float(conf):.2f}"
                    except (TypeError, ValueError):
                        pass
                track_id = detection.get('track_id')
                if track_id is not None:
                    label += f" ID:{track_id}"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                y_label = max(th + 4, y1)
                cv2.rectangle(
                    composite,
                    (x1, y_label - th - 4),
                    (x1 + tw + 6, y_label),
                    color,
                    -1,
                )
                cv2.putText(
                    composite,
                    label,
                    (x1 + 3, y_label - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

        return composite

    def save_event(
        self,
        model_id: UUID,
        metadata: Dict[str, Any],
        image_frame: Optional[np.ndarray] = None
    ):
        """
        Save detection event to database with image

        Args:
            model_id: ID of the model that detected the event
            metadata: Detection metadata
            image_frame: Optional annotated frame to save as evidence
        """
        can_save, dedup_key = self._reserve_recent_event_slot(model_id, metadata)
        if not can_save:
            logger.info(
                f"[SAVE_EVENT] Suppressed duplicate event for camera {self.camera_id}, "
                f"model {model_id}, type={metadata.get('type')}, violation={metadata.get('violation')}"
            )
            return

        logger.info(f"[SAVE_EVENT] Starting save_event for camera {self.camera_id}, model {model_id}")
        
        db = SessionLocal()
        try:
            camera_state = (
                db.query(Camera)
                .filter(Camera.id == self.camera_id)
                .first()
            )
            if (
                camera_state is None
                or bool(getattr(camera_state, "is_deleted", False))
                or not bool(getattr(camera_state, "status", False))
            ):
                self._release_recent_event_slot(dedup_key)
                logger.info(
                    "[SAVE_EVENT] Skip event for inactive camera %s "
                    "(exists=%s, deleted=%s, status=%s)",
                    self.camera_id,
                    camera_state is not None,
                    bool(getattr(camera_state, "is_deleted", False))
                    if camera_state is not None
                    else None,
                    bool(getattr(camera_state, "status", False))
                    if camera_state is not None
                    else None,
                )
                return

            # Get current time for event as naive datetime (local time, no timezone)
            # This ensures both filename and database show the same local time
            event_time_local = datetime.now()
            alert_data = metadata.get("alert") if isinstance(metadata.get("alert"), dict) else {}
            alert_level = AlertLevel.from_value(
                metadata.get("alert_level")
                or alert_data.get("level")
                or metadata.get("level")
                or metadata.get("severity"),
                AlertLevel.LOW,
            )
            
            logger.info(f"[SAVE_EVENT] Event time (local, naive): {event_time_local}, has_frame: {image_frame is not None}")

            # Save image if provided (uses local time for filename)
            image_path = None
            if image_frame is not None:
                logger.info(f"[SAVE_EVENT] Saving image for camera {self.camera_id}...")
                image_path = event_storage_service.save_event_image(
                    camera_id=self.camera_id,
                    frame=image_frame,
                    event_time=event_time_local
                )

                if image_path:
                    logger.info(f"[SAVE_EVENT] Image saved to: {image_path}")
                else:
                    logger.warning(f"[SAVE_EVENT] Failed to save image, continuing without image")
            else:
                logger.warning(f"[SAVE_EVENT] No image frame provided!")

            # Create event record with naive datetime (local time)
            logger.info(f"[SAVE_EVENT] Creating event record in database...")
            event = Event(
                camera_id=self.camera_id,
                model_id=model_id,
                detection_data=metadata,
                image_path=image_path,
                time=event_time_local,
                alert_level=alert_level.value,
                ready_for_mqtt_deployment=True  # Mark as ready for MQTT deployment
            )
            db.add(event)
            logger.info(f"[SAVE_EVENT] Event added to session (time in DB will be: {event.time}), committing...")
            db.commit()
            db.refresh(event)
            logger.info(f"[SAVE_EVENT] ✓ Event committed successfully! ID: {event.id}, DB time: {event.time}, Image: {event.image_path}")

            logger.info(f"[SAVE_EVENT] SUCCESS - Camera {self.camera_id}, model {model_id}, type: {metadata.get('type')}, image: {bool(image_path)}")

            # Broadcast event via WebSocket
            self._broadcast_event(event, db)

        except Exception as e:
            self._release_recent_event_slot(dedup_key)
            logger.error(f"[SAVE_EVENT] ✗ FAILED to save event: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()

    def _broadcast_event(self, event: Event, db: Session):
        """
        Broadcast event to WebSocket clients

        Args:
            event: Event object from database
            db: Database session
        """
        try:
            # Get camera and model details
            camera = db.query(Camera).filter(Camera.id == event.camera_id).first()
            ai_model = db.query(AiModel).filter(AiModel.id == event.model_id).first()

            if not camera or not ai_model:
                logger.warning(f"[BROADCAST] Cannot broadcast event {event.id}: missing camera or model")
                return

            # Hard guard: never broadcast events for inactive/deleted cameras.
            if bool(getattr(camera, "is_deleted", False)) or not bool(
                getattr(camera, "status", False)
            ):
                logger.info(
                    "[BROADCAST] Skip event %s for inactive camera %s (deleted=%s, status=%s)",
                    event.id,
                    camera.id,
                    bool(getattr(camera, "is_deleted", False)),
                    bool(getattr(camera, "status", False)),
                )
                return

            # Prepare event data for broadcasting
            event_data = {
                "id": str(event.id),
                "time": serialize_utc_datetime(event.time) if hasattr(event.time, "isoformat") else str(event.time),
                "camera_id": str(event.camera_id),
                "model_id": str(event.model_id),
                "detection_data": event.detection_data or {},
                "image_path": event.image_path,
                "camera": {
                    "id": str(camera.id),
                    "name": camera.name,
                },
                "ai_model": {
                    "id": str(ai_model.id),
                    "name": ai_model.name,
                    "model_type": ai_model.model_type,
                },
                # Legacy fields for backward compatibility with existing clients/scripts
                "event_id": str(event.id),
                "camera_name": camera.name,
                "model_name": ai_model.name,
                "model_type": ai_model.model_type,
                "metadata": event.detection_data or {},
            }

            # Schedule WebSocket broadcast on the main event loop
            loop = websocket_manager.event_loop
            if loop is not None and not loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    websocket_manager.broadcast_event(event_data),
                    loop
                )
                logger.info(f"[BROADCAST] Event {event.id} scheduled for WebSocket broadcast")
            else:
                logger.debug(f"[BROADCAST] Event loop closed/missing, skipping broadcast for event {event.id}")

        except RuntimeError as e:
            # Suppress "Event loop is closed" errors during shutdown
            if "closed" in str(e).lower():
                logger.debug(f"[BROADCAST] Skipped (shutting down): {e}")
            else:
                logger.error(f"[BROADCAST] Error broadcasting event {event.id}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[BROADCAST] Error broadcasting event {event.id}: {e}", exc_info=True)

    def cleanup(self):
        """Cleanup all models.

        Locked swap: replace ``self.models`` with an empty dict under the
        lock, then tear down the old instances outside the lock so that a
        processor thread already past the snapshot point in ``process_frame``
        still completes safely against the old models it captured.
        """
        with self._models_lock:
            old_models = self.models
            self.models = {}

        logger.info(
            f"Camera {self.camera_id}: Cleaning up {len(old_models)} models..."
        )
        for model_key, model_info in old_models.items():
            instance = model_info.get('model_instance')
            if instance is None:
                continue
            try:
                logger.debug(
                    f"Cleaning up model {model_info.get('model_name', model_key)}"
                )
                instance.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up model {model_key}: {e}")
        logger.info(f"Camera {self.camera_id}: Cleanup completed")

    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass  # Silently ignore errors in destructor
