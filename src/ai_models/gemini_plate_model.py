import cv2
import numpy as np
import time
import threading
import os
import logging
from datetime import date
from typing import Any, Optional
from ultralytics import YOLO
from collections import deque
from queue import Queue, Empty
from PIL import Image
import io
import random

from src.ai_models.base_model import (
    BaseModel,
    DetectedObject,
    DetectionResult,
    resolve_engine_path,
)
from src.utils.alert_levels import AlertLevel

try:
    # type: ignore[import] - optional dependency, may be missing in some envs
    import google.generativeai as genai  # pyright: ignore[reportMissingImports]
except ImportError:  # Library not installed; handled gracefully in code
    genai = None

logger = logging.getLogger(__name__)


class GeminiPlateModel(BaseModel):
    """Vehicle tracker + Gemini-API license plate extractor.

    Accepts ``model_path`` and arbitrary ``**kwargs`` (e.g. ``model_type``)
    so the ModelFactory can forward camera-specific parameters without
    blowing up instantiation.
    """

    def __init__(
        self,
        model_path: Optional[str] = "src/ai_models/model_weights/vehicle_detection.pt",
        api_key: Optional[str] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
        confidence_threshold: float = 0.6,
        detection_cooldown: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=kwargs.pop("model_name", "GeminiPlate"),
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **{k: v for k, v in kwargs.items() if k not in {"model_type"}},
        )

        self.api_key = api_key or os.getenv(
            "GEMINI_API_KEY",
            "AIzaSyA3muuo1IQlekF6cHSFfZcH9VwAzpC9NWs",
        )
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Resolve TensorRT engine when CUDA is available
        runtime_device = self.device if self.device in {"cuda", "cpu", "mps"} else None
        resolved_path = resolve_engine_path(model_path, runtime_device=runtime_device)
        self.model_path = resolved_path
        self.model = YOLO(resolved_path)

        self.truck_frames_dict = {}
        self.track_history = {}
        self.last_seen = {}
        self.processing_queue: Queue = Queue()
        self.display_messages = deque(maxlen=10)
        self.lock = threading.Lock()
        self.last_results = {}  # track_id -> (plate, confidence, timestamp, recognized_image)
        self.last_detection_time = 0
        self.detection_cooldown = detection_cooldown
        self.processor_thread = threading.Thread(target=self.queue_processor, daemon=True)
        self.processor_thread.start()

        # ── Vehicle counter / enter-exit event tracking ─────────────────
        # Per-day counter (resets at local midnight) so the dashboard can
        # show "N vehicles entered today" without scanning the events
        # table. Tracked entirely in memory — survives a process restart
        # because graceful_startup pulls cameras back up and the count
        # naturally re-derives from new detections.
        self.daily_vehicle_count = 0
        self.daily_count_date = date.today()
        # Track IDs we've already emitted an ENTER event for, so a
        # vehicle that lingers for hundreds of frames produces ONE enter
        # event and ONE exit event — not one per frame.
        self.tracks_entered: set = set()
        # Track IDs that have already emitted an EXIT event so we never
        # double-emit even if the queue processor finishes a Gemini call
        # after the timeout-based exit was emitted.
        self.tracks_exited: set = set()
        # Per-track time-of-first-detection so EXIT events can include
        # the dwell duration as evidence metadata.
        self.tracks_first_seen: dict = {}
        # Per-track number of frames the YOLO tracker has actually seen
        # this id. We only "confirm" a track (and emit ENTER) once it has
        # been observed for >= MIN_FRAMES_TO_CONFIRM frames — that way a
        # single-frame YOLO false positive does NOT spam an enter+exit
        # pair when the spurious id disappears 100ms later.
        self.tracks_frame_count: dict = {}
        self.MIN_FRAMES_TO_CONFIRM = int(
            os.getenv("GEMINI_PLATE_MIN_FRAMES_CONFIRM", "5")
        )
        # Minimum dwell BEFORE confirmation expires the candidate (no
        # ENTER ever emitted, so no EXIT either). Without this, every
        # 1-frame phantom track would clutter the events table.
        self.MIN_DWELL_TO_CONFIRM_SEC = float(
            os.getenv("GEMINI_PLATE_MIN_DWELL_CONFIRM_SEC", "1.0")
        )

    def _get_yolo_runtime_kwargs(self, **overrides):
        runtime = {}

        device = self.device
        if device == 'mps' or device is None:
            device = 'cpu'
        runtime['device'] = device

        raw_imgsz = (os.getenv("AIBE_YOLO_IMGSZ", "640") or "").strip()
        if raw_imgsz:
            try:
                imgsz = int(raw_imgsz)
                if imgsz > 0:
                    runtime["imgsz"] = imgsz
            except ValueError:
                pass

        raw_stride = (os.getenv("AIBE_YOLO_VID_STRIDE") or "").strip()
        if raw_stride:
            try:
                stride = int(raw_stride)
                if stride > 1:
                    runtime["vid_stride"] = stride
            except ValueError:
                pass

        tracker = (os.getenv("AIBE_YOLO_TRACKER", "bytetrack.yaml") or "").strip()
        if tracker:
            runtime["tracker"] = tracker

        use_half = os.getenv("AIBE_YOLO_HALF", "true").lower() == "true"
        if use_half and isinstance(device, int):
            runtime["half"] = True

        runtime.update(overrides)
        return runtime

    def _run_yolo_track(self, model, source, **kwargs):
        return model.track(source=source, **self._get_yolo_runtime_kwargs(**kwargs))

    def get_images(self, truck_frames, num_images=15, trim_percentage=10):
        try:
            if not truck_frames:
                return []
            total_frames = len(truck_frames)
            trim_percentage = max(0, min(trim_percentage, 50))
            if trim_percentage > 0:
                trim_count = int(total_frames * trim_percentage / 100)
                trimmed_frames = truck_frames[trim_count:total_frames - trim_count]
            else:
                trimmed_frames = truck_frames
            trimmed_total = len(trimmed_frames)
            if trimmed_total <= num_images:
                selected_frames = trimmed_frames
            else:
                indices = np.linspace(0, trimmed_total - 1, num_images, dtype=int)
                selected_frames = [trimmed_frames[i] for i in indices]
            return selected_frames
        except Exception as e:
            print(f"Error in get_images: {e}")
            return []

    def get_license_plate_from_gemini(self, images):
        try:
            # If Gemini client lib is not available or no API key, skip gracefully
            if not self.api_key or genai is None:
                return 'Not recognized', 0.0, None

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = (
                "Analyze this image and return the license plate number. "
                "Only return the license plate if clear, otherwise return 'Not recognized'. "
                "Result as plain text, e.g., '51H-12345' or 'Not recognized'. You can skip - or . in license plate. "
                "The format of license plate are:  SGNxxxxx format (there are only 7 available plates: SNG32001, SNG32002, SGN32003, SNG32006, SNG32008, SNG32009, SNG32015. So if you find a license plate that is not in this list and starts with SGN, compare it to get a license plate in the list.) and Regular car - 51F12345 format (2 numbers, 1 letter, 4-6 numbers)"
            )
            for img in images:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG')
                image_part = {
                    "mime_type": "image/jpeg",
                    "data": img_byte_arr.getvalue()
                }
                response = model.generate_content([prompt, image_part])
                try:
                    license_plate = response.candidates[0].content.parts[0].text.strip()
                    if license_plate and len(license_plate) <= 20 and license_plate != 'Not recognized':
                        # Confidence is not provided by Gemini, so set to a random value between 0.85 and 0.95 if recognized
                        confidence = random.uniform(0.85, 0.95)
                        return license_plate, confidence, img
                except (AttributeError, IndexError):
                    continue
            return 'Not recognized', 0.0, None
        except Exception as e:
            print(f"Error in get_license_plate_from_gemini: {e}")
            return 'Not recognized', 0.0, None

    def queue_processor(self):
        while True:
            try:
                track_id, frames = self.processing_queue.get(block=True, timeout=1)
                self.process_license_plate(track_id, frames)
                self.processing_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"Error in queue_processor: {e}")

    def process_license_plate(self, track_id, frames):
        try:
            selected_frames = self.get_images(frames, num_images=15)
            if selected_frames:
                license_plate, confidence, recognized_image = self.get_license_plate_from_gemini(selected_frames)
                timestamp = time.time()
                with self.lock:
                    self.last_results[track_id] = (license_plate, confidence, timestamp, recognized_image)

                if license_plate != 'Not recognized':
                    message = f"Xe ID {track_id} biển số {license_plate} đã xử lý"
                    self.display_messages.append((message, timestamp))
                    print(message)
                else:
                    print(f"Xe ID {track_id}: Không nhận diện được biển số.")

            with self.lock:
                if track_id in self.truck_frames_dict:
                    del self.truck_frames_dict[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
                if track_id in self.last_seen:
                    del self.last_seen[track_id]
        except Exception as e:
            print(f"Error processing track {track_id}: {e}")

    def _maybe_reset_daily_counter(self) -> None:
        """Roll the daily vehicle counter at local midnight."""
        today = date.today()
        if today != self.daily_count_date:
            self.daily_vehicle_count = 0
            self.daily_count_date = today
            self.tracks_entered.clear()
            self.tracks_exited.clear()
            self.tracks_first_seen.clear()
            self.logger.info(
                "Daily vehicle counter reset for %s", today.isoformat()
            )

    def process_frame(self, frame: np.ndarray, **_: Any) -> DetectionResult:
        # Resize input frame if needed
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        current_time = time.time()
        detection_result = None
        can_record = False
        annotated_frame = frame.copy()
        current_ids = set()
        # Vehicle enter/exit events accumulated this frame. Each entry
        # becomes a SEPARATE DB row via SingleThreadProcessor's
        # `metadata['violations']` batch-save path, so multiple cars
        # entering or exiting in the same frame all get recorded with
        # the right per-vehicle plate + dwell time without any spam.
        vehicle_violations: list = []
        self._maybe_reset_daily_counter()
        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame
        # Run YOLO tracking
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            persist=True,
            conf=0.6,
            iou=0.3,
            verbose=False,
        )
        # Per-frame detection records (consumed by compose_annotations to
        # redraw bboxes on the WebRTC overlay) — must be populated EVERY frame,
        # not just on the rare frames that emit a DB event, otherwise the
        # operator sees a clean stream with no boxes.
        frame_detections: list = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu()
            confidences = (
                results[0].boxes.conf.cpu().tolist()
                if getattr(results[0].boxes, "conf", None) is not None
                else []
            )
            track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []
            for idx, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x, y, w, h = box
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                # Expand crop area
                expand_ratio = 0.2
                w_expand = int(w * expand_ratio)
                h_expand = int(h * expand_ratio)
                x1 = max(0, x1 - w_expand)
                y1 = max(0, y1 - h_expand)
                x2 = min(frame.shape[1], x2 + w_expand)
                y2 = min(frame.shape[0], y2 + h_expand)
                # Capture for overlay redraw — uses the EXPANDED bbox so the
                # rectangle the operator sees matches the crop fed to Gemini.
                conf_value = float(confidences[idx]) if idx < len(confidences) else 0.0
                # Use the latest known plate text for this track, if Gemini
                # has produced one in the queue processor.
                with self.lock:
                    last_seen_plate = (
                        self.last_results.get(track_id, (None,))[0]
                        if track_id in self.last_results
                        else None
                    )
                detection_label = (
                    f"plate:{last_seen_plate}"
                    if last_seen_plate and last_seen_plate != "Not recognized"
                    else "vehicle"
                )
                frame_detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf_value,
                        "track_id": int(track_id),
                        "class_name": detection_label,
                        "label": detection_label,
                    }
                )
                # Crop license plate area (actually vehicle area)
                license_plate_crop = frame[y1:y2, x1:x2]
                # Store frame for this track
                with self.lock:
                    if track_id not in self.truck_frames_dict:
                        self.truck_frames_dict[track_id] = []
                    # Crop giữ nguyên size, chỉ resize nếu quá lớn
                    max_size = 1024
                    h, w = license_plate_crop.shape[:2]
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        crop_resized = cv2.resize(license_plate_crop, (new_w, new_h))
                    else:
                        crop_resized = license_plate_crop.copy()
                    self.truck_frames_dict[track_id].append(crop_resized)
                    # Update track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    self.track_history[track_id].append((int(x), int(y)))
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    # Update last seen
                    self.last_seen[track_id] = current_time
                current_ids.add(track_id)

                # ── Track confirmation gate ─────────────────────────
                # Only emit ENTER once the YOLO tracker has held this
                # ID for >= MIN_FRAMES_TO_CONFIRM frames AND for at
                # least MIN_DWELL_TO_CONFIRM_SEC. Single-frame phantom
                # detections (the dwell=0.1s spam we saw) get filtered
                # out silently — no ENTER, no EXIT.
                track_id_int = int(track_id)
                if track_id_int not in self.tracks_first_seen:
                    self.tracks_first_seen[track_id_int] = current_time
                self.tracks_frame_count[track_id_int] = (
                    self.tracks_frame_count.get(track_id_int, 0) + 1
                )

                first_seen_at = self.tracks_first_seen[track_id_int]
                dwell_so_far = current_time - first_seen_at
                frames_so_far = self.tracks_frame_count[track_id_int]
                confirmed = (
                    frames_so_far >= self.MIN_FRAMES_TO_CONFIRM
                    and dwell_so_far >= self.MIN_DWELL_TO_CONFIRM_SEC
                )

                if confirmed and track_id_int not in self.tracks_entered:
                    self.tracks_entered.add(track_id_int)
                    self.daily_vehicle_count += 1
                    vehicle_violations.append(
                        {
                            "violation_type": "vehicle_enter",
                            "event_type": "Xe vào khu vực",
                            "type": "Xe vào khu vực",
                            "title": "Xe vào khu vực",
                            "description": (
                                f"Xe #{track_id_int} đã vào khu vực "
                                f"(tổng hôm nay: {self.daily_vehicle_count})"
                            ),
                            "track_id": track_id_int,
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf_value,
                            "vehicles_today": self.daily_vehicle_count,
                            "direction": "enter",
                        }
                    )
                    self.logger.info(
                        "Vehicle ENTER (confirmed after %d frames / %.1fs): "
                        "track_id=%d, vehicles_today=%d",
                        frames_so_far,
                        dwell_so_far,
                        track_id_int,
                        self.daily_vehicle_count,
                    )

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Counter overlay so the operator sees "vehicles_today=N" live.
        cv2.putText(
            annotated_frame,
            f"Vehicles today: {self.daily_vehicle_count}",
            (10, self.frame_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        # ── EXIT events ─────────────────────────────────────────────
        # Vehicles that haven't been seen for >10s. Only emit EXIT for
        # tracks that were CONFIRMED (>= MIN_FRAMES_TO_CONFIRM frames,
        # >= MIN_DWELL_TO_CONFIRM_SEC seconds). Unconfirmed tracks are
        # phantoms — single-frame YOLO false positives — and we drop
        # them silently so the events table doesn't fill with
        # dwell=0.1s noise.
        exited_now: list = []
        with self.lock:
            for track_id in list(self.last_seen.keys()):
                track_id_int = int(track_id)
                if (
                    track_id in current_ids
                    or (current_time - self.last_seen[track_id]) <= 10
                ):
                    continue

                # Track has timed out. Decide enter-confirmed → emit
                # EXIT, OR unconfirmed phantom → drop silently.
                was_confirmed = track_id_int in self.tracks_entered
                already_exited = track_id_int in self.tracks_exited

                if was_confirmed and not already_exited:
                    self.tracks_exited.add(track_id_int)
                    plate_text = None
                    plate_conf = 0.0
                    if track_id in self.last_results:
                        plate_text = self.last_results[track_id][0]
                        plate_conf = float(self.last_results[track_id][1] or 0.0)
                    first_seen_at = self.tracks_first_seen.get(
                        track_id_int, self.last_seen[track_id]
                    )
                    dwell_seconds = max(
                        0.0, current_time - float(first_seen_at) - 10.0
                    )
                    exited_now.append(
                        {
                            "violation_type": "vehicle_exit",
                            "event_type": "Xe rời khu vực",
                            "type": "Xe rời khu vực",
                            "title": "Xe rời khu vực",
                            "description": (
                                f"Xe #{track_id_int} rời khu vực "
                                f"(plate: {plate_text or 'không nhận diện'}, "
                                f"dwell: {dwell_seconds:.1f}s)"
                            ),
                            "track_id": track_id_int,
                            "bbox": None,
                            "confidence": plate_conf,
                            "plate_number": plate_text or None,
                            "dwell_seconds": round(dwell_seconds, 2),
                            "vehicles_today": self.daily_vehicle_count,
                            "direction": "exit",
                        }
                    )
                    self.logger.info(
                        "Vehicle EXIT: track_id=%d, plate=%s, dwell=%.1fs",
                        track_id_int,
                        plate_text or "?",
                        dwell_seconds,
                    )
                elif not was_confirmed:
                    # Phantom track — log once at debug level so
                    # operators can tune MIN_FRAMES_TO_CONFIRM if needed
                    # without seeing the noise in production logs.
                    frames_seen = self.tracks_frame_count.get(track_id_int, 0)
                    self.logger.debug(
                        "Discarding phantom track_id=%d (only %d frames)",
                        track_id_int,
                        frames_seen,
                    )

                # Existing background-OCR queueing — unchanged. Only
                # queues for OCR if the track had enough frames to
                # collect a useful crop bundle anyway.
                if track_id in self.truck_frames_dict:
                    if len(self.truck_frames_dict[track_id]) <= 10:
                        del self.truck_frames_dict[track_id]
                    else:
                        self.processing_queue.put((track_id, self.truck_frames_dict[track_id]))
                        del self.truck_frames_dict[track_id]
                # Drop bookkeeping for this id so the tracker can reuse
                # it if needed without re-emitting events.
                self.last_seen.pop(track_id, None)
                self.tracks_first_seen.pop(track_id_int, None)
                self.tracks_frame_count.pop(track_id_int, None)
        vehicle_violations.extend(exited_now)
        # Check if any vehicle just left and has a result
        with self.lock:
            for track_id, (plate, confidence, timestamp, recognized_image) in list(self.last_results.items()):
                if plate and plate != 'Not recognized' and recognized_image is not None:
                    detection_result = {
                        'type': 'Nhận diện biển số (Gemini)',
                        'plate_number': plate,
                        'confidence': float(confidence),
                        'timestamp': timestamp,
                        'image_to_save': recognized_image
                    }
                    can_record = True
                else:
                    detection_result = None
                    can_record = False

                del self.last_results[track_id]
                if can_record:
                    break
        # Display messages (optional, for debug)
        y_offset = 30
        for message, timestamp in list(self.display_messages):
            if current_time - timestamp <= 3:
                seconds_ago = current_time - timestamp
                display_text = message.replace("{0:.1f}", f"{seconds_ago:.1f}")
                cv2.putText(annotated_frame, display_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        # Ensure output frame is the right size
        annotated_frame = cv2.resize(annotated_frame, (self.frame_width, self.frame_height))

        # Wrap into the standardized DetectionResult so the per-camera processor
        # actually triggers event saving + websocket broadcast. The previous
        # tuple return shape was silently ignored by SingleThreadProcessor.
        event_payload: dict = {}
        detected_objects: list = []
        if can_record and detection_result:
            event_payload = {
                "type": detection_result.get("type", "Nhận diện biển số (Gemini)"),
                "plate_number": detection_result.get("plate_number"),
                "confidence": float(detection_result.get("confidence", 0.0)),
                "vehicle_type": "Xe (Gemini)",
            }
            detected_objects.append(
                DetectedObject(
                    label=f"{event_payload['type']}: {event_payload['plate_number']}",
                    confidence=event_payload["confidence"],
                    bbox=None,
                    extra={"source": "gemini"},
                )
            )

        alert_info = None
        if event_payload:
            alert_info = self._create_alert_info(
                message=event_payload["type"],
                confidence=event_payload["confidence"],
                detected_objects=detected_objects,
            )

        # Pull plate-recognition event into the violations list too so
        # the camera processor saves it as a separate row alongside the
        # vehicle enter/exit events.
        if event_payload:
            vehicle_violations.append(
                {
                    "violation_type": "plate_recognized",
                    "event_type": event_payload["type"],
                    "type": event_payload["type"],
                    "title": event_payload["type"],
                    "description": (
                        f"Biển số: {event_payload.get('plate_number')} "
                        f"({event_payload.get('confidence', 0):.2f})"
                    ),
                    "track_id": None,
                    "bbox": None,
                    "confidence": event_payload["confidence"],
                    "plate_number": event_payload.get("plate_number"),
                }
            )

        extra_meta = {
            "model_type": "gemini_plate",
            "detections": frame_detections,
            "vehicles_today": self.daily_vehicle_count,
            "active_tracks": sorted(int(t) for t in current_ids),
        }
        if vehicle_violations:
            extra_meta["violations"] = vehicle_violations
        if event_payload:
            extra_meta.update(event_payload)

        metadata = self._build_metadata(
            alert_info=alert_info,
            extra=extra_meta,
        )

        # Trigger event=True whenever ANY violation accumulated this
        # frame — enter, exit, or plate recognition. The processor's
        # batch-save path will then create one DB row per violation.
        any_event = bool(vehicle_violations)

        # Always emit the annotated frame so the WebRTC overlay shows the
        # YOLO-drawn boxes + counter even on frames that don't trigger a
        # DB event. The processor only saves frames when `event=True`,
        # so this is safe.
        return DetectionResult(
            frame=annotated_frame,
            event=any_event,
            metadata=metadata,
        )
