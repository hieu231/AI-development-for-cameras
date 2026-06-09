from fast_alpr import ALPR
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import cv2
import os
import re
import time
import logging
import numpy as np
from datetime import datetime
from typing import Any, List, Optional

from src.ai_models.base_model import (
    BaseModel,
    DetectedObject,
    DetectionResult,
)
from src.utils.alert_levels import AlertLevel

logger = logging.getLogger(__name__)

# Function to normalize, validate and classify license plate
def validate_license_plate(plate_text):
    """Normalise OCR output and classify the plate.

    Strictness is adjustable — before, the validator required a perfect
    Vietnamese format (`\\d{2}[A-Z]\\d{4,6}` or `SGN\\d{4,6}`). The
    global fast-alpr OCR model doesn't consistently produce exact
    Vietnamese-format strings even for Vietnamese plates (minor
    mis-reads like "51F12345" → "51F1234S" are common), so we now
    accept anything that LOOKS like a plate — at least 4 alphanumeric
    chars with at least one letter and one digit — and still flag the
    strictly-matching Vietnamese formats as "refueling"/"car" for
    downstream routing. Set ALPR_STRICT_VN_FORMAT=true to fall back
    to the original behaviour.
    """
    if not plate_text:
        return None, False, None

    normalized = re.sub(r'[^A-Za-z0-9]', '', plate_text).upper()
    if not normalized or not re.match(r'^[A-Z0-9]+$', normalized):
        return normalized, False, None

    # Strict Vietnamese plate detection — always returns True for
    # properly-formed plates so downstream routing (refueling vs car)
    # still works.
    if re.match(r'^SGN\d{4,6}$', normalized):
        return normalized, True, "refueling"
    if re.match(r'^\d{2}[A-Z]\d{4,6}$', normalized):
        return normalized, True, "car"

    # Optional strict mode — keep legacy behaviour when the operator
    # wants Vietnamese-only plates.
    import os as _os
    if _os.getenv("ALPR_STRICT_VN_FORMAT", "false").lower() == "true":
        return normalized, False, None

    # Lenient mode (default): accept any sequence that's plausibly a
    # plate — 4–12 alphanumeric chars with at least one letter and
    # one digit.
    if (
        4 <= len(normalized) <= 12
        and re.search(r'[A-Z]', normalized)
        and re.search(r'\d', normalized)
    ):
        return normalized, True, "generic"

    return normalized, False, None

# Function to calculate similarity between two strings (0-1)
def string_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()

# Class to handle robust plate detection across multiple frames
class RobustPlateDetector:
    def __init__(self, buffer_size=5, similarity_threshold=0.7, inactivity_timeout=60):
        self.buffer_size = buffer_size
        self.similarity_threshold = similarity_threshold
        self.inactivity_timeout = inactivity_timeout  # Thời gian không hoạt động tính bằng giây
        self.plate_buffer = []  # Buffer to store recent valid plates
        self.confidence_buffer = []  # Buffer to store confidence values
        self.plate_type_buffer = []  # Buffer to store plate types
        self.timestamp_buffer = []  # Buffer to store timestamps
        self.last_confirmed_plate = None
        self.last_confirmed_type = None
        self.confidence_score = 0.0
        self.last_activity_time = time.time()  # Thời gian hoạt động cuối cùng
    
    def add_detection(self, plate_text, confidence, plate_type):
        current_time = time.time()
        
        # Kiểm tra nếu đã quá thời gian không hoạt động, reset buffer
        if current_time - self.last_activity_time > self.inactivity_timeout:
            self.plate_buffer = []
            self.confidence_buffer = []
            self.plate_type_buffer = []
            self.timestamp_buffer = []
            self.last_confirmed_plate = None
            self.last_confirmed_type = None
            self.confidence_score = 0.0
            print("Buffer được reset do không hoạt động trong", self.inactivity_timeout, "giây")
        
        # Cập nhật thời gian hoạt động cuối cùng
        self.last_activity_time = current_time
        
        if not plate_text:
            return None, 0.0, None
        
        # Add to buffer
        self.plate_buffer.append(plate_text)
        self.confidence_buffer.append(confidence)
        self.plate_type_buffer.append(plate_type)
        self.timestamp_buffer.append(current_time)
        
        # Keep buffer at desired size
        if len(self.plate_buffer) > self.buffer_size:
            self.plate_buffer.pop(0)
            self.confidence_buffer.pop(0)
            self.plate_type_buffer.pop(0)
            self.timestamp_buffer.pop(0)
        
        # Nếu biển số hiện tại có confidence cao (>0.8), cho phép nhận diện ngay
        if confidence > 0.8:
            self.last_confirmed_plate = plate_text
            self.last_confirmed_type = plate_type
            self.confidence_score = confidence
            return plate_text, confidence, plate_type
            
        # Kiểm tra sự thay đổi đột ngột trong buffer
        if self.detect_sudden_change():
            # Nếu phát hiện có sự thay đổi đột ngột, ưu tiên biển số mới
            new_plates = self.get_recent_plates(3)  # Lấy 3 biển số gần nhất
            if new_plates:
                plate_counter = Counter(new_plates)
                most_common = plate_counter.most_common(1)[0]
                if most_common[1] >= 2:  # Nếu biển mới xuất hiện ít nhất 2 lần
                    idx = self.plate_buffer.index(most_common[0])
                    self.last_confirmed_plate = most_common[0]
                    self.last_confirmed_type = self.plate_type_buffer[idx]
                    self.confidence_score = self.confidence_buffer[idx]
                    return self.last_confirmed_plate, self.confidence_score, self.last_confirmed_type
        
        # Nếu buffer chưa đủ, nhưng biển số có format đúng và confidence > 0.7
        if len(self.plate_buffer) < self.buffer_size and confidence > 0.7:
            # Kiểm tra xem biển số có xuất hiện ít nhất 2 lần trong buffer không
            if self.plate_buffer.count(plate_text) >= 2:
                self.last_confirmed_plate = plate_text
                self.last_confirmed_type = plate_type
                self.confidence_score = confidence
                return plate_text, confidence, plate_type
        
        # Nếu buffer đã đủ, thực hiện kiểm tra consensus như bình thường
        if len(self.plate_buffer) >= 3:  # Chỉ cần 3 mẫu để bắt đầu voting
            return self.get_consensus_plate()
            
        return None, 0.0, None
    
    def get_recent_plates(self, n):
        """Lấy n biển số gần đây nhất"""
        return self.plate_buffer[-n:] if len(self.plate_buffer) >= n else self.plate_buffer
    
    def detect_sudden_change(self):
        """Phát hiện sự thay đổi đột ngột trong buffer"""
        if len(self.plate_buffer) < 3:
            return False
            
        # Lấy 3 biển số gần nhất
        recent_plates = self.get_recent_plates(3)
        
        # Tính tần suất xuất hiện của biển số gần đây nhất
        newest_plate = recent_plates[-1]
        if newest_plate != self.last_confirmed_plate and self.last_confirmed_plate is not None:
            # Nếu biển số mới khác với biển số đã xác nhận trước đó
            # Kiểm tra độ tương đồng
            similarity = string_similarity(newest_plate, self.last_confirmed_plate)
            if similarity < 0.6:  # Nếu độ tương đồng thấp, có thể là xe mới
                # Kiểm tra biển số mới có xuất hiện nhiều lần gần đây không
                if recent_plates.count(newest_plate) >= 2:
                    return True
                
        return False
    
    def get_consensus_plate(self):
        if not self.plate_buffer:
            return None, 0.0, None
        
        # Áp dụng trọng số cho các mẫu gần đây hơn
        weighted_votes = {}
        for i, plate in enumerate(self.plate_buffer):
            # Mẫu gần đây hơn có trọng số cao hơn
            weight = (i + 1) / len(self.plate_buffer)
            weighted_votes[plate] = weighted_votes.get(plate, 0) + weight * self.confidence_buffer[i]
        
        # Xác định biển số có vote cao nhất
        best_plate = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        # Tính tỷ lệ vote của biển số được chọn
        total_weight = sum(weighted_votes.values())
        vote_ratio = weighted_votes[best_plate] / total_weight if total_weight > 0 else 0
        
        # Chỉ xác nhận nếu tỷ lệ vote đủ cao
        if vote_ratio >= 0.4:  # Giảm ngưỡng xuống 40% nhưng có trọng số
            # Lấy chỉ số của biển số được chọn trong buffer
            indices = [i for i, plate in enumerate(self.plate_buffer) if plate == best_plate]
            
            # Tính confidence trung bình và lấy loại xe phổ biến nhất
            total_confidence = sum(self.confidence_buffer[i] for i in indices)
            avg_confidence = total_confidence / len(indices) if indices else 0
            
            plate_types = [self.plate_type_buffer[i] for i in indices]
            plate_type_counter = Counter(plate_types)
            most_common_type = plate_type_counter.most_common(1)[0][0]
            
            self.last_confirmed_plate = best_plate
            self.last_confirmed_type = most_common_type
            self.confidence_score = avg_confidence
            return best_plate, avg_confidence, most_common_type
        
        # Kiểm tra nếu có biển số đã được xác nhận trước đó
        if self.last_confirmed_plate and self.last_confirmed_plate in self.plate_buffer:
            current_count = self.plate_buffer.count(self.last_confirmed_plate)
            if current_count / len(self.plate_buffer) >= 0.3:  # Ít nhất 30% hiện diện
                return self.last_confirmed_plate, self.confidence_score, self.last_confirmed_type
                
        return None, 0.0, None

# Class to handle duplicate detection prevention with cooldown
class DuplicateDetectionPreventer:
    def __init__(self, cooldown_seconds=15):
        self.cooldown_seconds = cooldown_seconds
        self.last_detection_times = defaultdict(float)
    
    def can_record(self, plate, plate_type, current_time):
        # Create a unique key for this plate and type
        key = f"{plate_type}:{plate}"
        
        # Check if we've seen this plate before and if the cooldown period has passed
        last_time = self.last_detection_times.get(key, 0)
        time_since_last = current_time - last_time
        
        if time_since_last >= self.cooldown_seconds or last_time == 0:
            # Update the last detection time
            self.last_detection_times[key] = current_time
            return True, time_since_last
        
        # Plate is in cooldown
        return False, time_since_last

class ALPRModel(BaseModel):
    """License-plate recognition powered by fast-alpr (ONNX detector + OCR).

    fast-alpr loads detector + ocr weights internally by *identifier*, not from
    a YOLO .pt file — so the ``model_path`` kwarg the factory passes is
    accepted-and-ignored here. Likewise we accept ``**kwargs`` so the factory
    can forward camera-specific parameters (``model_type``, ``conf_threshold``,
    etc.) without breaking instantiation.
    """

    DEFAULT_DETECTOR = "yolo-v9-t-640-license-plate-end2end"
    DEFAULT_OCR = "global-plates-mobile-vit-v2-model"

    def __init__(
        self,
        model_path: Optional[str] = None,
        detector_model: Optional[str] = None,
        ocr_model: Optional[str] = None,
        buffer_size: int = 5,
        similarity_threshold: float = 0.7,
        inactivity_timeout: int = 60,
        cooldown_seconds: int = 25,
        confidence_threshold: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=kwargs.pop("model_name", "ALPR"),
            default_alert_level=AlertLevel.LOW,
            confidence_threshold=confidence_threshold,
            model_path=model_path,
            **{k: v for k, v in kwargs.items() if k not in {"model_type"}},
        )

        # ── Provider selection (same story as insightface) ────────────
        # The CUDA 12.9 onnxruntime-gpu wheel crashes at kernel launch
        # on the Jetson CUDA 13 host with `cudaErrorSymbolNotFound`, so
        # force TRT EP first and explicitly EXCLUDE CUDAExecutionProvider
        # — without this, open_image_models and fast_plate_ocr fall back
        # to CUDA EP for unsupported ops and crash mid-inference.
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            available = []
        providers: List[str] = []
        if "TensorrtExecutionProvider" in available:
            providers.append("TensorrtExecutionProvider")
        allow_cuda = os.getenv("ALPR_ALLOW_CUDA_EP", "false").lower() == "true"
        if allow_cuda and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.logger.info("ALPR providers request: %s (available=%s)", providers, available)

        # fast-alpr exposes `detector_providers` + `ocr_providers` directly
        # on the ALPR constructor — pass our pruned list so the underlying
        # open_image_models + fast_plate_ocr sessions skip the broken
        # CUDA EP.
        self.alpr = ALPR(
            detector_model=detector_model or self.DEFAULT_DETECTOR,
            ocr_model=ocr_model or self.DEFAULT_OCR,
            detector_providers=providers,
            ocr_providers=providers,
        )
        self.robust_detector = RobustPlateDetector(
            buffer_size=buffer_size,
            similarity_threshold=similarity_threshold,
            inactivity_timeout=inactivity_timeout,
        )
        self.duplicate_preventer = DuplicateDetectionPreventer(
            cooldown_seconds=cooldown_seconds,
        )

    # Throttled heartbeat so the log shows whether fast-alpr is
    # actually returning detections without spamming once per frame.
    _diag_last_log = 0.0
    _diag_frame_count = 0
    _diag_plate_detections = 0
    _diag_valid_plates = 0

    def process_frame(self, frame: np.ndarray, **_: Any) -> DetectionResult:
        # fast-alpr inference + OCR
        try:
            results = self.alpr.predict(frame)
        except Exception as exc:
            self.logger.error("ALPR inference failed: %s", exc, exc_info=True)
            return DetectionResult(
                frame=None,
                event=False,
                metadata={
                    "model": self.model_name,
                    "model_type": "alpr",
                    "detections": [],
                    "error": str(exc),
                },
            )

        self._diag_frame_count += 1
        if results:
            self._diag_plate_detections += len(results)
        now_ts = time.time()
        if now_ts - self._diag_last_log >= 5.0:
            # Use both print() and logger so whichever is wired up
            # surfaces the counter. print() bypasses log level config.
            msg = (
                f"ALPR diag: frames={self._diag_frame_count} "
                f"plate_boxes={self._diag_plate_detections} "
                f"valid_plates={self._diag_valid_plates}"
            )
            print(msg, flush=True)
            self.logger.info(msg)
            self._diag_last_log = now_ts
            self._diag_frame_count = 0
            self._diag_plate_detections = 0
            self._diag_valid_plates = 0

        best_plate = None
        best_confidence = 0.0
        best_plate_type = None
        confirmed_plate = None
        confirmed_confidence = 0.0
        confirmed_type = None
        detection_objects: list = []
        raw_detections: list = []

        # Helper: fast-plate-ocr returns per-character confidences as a
        # list now. Older builds returned a scalar. Accept either.
        def _scalar_conf(v: Any) -> float:
            try:
                if isinstance(v, (list, tuple)):
                    return float(sum(v) / len(v)) if v else 0.0
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        if results:
            current_frame_plates = []
            for result in results:
                try:
                    if hasattr(result, "ocr") and result.ocr is not None:
                        raw_text = result.ocr.text
                        confidence = _scalar_conf(result.ocr.confidence)

                        # Capture the raw box for overlay metadata
                        bbox = None
                        det = getattr(result, "detection", None)
                        det_box = getattr(det, "bounding_box", None) if det is not None else None
                        if det_box is not None:
                            try:
                                bbox = (
                                    int(det_box.x1),
                                    int(det_box.y1),
                                    int(det_box.x2),
                                    int(det_box.y2),
                                )
                            except Exception:
                                bbox = None

                        normalized_text, is_valid, plate_type = validate_license_plate(raw_text)
                        if is_valid:
                            self._diag_valid_plates += 1
                        # Use compose_annotations-friendly keys (`label`,
                        # `class_name`) so the WebRTC overlay can redraw the
                        # plate text on the live stream every frame.
                        overlay_label = (
                            f"plate:{normalized_text}"
                            if is_valid
                            else (raw_text or "plate?")
                        )
                        raw_detections.append(
                            {
                                "raw_text": raw_text,
                                "normalized_text": normalized_text,
                                "valid": bool(is_valid),
                                "plate_type": plate_type,
                                "confidence": confidence,
                                "bbox": bbox,
                                "label": overlay_label,
                                "class_name": overlay_label,
                            }
                        )
                        if is_valid:
                            current_frame_plates.append(
                                (normalized_text, confidence, plate_type, bbox)
                            )
                except Exception as exc:
                    self.logger.debug("Skipping malformed ALPR result: %s", exc)

            if current_frame_plates:
                current_frame_plates.sort(key=lambda x: x[1], reverse=True)
                best_plate, best_confidence, best_plate_type, best_bbox = current_frame_plates[0]
                confirmed_plate, confirmed_confidence, confirmed_type = self.robust_detector.add_detection(
                    best_plate, best_confidence, best_plate_type
                )

        # Build our own overlay from the results we already have —
        # fast-alpr's `draw_predictions()` internally RE-RUNS the whole
        # detect+OCR pipeline just to draw, doubling per-frame inference
        # cost and stalling TRT compile. We draw from `results` directly.
        annotated_frame = frame.copy() if frame is not None else None
        if annotated_frame is not None and results:
            for r in results:
                det = getattr(r, "detection", None)
                box = getattr(det, "bounding_box", None) if det is not None else None
                if box is None:
                    continue
                try:
                    x1 = int(box.x1); y1 = int(box.y1)
                    x2 = int(box.x2); y2 = int(box.y2)
                except Exception:
                    continue
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ocr = getattr(r, "ocr", None)
                if ocr is not None and getattr(ocr, "text", None):
                    txt = f"{ocr.text} ({_scalar_conf(ocr.confidence):.2f})"
                    cv2.putText(
                        annotated_frame, txt, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

        event_triggered = False
        event_payload: dict = {}
        if confirmed_plate:
            current_time = time.time()
            can_record, _ = self.duplicate_preventer.can_record(
                confirmed_plate, confirmed_type, current_time,
            )
            if can_record:
                vehicle_label = "Xe Bồn" if confirmed_type == "refueling" else "Xe Con"
                event_triggered = True
                event_payload = {
                    "type": "Nhận diện biển số",
                    "plate_number": confirmed_plate,
                    "vehicle_type": vehicle_label,
                    "confidence": float(confirmed_confidence),
                }
                detection_objects.append(
                    DetectedObject(
                        label=f"{vehicle_label}: {confirmed_plate}",
                        confidence=float(confirmed_confidence),
                        bbox=None,
                        extra={"plate_type": confirmed_type},
                    )
                )
                # Burn the confirmed plate text onto the overlay
                cv2.putText(
                    annotated_frame,
                    f"{vehicle_label}: {confirmed_plate} ({confirmed_confidence:.2f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

        alert_info = None
        if event_triggered:
            alert_info = self._create_alert_info(
                message=event_payload["type"],
                confidence=event_payload["confidence"],
                detected_objects=detection_objects,
            )

        metadata = self._build_metadata(
            alert_info=alert_info,
            extra={
                "model_type": "alpr",
                "detections": raw_detections,
                **event_payload,
            },
        )

        # Always emit annotated_frame so the WebRTC overlay shows fast-alpr's
        # detection boxes on every frame, not just on event-trigger frames.
        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )