# File: src/ai_models/evn_and_smartech_detection_model.py

"""
EVN & Smartech Person Monitoring Model
→ Lưu ảnh + tạo event cho TẤT CẢ người xuất hiện trong vùng giám sát:
   • EVN-Worker
   • Smartech-Worker
   • no vest → cảnh báo đỏ + log WARNING
→ Chống duplicate: mỗi người chỉ lưu 1 lần duy nhất cho đến khi rời ROI
   (không thấy trong 60 giây = đã rời)
→ event=True ngay khi có bất kỳ ai mới → tự động lưu ảnh + DB
"""

from typing import Dict, Any, List
import cv2
import numpy as np
import time
import logging
from ultralytics import YOLO

from src.ai_models.base_model import BaseModel, DetectionResult, resolve_engine_path
from src.utils.roi_utils import (
    DEFAULT_NORMALIZED_ROI,
    denormalize_roi,
    build_roi_poly_arrays,
    is_point_in_any_roi,
    draw_roi_overlays,
)


class EvnAndSmartechDetectionModel(BaseModel):
    """Model giám sát ra vào khu vực EVN & Smartech – lưu ảnh + cảnh báo người không mặc áo bảo hộ"""

    def __init__(
        self, model_path: str = "src/ai_models/model_weights/ppe_100ep.pt", **kwargs
    ):
        super().__init__(
            model_name="EvnAndSmartechDetectionModel", model_path=model_path, **kwargs
        )

        # Load model
        model_path = resolve_engine_path(model_path, runtime_device=self.device)
        self.model = YOLO(model_path)
        if self.device and model_path.endswith(".pt"):
            self.model.to(self.device)

        # Thresholds
        self.conf_threshold = kwargs.get("conf_threshold", 0.55)
        self.no_vest_conf_threshold = kwargs.get(
            "stranger_conf_threshold", 0.35
        )  # giữ tên cũ để tương thích

        # Classes
        self.classes_to_keep = [0, 1, 2]
        self.class_names = {
            # 0: 'EVN-Worker',
            # 1: 'Smartech-Worker',
            2: "no vest"
        }

        # Màu sắc
        self.color_map = {
            "EVN-Worker": (0, 255, 0),  # Xanh lá
            "Smartech-Worker": (0, 255, 255),  # Vàng
            "no vest": (0, 0, 255),  # Đỏ - Cảnh báo
        }

        # Chống duplicate: mỗi (track_id, class_name) chỉ lưu 1 lần
        # Người được xóa khỏi tracker sau _absence_timeout giây không thấy = rời ROI
        self._person_events: Dict[int, set] = {}  # track_id → set of class_names đã lưu
        self._last_seen: Dict[int, float] = {}  # track_id → timestamp lần cuối thấy
        self._absence_timeout = 60.0  # 60s không thấy = đã rời ROI
        self._roi_missing_warned = False

        self.logger.info(
            f"EvnAndSmartechDetectionModel LOADED | "
            f"Classes: {list(self.class_names.values())} | "
            f"Conf: {self.conf_threshold} | NoVest_Conf: {self.no_vest_conf_threshold} | "
            f"Device: {self.device}"
        )

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        h, w = frame.shape[:2]
        annotate = kwargs.get("annotate", True)
        current_time = time.time()

        # Xóa người đã rời ROI (không thấy >= 60s)
        self._cleanup_absent_persons(current_time)

        # ROI
        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), w, h)
        if "roi" not in kwargs:
            if not self._roi_missing_warned:
                self.logger.warning(
                    f"[ROI MISSING] ROI not in kwargs. Available keys: {list(kwargs.keys())}. "
                    f"Using default full frame ROI. Check camera_model.additional_parameters in DB. "
                    f"Further identical warnings are suppressed for this model instance."
                )
                self._roi_missing_warned = True
        else:
            self._roi_missing_warned = False
            self.logger.debug(f"[ROI OK] ROI received, {len(roi_polys)} polygon(s)")

        # Keep clean copy for inference (without annotations from previous models)
        clean_frame = frame

        # Inference + tracking
        results = self._run_yolo_track(
            self.model,
            clean_frame,
            conf=self.conf_threshold,
            classes=self.classes_to_keep,
            persist=True,
            verbose=False,
        )

        annotated_frame = frame.copy() if annotate else None
        all_detections: List[Dict[str, Any]] = []
        events_to_save: List[Dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            track_ids = (
                boxes.id.int().cpu().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Kiểm tra class_id có trong class_names không
                if cls_id not in self.class_names:
                    continue

                class_name = self.class_names[cls_id]

                # Tăng độ nhạy cho "no vest"
                if class_name == "no vest" and conf < self.no_vest_conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Kiểm tra trong ROI
                if not is_point_in_any_roi((cx, cy), roi_polys):
                    continue

                # Lấy track_id an toàn
                raw_track_id = track_ids[i] if i < len(track_ids) else None

                # Ghi nhận detection (giữ nguyên raw_track_id để backend biết có tracker thật hay không)
                detection = {
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "track_id": raw_track_id,  # None nếu không có tracker
                }
                all_detections.append(detection)

                # === CHỐNG DUPLICATE: mỗi (track_id, class_name) chỉ lưu 1 lần ===
                # Synthetic track_id fallback khi YOLO tracker không gán được ID
                # (thường xảy ra với VSS HTTP-FLV streams)
                if raw_track_id is None:
                    gx, gy = cx // 50, cy // 50
                    raw_track_id = hash((class_name, gx, gy)) & 0x7FFFFFFF

                track_id_int = int(raw_track_id)

                # Cập nhật last_seen
                self._last_seen[track_id_int] = current_time

                # Kiểm tra đã lưu event cho (track_id, class_name) chưa
                if track_id_int not in self._person_events:
                    self._person_events[track_id_int] = set()

                if class_name not in self._person_events[track_id_int]:
                    # Lần đầu thấy người này với class này → tạo event
                    self._person_events[track_id_int].add(class_name)

                    event_data = {
                        "track_id": raw_track_id,
                        "person_type": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "timestamp": current_time,
                    }
                    events_to_save.append(event_data)

                    if class_name == "no vest":
                        self.logger.warning(
                            f"[WARNING] Detected person WITHOUT SAFETY VEST | "
                            f"ID: {track_id_int} | Conf: {conf:.3f} | Pos: ({cx},{cy})"
                        )
                    else:
                        self.logger.info(
                            f"[LƯU ẢNH] Phát hiện {class_name} | "
                            f"ID: {track_id_int} | Conf: {conf:.3f}"
                        )

                # Vẽ annotation
                if annotate and annotated_frame is not None:
                    color = self.color_map[class_name]
                    thickness = 6 if class_name == "no vest" else 3

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    # Label đẹp
                    short_name = (
                        class_name.split("-")[0] if "-" in class_name else class_name
                    )
                    label = f"{short_name} {conf:.2f}"
                    if raw_track_id is not None:
                        label += f" ID:{int(raw_track_id)}"

                    if class_name == "no vest":
                        label = f"NO VEST {conf:.2f}"
                        if raw_track_id is not None:
                            label += f" ID:{int(raw_track_id)}"

                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
                    )
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - th - 20),
                        (x1 + tw + 30, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 10, y1 - 8),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

        # Vẽ vùng giám sát
        if annotate and annotated_frame is not None:
            draw_roi_overlays(
                annotated_frame, roi_polys, color=(255, 0, 255), thickness=4
            )

        # Xác định event chính (ưu tiên no vest)
        event_triggered = len(events_to_save) > 0
        primary_event = None
        if events_to_save:
            primary_event = next(
                (e for e in events_to_save if e["person_type"] == "no vest"),
                events_to_save[0],
            )

        # Metadata chi tiết
        metadata: Dict[str, Any] = {
            "type": "Trang phục bảo hộ EVN & Smartech",
            "severity": (
                "high"
                if primary_event and primary_event.get("person_type") == "no vest"
                else "low"
            ),
            "detections": all_detections,
            "saved_events": events_to_save,
            "evn_worker_count": sum(
                1 for e in events_to_save if e["person_type"] == "EVN-Worker"
            ),
            "smartech_worker_count": sum(
                1 for e in events_to_save if e["person_type"] == "Smartech-Worker"
            ),
            "no_vest_count": sum(
                1 for e in events_to_save if e["person_type"] == "no vest"
            ),
            "total_persons_logged": len(events_to_save),
            "timestamp": time.strftime("%Y%m%d%H%M%S"),
            "model_version": "v2.0-final",
        }

        if primary_event:
            person_type = primary_event["person_type"]
            metadata.update(
                {
                    "violation": "no_vest_detected"
                    if person_type == "no vest"
                    else "authorized_person",
                    "person_type": person_type,
                    "confidence": primary_event["confidence"],
                    "track_id": primary_event["track_id"],
                    "violation_level": "high" if person_type == "no vest" else "none",
                }
            )

        return DetectionResult(
            frame=annotated_frame if event_triggered or annotate else None,
            event=event_triggered,
            metadata=metadata,
        )

    def _cleanup_absent_persons(self, current_time: float):
        """Xóa người không thấy trong _absence_timeout giây (= đã rời ROI)"""
        expired = [
            tid
            for tid, last_t in self._last_seen.items()
            if current_time - last_t >= self._absence_timeout
        ]
        for tid in expired:
            self._person_events.pop(tid, None)
            self._last_seen.pop(tid, None)
            self.logger.info(
                f"Người ID={tid} đã rời ROI (không thấy >= {self._absence_timeout}s) → cho phép ghi lại"
            )
