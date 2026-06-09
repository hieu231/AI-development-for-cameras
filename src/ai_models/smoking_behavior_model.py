"""
Smoking behavior model.
Detects smoking-related behavior using OpenVINO + BYTETracker.
"""

import cv2
import time
import os
import numpy as np
# OpenVINO 2024+ removed the `openvino.runtime` shim in 2026.x. Modern API
# exposes Core / Tensor at the top-level `openvino` namespace, so we try the
# new layout first and fall back to the legacy import for older base images.
try:
    import openvino as ov                # OpenVINO ≥ 2024
    if not hasattr(ov, "Core"):          # very old wheels lack top-level Core
        raise ImportError("openvino.Core missing")
except ImportError:
    import openvino.runtime as ov        # Legacy ≤ 2023
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.ai_models.base_model import BaseModel, DetectionResult, AlertLevel
from src.utils.roi_utils import build_roi_poly_arrays, draw_roi_overlays, is_point_in_any_roi
from src.ai_models.bytetrack_init import bytetrack, make_parser
from src.yolov8onnx.utils import xywh2xyxy, nms
from src.trackers.byte_tracker import BYTETracker


class SmokingBehaviorModel(BaseModel):
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        tracker_args=None,
        model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        **kwargs,
    ):
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        self._weights_dir = os.path.join(self._current_dir, "model_weights")
        self._cgr_model_path = os.path.abspath(model_path or os.path.join(self._weights_dir, "last.onnx"))
        default_pose_model = os.path.join(os.path.dirname(self._cgr_model_path), "yolov8n-pose.onnx")
        self._pose_model_path = os.path.abspath(pose_model_path or default_pose_model)
        self._build_visual_assets()

        super().__init__(
            model_name="SmokingBehaviorModel",
            default_alert_level=AlertLevel.HIGH,
            confidence_threshold=confidence_threshold,
            model_path=self._cgr_model_path,
            **kwargs,
        )

        if not os.path.isfile(self._cgr_model_path):
            raise FileNotFoundError(f"SmokingBehaviorModel base model not found: {self._cgr_model_path}")
        if not os.path.isfile(self._pose_model_path):
            raise FileNotFoundError(f"SmokingBehaviorModel pose model not found: {self._pose_model_path}")

        self._core1 = ov.Core()
        self._core2 = ov.Core()
        self._ov_device = (os.getenv("AIBE_SMOKING_OV_DEVICE") or "CPU").strip() or "CPU"
        self._cgr_model = self._core1.compile_model(
            self._cgr_model_path,
            self._ov_device,
        )
        self._pose_model = self._core2.compile_model(
            self._pose_model_path,
            self._ov_device,
        )
        self._output_node = self._pose_model.outputs[0]
        self._infer_cgr = self._cgr_model.create_infer_request()
        self._infer_pose = self._pose_model.create_infer_request()
        self._tracker_args = tracker_args or make_parser().parse_args([])
        self._tracker = BYTETracker(self._tracker_args, frame_rate=30)
        self._tracker_cgr = BYTETracker(self._tracker_args, frame_rate=60)

        self._count = 0
        self._cgr_conf = float(confidence_threshold)
        self._ids: Dict[int, np.ndarray] = {}

    @dataclass
    class _PoseResult:
        xyxy: np.ndarray
        conf: float
        id: Optional[int]
        keypoints: np.ndarray

    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        start_time = time.time()
        cgr_conf = float(kwargs.get("cgr_conf", self._cgr_conf))
        skeleton = bool(kwargs.get("skeleton", False))
        cig_box = bool(kwargs.get("cig_box", False))
        threshold = int(kwargs.get("threshold", 50))
        annotate = bool(kwargs.get("annotate", True))
        roi_polys = build_roi_poly_arrays(kwargs.get("roi"), frame.shape[1], frame.shape[0])

        annotated_frame = frame.copy() if annotate else frame
        boxes, scores, tracked_ids, kpts = self._pose_estimate_with_onnx(frame)
        frame_detections: List[Dict[str, Any]] = []
        event_triggered = False

        if (
            tracked_ids is not None
            and isinstance(tracked_ids, (list, np.ndarray))
            and len(tracked_ids) > 0
            and boxes is not None
        ):
            pose_result = [
                self._PoseResult(i, j, k, m)
                for i, j, k, m in zip(boxes, scores, tracked_ids, kpts)
            ]
            pose_result = [
                pose
                for pose in pose_result
                if is_point_in_any_roi(
                    (
                        float((pose.xyxy[0] + pose.xyxy[2]) / 2.0),
                        float((pose.xyxy[1] + pose.xyxy[3]) / 2.0),
                    ),
                    roi_polys,
                )
            ]
            annotated_frame, event_triggered, frame_detections = self._detect_and_draw(
                pose_result,
                annotated_frame,
                cgr_conf=cgr_conf,
                skeleton=skeleton,
                cig_box=cig_box,
                threshold=threshold,
            )

        if annotate and annotated_frame is not None:
            draw_roi_overlays(annotated_frame, roi_polys, color=(0, 255, 255), thickness=3)

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0.0
        if annotate and annotated_frame is not None:
            cv2.putText(
                annotated_frame,
                f"FPS: {round(fps, 2)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        detection_by_track: Dict[int, Dict[str, Any]] = {
            int(d.get("track_id")): d
            for d in frame_detections
            if d.get("track_id") is not None
        }

        violating_tracks = [
            {
                "track_id": track_id,
                "violation_type": "smoking_behavior",
                "smoke_score": int(condition[1]),
                "confidence": detection_by_track.get(track_id, {}).get("confidence", 0.0),
                "bbox": detection_by_track.get(track_id, {}).get("bbox"),
                "event_type": "Giám sát hành vi hút thuốc",
                "description": "Phát hiện đối tượng có hành vi hút thuốc",
            }
            for track_id, condition in self._ids.items()
            if int(condition[1]) > threshold
        ]
        compliant_tracks = [
            {
                "track_id": track_id,
                "smoke_score": int(condition[1]),
                "bbox": detection_by_track.get(track_id, {}).get("bbox"),
            }
            for track_id, condition in self._ids.items()
            if int(condition[1]) <= threshold
        ]

        metadata = self._build_metadata(
            event_triggered=event_triggered,
            threshold=threshold,
            cgr_conf=cgr_conf,
            fps=fps,
            elapsed_time=elapsed_time,
            roi_count=len(roi_polys),
            detections=frame_detections,
            violations=violating_tracks,
            compliant=compliant_tracks,
        )

        return DetectionResult(
            frame=annotated_frame,
            event=event_triggered,
            metadata=metadata,
        )

    def _build_metadata(
        self,
        *,
        event_triggered: bool,
        threshold: int,
        cgr_conf: float,
        fps: float,
        elapsed_time: float,
        roi_count: int,
        detections: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
        compliant: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "type": "Giám sát hành vi hút thuốc",
            "eventType": "Giám sát hành vi hút thuốc",
            "title": "Giám sát hành vi hút thuốc",
            "severity": "high" if event_triggered else "low",
            "description": "Phát hiện hành vi hút thuốc trong khu vực giám sát",
            "model_type": "smoking_behavior",
            "threshold": threshold,
            "cgr_conf": cgr_conf,
            "fps": fps,
            "elapsed_time": elapsed_time,
            "detections": detections,
            "count": len(detections),
            "track_count": len(self._ids),
            "roi_count": roi_count,
            "violation_count": len(violations),
            "violations": violations,
            "compliant": compliant,
        }

        if violations:
            metadata["violation"] = "smoking_behavior"
            metadata["violation_type"] = "smoking_behavior"
            metadata["confidence"] = float(violations[0].get("confidence", 0.0))
            metadata["track_id"] = violations[0]["track_id"]
            if violations[0].get("bbox"):
                metadata["bbox"] = violations[0]["bbox"]

        return metadata

    def _build_visual_assets(self) -> None:
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self._palette = [self._hex2rgb(f"#{c}") for c in hexs]
        self._palette_len = len(self._palette)
        pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )
        self._kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self._limb_color = pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        self._skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

    @staticmethod
    def _hex2rgb(h: str) -> Tuple[int, int, int]:
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

    def _prepare_input(self, image: np.ndarray):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img, border = self._proportional_resize_with_padding(input_img, (640, 640))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor, border

    def _process_output(self, output, conf_threshold, iou_threshold, img, inputimg, border):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_threshold, :]
        scores = scores[scores > conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self._extract_boxes(predictions, img, inputimg, border)
        indices = nms(boxes, scores, iou_threshold)
        return boxes[indices], scores[indices], class_ids[indices]

    def _extract_boxes(self, predictions, ori, inputimg, border):
        boxes = predictions[:, :4]
        boxes = self._rscale_box_with_padding(
            (border[4], border[5]),
            boxes,
            (ori.shape[1], ori.shape[0]),
            border,
        )
        boxes = xywh2xyxy(boxes)
        return boxes

    def _proportional_resize_with_padding(self, img: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        try:
            original_height, original_width, _ = img.shape
            width_ratio = new_shape[1] / original_width
            height_ratio = new_shape[0] / original_height
            resize_ratio = min(width_ratio, height_ratio)
            new_width = int(original_width * resize_ratio)
            new_height = int(original_height * resize_ratio)
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            top = (new_shape[0] - new_height) // 2
            bottom = new_shape[0] - new_height - top
            left = (new_shape[1] - new_width) // 2
            right = new_shape[1] - new_width - left
            padded_img = cv2.copyMakeBorder(
                resized_img,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
            )
            return padded_img, (top, bottom, left, right, new_width, new_height)
        except Exception as e:
            self.logger.warning("Resize error: %s", e)
            return img, (0, 0, 0, 0, img.shape[1], img.shape[0])

    @staticmethod
    def _xywh2xyxy_rescale(x, scale, is_scale):
        y = np.copy(x)
        if is_scale:
            y[..., 0] = x[..., 0] * scale
            y[..., 1] = x[..., 1] * scale
            y[..., 2] = (x[..., 0] + x[..., 2]) * scale
            y[..., 3] = (x[..., 1] + x[..., 3]) * scale
        else:
            y[..., 0] = x[..., 0]
            y[..., 1] = x[..., 1]
            y[..., 2] = x[..., 0] + x[..., 2]
            y[..., 3] = x[..., 1] + x[..., 3]
        return y

    def _rscale_box_with_padding(self, original_size, boxes, target_size, border):
        original_width, original_height = original_size
        target_width, target_height = target_size
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        boxes -= np.array([border[2], border[0], 0, 0])
        boxes *= np.array([scale_x, scale_y, scale_x, scale_y])
        return boxes

    def _cgr_detect_with_onnx(self, image):
        img, border = self._prepare_input(image)
        input_tensor = ov.Tensor(array=img)
        self._infer_cgr.set_input_tensor(input_tensor)
        self._infer_cgr.infer()
        output = self._infer_cgr.get_output_tensor()
        output_buffer = output.data
        boxes, scores, class_ids = self._process_output(output_buffer, 0.25, 0.7, image, img, border)
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        if isinstance(boxes, np.ndarray) and boxes.shape[0] != 0:
            return boxes, scores
        return [], []

    def _pose_estimate_with_onnx(self, frame):
        height, width, _ = frame.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = frame
        scale = length / 640
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        outputs = self._infer_pose.infer(blob)[self._output_node]
        outputs = np.array([cv2.transpose(outputs[0])])

        classes_scores = outputs[:, :, 4]
        key_points = outputs[:, :, 5:]
        mask = classes_scores >= 0.5
        filtered_outputs = outputs[mask]

        if filtered_outputs.size == 0:
            return [], [], [], []

        boxes = filtered_outputs[:, 0:4] - np.column_stack(
            [
                (0.5 * filtered_outputs[:, 2]),
                (0.5 * filtered_outputs[:, 3]),
                np.zeros_like(filtered_outputs[:, 2]),
                np.zeros_like(filtered_outputs[:, 3]),
            ]
        )
        scores = filtered_outputs[:, 4]
        preds_kpts = key_points[mask]
        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.5, 0.5)
        if len(result_boxes) == 0:
            return [], [], [], []

        box = np.array(boxes)[result_boxes].reshape(-1, 4)
        box = self._xywh2xyxy_rescale(box, scale, True)
        scores = np.array(scores)[result_boxes]
        cls = np.zeros(box.shape[0])
        box, result = bytetrack(box, scores, cls, self._tracker)
        if isinstance(box, np.ndarray) and box.shape[0] != 0:
            box = self._xywh2xyxy_rescale(box, scale, False)
        kpts = np.array(preds_kpts)[result_boxes].reshape(-1, 17, 3) * scale
        return box, scores, result, kpts

    def _judge_smoke(self, pose_result, img, label):
        k = pose_result.keypoints
        left_angle, right_angle = self._cal_angle(k)
        left_hand_index = 9
        right_hand_index = 10
        if int(left_angle) < 55 or self._cal_dis(k, left_hand_index) < 0.8:
            if self._cgr_detect(pose_result, img, left_hand_index, label):
                return 2
            return 1

        if int(right_angle) < 55 or self._cal_dis(k, right_hand_index) < 0.8:
            if self._cgr_detect(pose_result, img, right_hand_index, label):
                return 2
            return 1

        return 0

    def _detect_and_draw(self, pose_result, img, *, cgr_conf: float, skeleton: bool, cig_box: bool, threshold: int):
        if not isinstance(self._ids, dict):
            self._ids = {}

        self._cgr_conf = cgr_conf
        cgrlabel = []
        event_triggered = False
        detections: List[Dict[str, Any]] = []

        for d in pose_result:
            conf, idd = float(d.conf), None if d.id is None else int(d.id)
            if idd is None:
                continue

            if idd not in self._ids:
                self._ids[idd] = np.array([idd, 0])

            condition = self._ids[idd]
            status = self._judge_smoke(d, img, cgrlabel)

            if status == 2:
                if condition[1] < 100:
                    condition[1] += 10
                if condition[1] < threshold:
                    self._box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
            elif status == 1:
                if condition[1] > 0:
                    condition[1] -= 1
                self._box_label(d.xyxy, img, 3, "Suspicious", (28, 172, 255))
            else:
                if condition[1] > 0:
                    condition[1] -= 1

            if condition[1] > threshold:
                event_triggered = True
                self._box_label(d.xyxy, img, 3, "Target is Smoking", (0, 0, 255))

            smoke_score = int(condition[1])
            is_smoking = smoke_score > threshold
            box = [int(v) for v in d.xyxy[:4]]
            detections.append(
                {
                    "class_id": 0,
                    "class_name": "smoking_behavior",
                    "display_name": "Target is Smoking" if is_smoking else "Suspicious",
                    "label": "Target is Smoking" if is_smoking else "Suspicious",
                    "confidence": conf,
                    "bbox": box,
                    "track_id": idd,
                    "smoke_score": smoke_score,
                    "is_smoking": is_smoking,
                }
            )

            self._ids[idd] = condition
            if skeleton:
                self._key_label(d.keypoints, img, img.shape, kpt_line=True)

        cgr_box = np.array([t[:4] for t in cgrlabel]) if cgrlabel else np.array([])
        if cig_box and len(cgr_box) > 0:
            for i in cgr_box:
                self._box_label(i, img, 3, label="Cig", color=(0, 0, 255), txt_color=(255, 255, 255))

        return img, event_triggered, detections

    def _cal_dis(self, kpt, direction):
        nose, wrist, shoulder, hip = kpt[0], kpt[direction], kpt[5], kpt[11]
        difference = nose - wrist
        standard = shoulder - hip
        distance = np.linalg.norm(difference)
        standdis = np.linalg.norm(standard)
        return distance / standdis if standdis > 0 else 0.0

    def _cal_angle(self, kpt):
        lshoulder, lelbow, lwrist = kpt[5], kpt[7], kpt[9]
        rshoulder, relbow, rwrist = kpt[6], kpt[8], kpt[10]
        left_shoulder_vector = lshoulder - lelbow
        left_wrist_vector = lwrist - lelbow
        right_shoulder_vector = rshoulder - relbow
        right_wrist_vector = rwrist - relbow
        left_denominator = np.linalg.norm(left_shoulder_vector) * np.linalg.norm(left_wrist_vector)
        right_denominator = np.linalg.norm(right_shoulder_vector) * np.linalg.norm(right_wrist_vector)
        left_angle_radian = np.arccos(np.clip(np.dot(left_shoulder_vector, left_wrist_vector) / left_denominator, -1.0, 1.0)) if left_denominator > 0 else 0.0
        right_angle_radian = np.arccos(np.clip(np.dot(right_shoulder_vector, right_wrist_vector) / right_denominator, -1.0, 1.0)) if right_denominator > 0 else 0.0
        right_angle_degree = np.degrees(right_angle_radian)
        left_angle_degree = np.degrees(left_angle_radian)
        return left_angle_degree, right_angle_degree

    def _box_label(self, box, im, lw, label='', color=(255, 255, 64), txt_color=(255, 255, 255)):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)
            cv2.putText(
                im,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    def _cgr_detect(self, k, img, direction, label):
        self._count += 1
        box = k.xyxy
        right = k.keypoints[0]
        length = int(0.4 * (box[2] - box[0]))
        lengths = int(0.3 * (box[3] - box[1]))
        box = box.astype(np.int32)
        box[1] = np.max([int(right[1]) - length, 0])
        box[3] = np.min([int(right[1]) + length, img.shape[0]])
        box[0] = np.max([int(right[0]) - lengths, 0])
        box[2] = np.min([int(right[0]) + lengths, img.shape[1]])
        person = img[box[1]:box[3], box[0]:box[2]]

        if person.shape[0] != 0 and person.shape[1] != 0:
            boxes, scores = self._cgr_detect_with_onnx(person)
            if len(scores) > 0:
                for i, c in enumerate(scores):
                    if c > self._cgr_conf:
                        label.append(
                            [
                                int(boxes[i][0]) + int(box[0]),
                                int(boxes[i][1]) + int(box[1]),
                                int(boxes[i][2]) + int(box[0]),
                                int(boxes[i][3]) + int(box[1]),
                                c,
                            ]
                        )
                        return True
            return False
        return False

    def _key_label(self, kpts, im, shape=(640, 640), radius=5, kpt_line=True):
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self._kpt_color[i]]
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.4:
                        continue
                cv2.circle(im, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self._skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(im, pos1, pos2, [int(x) for x in self._limb_color[i]], thickness=2, lineType=cv2.LINE_AA)
