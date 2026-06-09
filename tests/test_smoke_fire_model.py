import logging
import threading

import numpy as np

from src.ai_models.base_model import DetectionResult
from src.ai_models.smoke_fire_model import SmokeFireModel
from src.core.camera_single_thread import SingleThreadProcessor
from src.utils.parameter_validator import validate_additional_parameters


class FakeTrackIds:
    def __init__(self, values):
        self._values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeBox:
    def __init__(self, xyxy, conf, cls_id):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [float(conf)]
        self.cls = [int(cls_id)]


class FakeBoxes:
    def __init__(self, boxes, track_ids=None):
        self._boxes = boxes
        self.id = FakeTrackIds(track_ids) if track_ids is not None else None

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)

    def __bool__(self):
        return bool(self._boxes)


class FakeResult:
    def __init__(self, names, boxes):
        self.names = names
        self.boxes = boxes


def build_smoke_fire_model(results, run_yolo_track=None):
    class DummyTracker:
        def should_record_event(self, *_args, **_kwargs):
            return True

    class DummyAuxModel:
        def predict(self, *args, **kwargs):
            return []

    model = SmokeFireModel.__new__(SmokeFireModel)
    model.model = object()
    model.aux_model = DummyAuxModel()
    model.device = "cpu"
    model._yolo_track_available = True
    model.conf_threshold = 0.30
    model.fire_conf_threshold = 0.30
    model.smoke_conf_threshold = 0.25
    model.track_conf_threshold = 0.25
    model.iou_threshold = 0.3
    model.detection_time_threshold = 0
    model.smoke_persist_seconds = 0
    model._stale_timeout = 5.0
    model._pending_detections = {}
    model.min_bbox_area_ratio = 0.0008
    model.max_bbox_area_ratio = 0.60
    model.tail_light_max_area_ratio = 0.012
    model.tail_light_min_red_ratio = 0.55
    model.tail_light_max_orange_ratio = 0.18
    model.tail_light_min_bright_ratio = 0.35
    model.tail_light_max_value_std = 0.10
    model.tail_light_min_aspect_ratio = 0.35
    model.tail_light_max_aspect_ratio = 3.50
    model.tail_light_max_motion_ratio = 0.06
    model.tail_light_motion_diff_threshold = 18
    model.detection_iou_merge_threshold = 0.75
    model.foreground_conf_threshold = 0.35
    model.foreground_iou_threshold = 0.35
    model.foreground_cover_ratio_threshold = 0.60
    model.foreground_proximity_ratio = 0.18
    model.recent_foreground_hold_seconds = 1.0
    model._recent_foreground_boxes = []
    model.smoke_max_saturation = 60.0
    model._last_tracked_boxes = {}
    model._next_synthetic_id = 900000
    model._prev_gray_frame = None
    model._state_lock = threading.Lock()
    model.target_classes = {"fire", "smoke"}
    model.object_tracker = DummyTracker()
    model.class_name_mapping = {
        "Fire": "Fire",
        "Smoke": "Smoke",
        "fire": "Fire",
        "smoke": "Smoke",
    }
    model.colors = {
        "Fire": (0, 0, 255),
        "Smoke": (128, 128, 128),
        "fire": (0, 0, 255),
        "smoke": (128, 128, 128),
    }
    model.logger = logging.getLogger("tests.smoke_fire_model")
    model._cleanup_stale_pending = lambda: None
    model._resolve_track_id = lambda track_id, bbox, class_name: track_id if track_id is not None else {
        "fire": 101,
        "smoke": 202,
    }[str(class_name).lower()]
    model._run_yolo_track = run_yolo_track or (lambda *args, **kwargs: results)
    return model


def test_smoke_fire_filters_non_target_classes_and_preserves_original_class_ids():
    result = FakeResult(
        names={0: "fire", 1: "other", 2: "smoke"},
        boxes=FakeBoxes(
            [
                FakeBox((10, 10, 30, 30), 0.91, 0),
                FakeBox((35, 10, 55, 30), 0.88, 1),
                FakeBox((10, 35, 30, 55), 0.86, 2),
            ]
        ),
    )
    model = build_smoke_fire_model([result])

    output = model.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), annotate=False)
    detections = output.metadata["detections"]

    assert [d["class_name"] for d in detections] == ["Fire", "Smoke"]
    assert [d["class_id"] for d in detections] == [0, 2]
    assert output.metadata["count"] == 2
    assert [v["violation_type"] for v in output.metadata["violations"]] == ["Fire", "Smoke"]


def test_smoke_fire_uses_inference_frame_for_model_input():
    captured = {}

    def fake_run_yolo_track(model_obj, source, **kwargs):
        captured["source"] = source
        return []

    model = build_smoke_fire_model([], run_yolo_track=fake_run_yolo_track)
    display_frame = np.full((32, 32, 3), 255, dtype=np.uint8)
    raw_frame = np.zeros_like(display_frame)

    model.process_frame(display_frame, annotate=False, inference_frame=raw_frame)

    assert captured["source"] is raw_frame


def test_smoke_fire_uses_lower_smoke_threshold_than_fire():
    result = FakeResult(
        names={0: "fire", 2: "smoke"},
        boxes=FakeBoxes(
            [
                FakeBox((10, 10, 22, 22), 0.27, 0),
                FakeBox((30, 30, 42, 42), 0.27, 2),
            ]
        ),
    )
    model = build_smoke_fire_model([result])

    output = model.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), annotate=False)

    assert [d["class_name"] for d in output.metadata["detections"]] == ["Smoke"]
    assert output.metadata["detections"][0]["class_id"] == 2


def test_smoke_fire_limits_per_frame_detection_overflow():
    boxes = [
        FakeBox((5 + i * 12, 10, 15 + i * 12, 24), 0.95 - (i * 0.01), 2)
        for i in range(6)
    ]
    result = FakeResult(
        names={2: "smoke"},
        boxes=FakeBoxes(boxes, track_ids=[200 + i for i in range(6)]),
    )
    model = build_smoke_fire_model([result])
    model.max_detections_per_frame = 3
    model.max_raw_detections = 20

    output = model.process_frame(np.zeros((120, 120, 3), dtype=np.uint8), annotate=False)

    assert output.metadata["count"] == 3
    assert output.metadata["dropped_overflow_detections"] == 3


def test_smoke_fire_merges_overlapping_same_class_boxes():
    result = FakeResult(
        names={2: "smoke"},
        boxes=FakeBoxes(
            [
                FakeBox((10, 10, 40, 40), 0.94, 2),
                FakeBox((12, 12, 42, 42), 0.92, 2),
            ],
            track_ids=[201, 202],
        ),
    )
    model = build_smoke_fire_model([result])
    model.detection_iou_merge_threshold = 0.70

    output = model.process_frame(np.zeros((100, 100, 3), dtype=np.uint8), annotate=False)

    assert output.metadata["count"] == 1


def test_smoke_fire_rejects_tail_light_like_fire_detection():
    result = FakeResult(
        names={0: "fire"},
        boxes=FakeBoxes([FakeBox((10, 10, 20, 22), 0.92, 0)]),
    )
    model = build_smoke_fire_model([result])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:22, 10:20] = (0, 0, 255)

    output = model.process_frame(frame, annotate=False)

    assert output.metadata["detections"] == []
    assert output.event is False


def test_smoke_fire_keeps_real_fire_like_patch():
    result = FakeResult(
        names={0: "fire"},
        boxes=FakeBoxes([FakeBox((10, 10, 20, 22), 0.92, 0)]),
    )
    model = build_smoke_fire_model([result])
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[10:16, 10:20] = (0, 140, 255)
    frame[16:22, 10:20] = (0, 220, 255)
    frame[13:19, 12:18] = (0, 0, 255)

    output = model.process_frame(frame, annotate=False)

    assert [d["class_name"] for d in output.metadata["detections"]] == ["Fire"]
    assert output.event is True


def test_single_thread_processor_passes_raw_frame_to_smoke_fire_inference():
    class DummyAnnotator:
        def process_frame(self, frame, **kwargs):
            annotated = np.full_like(frame, 255)
            return DetectionResult(frame=annotated, event=False, metadata={})

    class DummySmokeModel:
        def __init__(self):
            self.display_frame = None
            self.inference_frame = None

        def process_frame(self, frame, **kwargs):
            self.display_frame = frame.copy()
            self.inference_frame = kwargs.get("inference_frame")
            return DetectionResult(frame=frame.copy(), event=False, metadata={})

    smoke_model = DummySmokeModel()
    processor = SingleThreadProcessor.__new__(SingleThreadProcessor)
    processor.camera_id = "camera-test"
    processor._models_lock = threading.RLock()
    processor.models = {
        "annotator": {
            "model_instance": DummyAnnotator(),
            "model_id": "annotator",
            "model_name": "Annotator",
            "model_type": "helmet_detection",
            "additional_params": {},
        },
        "smoke": {
            "model_instance": smoke_model,
            "model_id": "smoke",
            "model_name": "Smoke Fire",
            "model_type": "smoke_fire",
            "additional_params": {},
        },
    }

    raw_frame = np.zeros((16, 16, 3), dtype=np.uint8)
    SingleThreadProcessor.process_frame(processor, raw_frame)

    assert np.all(smoke_model.display_frame == 0)
    assert smoke_model.inference_frame is raw_frame


def test_validate_additional_parameters_accepts_smoke_fire_tuning_overrides():
    is_valid, errors = validate_additional_parameters(
        "smoke_fire",
        {
            "roi": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "fire_conf_threshold": 0.3,
            "smoke_conf_threshold": 0.25,
            "min_bbox_area_ratio": 0.0008,
            "tail_light_max_area_ratio": 0.012,
        },
        strict=True,
    )

    assert is_valid is True
    assert errors == []
