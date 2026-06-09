import numpy as np
import pytest

import src.ai_models.petrolimex_detection_model as ppe_module


class DummyTensorLike:
    def __init__(self, values):
        self._values = values

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class DummyBox:
    def __init__(self, xyxy, conf, cls_id):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = [float(conf)]
        self.cls = [int(cls_id)]


class DummyBoxes(list):
    def __init__(self, boxes, ids):
        super().__init__(boxes)
        self.id = DummyTensorLike(ids)


class DummyResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class DummyYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        if model_path.endswith("petrolimex.pt"):
            self.names = {0: "vest", 1: "novest"}
            self.kind = "petrolimex"
        elif model_path.endswith("yolo11m.pt"):
            self.names = {0: "person"}
            self.kind = "people"
        else:
            self.names = {0: "Hardhat", 1: "No Hardhat"}
            self.kind = "hardhat"

    def to(self, device):
        self.device = device
        return self

    def track(self, source, **kwargs):
        return []


def make_result(box_specs, names, ids=None):
    boxes = [DummyBox(spec["bbox"], spec["conf"], spec["cls_id"]) for spec in box_specs]
    if ids is None:
        ids = [spec.get("track_id") for spec in box_specs]
    return [DummyResult(DummyBoxes(boxes, ids), names)]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(ppe_module, "YOLO", DummyYOLO)
    return ppe_module.PetrolimexDetectionModel(
        model_path="src/ai_models/model_weights/petrolimex.pt",
        no_hardhat_model_path="src/ai_models/model_weights/atld_92.pt",
        no_hardhat_conf_threshold=0.5,
    )


def test_petrolimex_builds_no_hardhat_class_mapping(model):
    assert model.no_hardhat_classes_to_keep == [1]
    assert model.no_hardhat_class_names[1] == "no_hardhat"


def test_petrolimex_records_no_hardhat_violation(model, monkeypatch):
    def fake_run_yolo_track(target_model, source, **kwargs):
        if target_model is model.model:
            return []
        if target_model is model.people_model:
            return []
        return make_result(
            [{"bbox": [30, 30, 90, 150], "conf": 0.88, "cls_id": 1, "track_id": 202}],
            model.no_hardhat_model.names,
        )

    monkeypatch.setattr(model, "_run_yolo_track", fake_run_yolo_track)

    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    result = model.process_frame(frame, annotate=False)

    assert result.event is True
    violations = result.metadata.get("violations", [])
    assert len(violations) == 1
    assert violations[0]["violation_type"] == "no_hardhat"


def test_people_aux_marks_person_positive_and_suppresses_novest(model, monkeypatch):
    def fake_run_yolo_track(target_model, source, **kwargs):
        if target_model is model.model:
            return make_result(
                [
                    {"bbox": [40, 45, 95, 150], "conf": 0.91, "cls_id": 0, "track_id": 101},
                    {"bbox": [38, 40, 98, 155], "conf": 0.62, "cls_id": 1, "track_id": 102},
                ],
                model.model.names,
            )
        if target_model is model.people_model:
            return make_result(
                [{"bbox": [20, 20, 120, 180], "conf": 0.87, "cls_id": 0, "track_id": 301}],
                model.people_model.names,
            )
        return []

    monkeypatch.setattr(model, "_run_yolo_track", fake_run_yolo_track)

    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    result = model.process_frame(frame, annotate=False)

    assert result.event is False
    assert result.metadata.get("violations", []) == []
    assert [d["class_name"] for d in result.metadata.get("detections", [])] == ["vest"]
    people_statuses = result.metadata.get("people_statuses", [])
    assert len(people_statuses) == 1
    assert people_statuses[0]["ppe_status"] == "positive"
    assert result.metadata["people_positive_count"] == 1
    assert result.metadata["people_negative_count"] == 0


def test_people_aux_marks_person_negative_when_only_person_detected(model, monkeypatch):
    def fake_run_yolo_track(target_model, source, **kwargs):
        if target_model is model.model:
            return []
        if target_model is model.people_model:
            return make_result(
                [{"bbox": [25, 20, 115, 175], "conf": 0.83, "cls_id": 0, "track_id": 401}],
                model.people_model.names,
            )
        return []

    monkeypatch.setattr(model, "_run_yolo_track", fake_run_yolo_track)

    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    result = model.process_frame(frame, annotate=False)

    assert result.event is True
    people_statuses = result.metadata.get("people_statuses", [])
    assert len(people_statuses) == 1
    assert people_statuses[0]["ppe_status"] == "negative"
    violations = result.metadata.get("violations", [])
    assert len(violations) == 1
    assert violations[0]["violation_type"] == "novest"
    assert result.metadata["people_positive_count"] == 0
    assert result.metadata["people_negative_count"] == 1


def test_people_aux_matches_vest_one_to_one_across_overlapping_people(model, monkeypatch):
    def fake_run_yolo_track(target_model, source, **kwargs):
        if target_model is model.model:
            return make_result(
                [{"bbox": [78, 45, 118, 120], "conf": 0.9, "cls_id": 0, "track_id": 601}],
                model.model.names,
            )
        if target_model is model.people_model:
            return make_result(
                [
                    {"bbox": [10, 20, 110, 180], "conf": 0.91, "cls_id": 0, "track_id": 701},
                    {"bbox": [70, 20, 170, 180], "conf": 0.89, "cls_id": 0, "track_id": 702},
                ],
                model.people_model.names,
            )
        return []

    monkeypatch.setattr(model, "_run_yolo_track", fake_run_yolo_track)

    frame = np.zeros((220, 240, 3), dtype=np.uint8)
    result = model.process_frame(frame, annotate=False)

    people_statuses = sorted(
        result.metadata.get("people_statuses", []),
        key=lambda item: item["track_id"],
    )
    assert len(people_statuses) == 2
    assert sorted(item["ppe_status"] for item in people_statuses) == ["negative", "positive"]
    assert result.metadata["people_positive_count"] == 1
    assert result.metadata["people_negative_count"] == 1
    negative_tracks = {
        item["track_id"] for item in people_statuses if item["ppe_status"] == "negative"
    }
    violations = result.metadata.get("violations", [])
    assert len(violations) == 1
    assert violations[0]["track_id"] in negative_tracks
    assert violations[0]["violation_type"] == "novest"


def test_run_yolo_track_falls_back_to_predict_when_scipy_missing(model):
    class TrackDependencyMissingModel:
        def __init__(self):
            self.track_calls = 0
            self.predict_calls = []

        def track(self, source, **kwargs):
            self.track_calls += 1
            raise ModuleNotFoundError("No module named 'scipy'")

        def predict(self, source, **kwargs):
            self.predict_calls.append(kwargs)
            return ["predicted"]

    backend = TrackDependencyMissingModel()
    frame = np.zeros((64, 64, 3), dtype=np.uint8)

    first_result = model._run_yolo_track(
        backend,
        frame,
        conf=0.55,
        persist=True,
    )
    second_result = model._run_yolo_track(
        backend,
        frame,
        conf=0.60,
        persist=True,
    )

    assert first_result == ["predicted"]
    assert second_result == ["predicted"]
    assert backend.track_calls == 1
    assert len(backend.predict_calls) == 2
    assert all("persist" not in call for call in backend.predict_calls)
    assert all("tracker" not in call for call in backend.predict_calls)
