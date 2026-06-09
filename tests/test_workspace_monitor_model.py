from types import SimpleNamespace

import numpy as np

from src.ai_models.workspace_monitor_model import WorkspaceMonitorModel


def _make_model():
    model = WorkspaceMonitorModel.__new__(WorkspaceMonitorModel)
    model.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    model.ppe_conflict_margin = 0.1
    model.ppe_conflict_iou_threshold = 0.1
    model.ppe_model = SimpleNamespace(
        names={0: "EVNWorker", 1: "SmartTech", 2: "no-vest"}
    )
    return model


def test_configure_ppe_classes_reads_model_metadata():
    model = _make_model()

    model._configure_ppe_classes()

    assert model.ppe_violation_classes == {2: "No Vest"}
    assert model.ppe_safe_classes == {}
    assert model.ppe_classes_to_track == [2]


def test_match_ppe_to_person_track_prefers_person_containing_bbox_center():
    model = _make_model()

    tracked_people = [
        {"track_id": 11, "bbox": [100, 80, 220, 420]},
        {"track_id": 22, "bbox": [260, 80, 380, 420]},
    ]

    matched_track_id = model._match_ppe_to_person_track(
        [120, 180, 190, 330], tracked_people
    )

    assert matched_track_id == 11


def test_should_suppress_no_vest_when_vest_exists_on_same_person():
    model = _make_model()

    no_vest_detection = {
        "bbox": [120, 180, 190, 330],
        "confidence": 0.48,
        "person_track_id": 11,
    }
    vest_detection = {
        "bbox": [118, 178, 192, 332],
        "confidence": 0.44,
        "person_track_id": 11,
    }

    assert model._should_suppress_no_vest(no_vest_detection, vest_detection) is True


def test_should_not_suppress_no_vest_for_other_person():
    model = _make_model()

    no_vest_detection = {
        "bbox": [120, 180, 190, 330],
        "confidence": 0.48,
        "person_track_id": 11,
    }
    vest_detection = {
        "bbox": [260, 180, 330, 330],
        "confidence": 0.90,
        "person_track_id": 22,
    }

    assert model._should_suppress_no_vest(no_vest_detection, vest_detection) is False


def test_resolve_workspace_roi_input_promotes_stable_auto_roi():
    model = WorkspaceMonitorModel.__new__(WorkspaceMonitorModel)
    model.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    model.auto_roi_enabled = True
    model.auto_roi_source = "reflective_wire"
    model.auto_roi_min_conf = 0.4
    model.auto_roi_update_interval = 1
    model.auto_roi_stable_frames = 2
    model.auto_roi_lock = False
    model._auto_roi_similarity_threshold = 0.08
    model._auto_roi_last = None
    model._auto_roi_last_confidence = 0.0
    model._auto_roi_candidate = None
    model._auto_roi_candidate_count = 0
    model._auto_roi_locked = False
    detections = [
        SimpleNamespace(
            roi=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            confidence=0.82,
        ),
        SimpleNamespace(
            roi=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            confidence=0.83,
        ),
    ]
    model._auto_roi_detector = SimpleNamespace(detect=lambda _frame: detections.pop(0))

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manual_roi = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    model._frame_counter = 1
    roi_first, meta_first = model._resolve_workspace_roi_input(frame, {"roi": manual_roi})
    assert roi_first == manual_roi
    assert meta_first["auto_roi_used"] is False

    model._frame_counter = 2
    roi_second, meta_second = model._resolve_workspace_roi_input(
        frame, {"roi": manual_roi}
    )
    assert roi_second == [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    assert meta_second["auto_roi_used"] is True
    assert meta_second["roi_source"] == "auto_reflective_wire"


def test_resolve_workspace_roi_input_uses_locked_auto_roi():
    model = WorkspaceMonitorModel.__new__(WorkspaceMonitorModel)
    model.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    model.auto_roi_enabled = True
    model.auto_roi_source = "reflective_wire"
    model.auto_roi_min_conf = 0.4
    model.auto_roi_update_interval = 15
    model.auto_roi_stable_frames = 2
    model.auto_roi_lock = True
    model._auto_roi_similarity_threshold = 0.08
    model._auto_roi_last = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    model._auto_roi_last_confidence = 0.91
    model._auto_roi_candidate = None
    model._auto_roi_candidate_count = 0
    model._auto_roi_locked = True
    model._auto_roi_detector = None
    model._frame_counter = 5

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    manual_roi = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    resolved_roi, meta = model._resolve_workspace_roi_input(
        frame,
        {
            "roi": manual_roi,
            "auto_roi_enabled": True,
            "auto_roi_lock": True,
        },
    )

    assert resolved_roi == model._auto_roi_last
    assert meta["auto_roi_used"] is True
    assert meta["auto_roi_status"] == "locked"
