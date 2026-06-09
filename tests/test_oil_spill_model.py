import pytest

import src.ai_models.oil_spill_model as oil_spill_module


class DummyYOLO:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def to(self, device):
        self.device = device
        return self


@pytest.fixture
def oil_spill_model(monkeypatch):
    monkeypatch.setattr(oil_spill_module, "YOLO", DummyYOLO)
    return oil_spill_module.OilSpillModel(
        model_path="dummy.pt",
        aux_model_path="dummy_aux.pt",
        min_area_growth_ratio=0.08,
        max_area_shrink_ratio=0.15,
    )


def test_oil_spill_candidate_requires_real_growth_and_stable_center(oil_spill_model):
    pending = {
        "initial_center": (100, 100),
        "previous_center": (100, 100),
        "center": (100, 100),
        "initial_area": 1000,
        "previous_area": 1000,
        "area": 1000,
    }

    assert not oil_spill_model._is_stable_candidate(
        pending,
        center=(520, 100),
        bbox_area=1000,
        motion_ratio=0.02,
        edge_motion_ratio=0.02,
        frame_w=1920,
        frame_h=1080,
    )
    # No growth AND no accumulated persistence (no count/first_seen) ->
    # both confirmation paths must reject.
    assert not oil_spill_model._has_confirmation_signal(
        pending, bbox_area=1000, current_time=100.0
    )


def test_oil_spill_candidate_accepts_small_drift_only_when_area_grows(oil_spill_model):
    pending = {
        "initial_center": (100, 100),
        "previous_center": (100, 100),
        "center": (100, 100),
        "initial_area": 1000,
        "previous_area": 1000,
        "area": 1000,
        "motion_history": [
            {"timestamp": 1.0, "center": (100, 100), "area": 1000},
            {"timestamp": 2.0, "center": (102, 101), "area": 1030},
        ],
    }

    assert oil_spill_model._is_stable_candidate(
        pending,
        center=(104, 102),
        bbox_area=1085,
        motion_ratio=0.02,
        edge_motion_ratio=0.02,
        frame_w=1920,
        frame_h=1080,
    )
    assert oil_spill_model._has_confirmation_signal(
        pending, bbox_area=1085, current_time=100.0
    )


def test_oil_spill_rejects_flat_history_without_persistence(oil_spill_model):
    """Flat (non-growing) history without accumulated count/duration is rejected."""
    pending = {
        "initial_center": (100, 100),
        "previous_center": (101, 100),
        "center": (101, 100),
        "initial_area": 1000,
        "previous_area": 1002,
        "area": 1002,
        "motion_history": [
            {"timestamp": 1.0, "center": (100, 100), "area": 1000},
            {"timestamp": 2.0, "center": (101, 100), "area": 1001},
            {"timestamp": 3.0, "center": (102, 100), "area": 1002},
        ],
    }

    # Path A fails (growth ~0.2% < 8%). Path B fails (no count/first_seen).
    assert not oil_spill_model._has_confirmation_signal(
        pending, bbox_area=1002, current_time=100.0
    )


def test_oil_spill_accepts_sustained_static_region(oil_spill_model):
    """
    Regression: an already-formed static spill (no growth but observed
    persistently for long enough) MUST be confirmed. The previous
    growth-only gate dropped these silently.
    """
    first_seen = 100.0
    now = 105.0  # 5 seconds of sustained observation
    pending = {
        "first_seen": first_seen,
        "count": 12,
        "initial_center": (100, 100),
        "previous_center": (100, 100),
        "center": (100, 100),
        "initial_area": 2000,
        "previous_area": 2000,
        "area": 2000,
        "motion_history": [
            {"timestamp": first_seen + i * 0.5, "center": (100, 100), "area": 2000}
            for i in range(6)
        ],
    }

    assert oil_spill_model._has_confirmation_signal(
        pending, bbox_area=2000, current_time=now
    )


def test_oil_spill_rejects_shrinking_region(oil_spill_model):
    """Actively shrinking regions are never real spills."""
    pending = {
        "first_seen": 100.0,
        "count": 12,
        "initial_center": (100, 100),
        "previous_center": (100, 100),
        "center": (100, 100),
        "initial_area": 3000,
        "previous_area": 3000,
        "area": 2000,  # shrank 33%
        "motion_history": [
            {"timestamp": 100.0 + i * 0.5, "center": (100, 100), "area": 3000}
            for i in range(6)
        ],
    }

    assert not oil_spill_model._has_confirmation_signal(
        pending, bbox_area=2000, current_time=106.0
    )