from src.utils.parameter_validator import (
    validate_additional_parameters,
    validate_general_parameters,
    validate_parameter_value,
)
from src.utils.roi_utils import get_effective_roi, get_roi_polygons, is_roi_polygon_set


def test_validate_parameter_value_accepts_single_polygon_roi_for_workspace_monitor():
    roi = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]

    is_valid, error = validate_parameter_value(
        "roi", roi, model_type="workspace_monitor"
    )

    assert is_valid is True
    assert error is None


def test_validate_parameter_value_accepts_polygon_set_roi_for_workspace_monitor():
    roi = [
        [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
        [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
    ]

    is_valid, error = validate_parameter_value(
        "roi", roi, model_type="workspace_monitor"
    )

    assert is_valid is True
    assert error is None


def test_validate_parameter_value_rejects_polygon_set_over_limit():
    roi = [
        [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
        [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]],
        [[0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.2, 0.3]],
        [[0.3, 0.3], [0.4, 0.3], [0.4, 0.4], [0.3, 0.4]],
        [[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]],
        [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6], [0.5, 0.6]],
    ]

    is_valid, error = validate_parameter_value(
        "roi", roi, model_type="workspace_monitor"
    )

    assert is_valid is False
    assert error == "ROI polygon set can have max 5 polygons, got 6"


def test_validate_additional_parameters_accepts_polygon_set_for_oil_spill():
    is_valid, errors = validate_additional_parameters(
        "oil_spill",
        {
            "roi": [
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
            ]
        },
    )

    assert is_valid is True
    assert errors == []


def test_is_roi_polygon_set_detects_shape_correctly():
    assert is_roi_polygon_set([[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]]) is True
    assert is_roi_polygon_set([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]) is False
    assert is_roi_polygon_set([]) is False


def test_get_roi_polygons_normalizes_single_polygon_and_polygon_set():
    single_polygon = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    polygon_set = [
        single_polygon,
        [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]],
    ]

    assert get_roi_polygons(single_polygon) == [single_polygon]
    assert get_roi_polygons(polygon_set) == polygon_set
    assert get_roi_polygons(None) == []


def test_get_effective_roi_returns_first_polygon_for_backward_compatibility():
    polygon_a = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    polygon_b = [[0.2, 0.2], [0.4, 0.2], [0.4, 0.4], [0.2, 0.4]]

    assert get_effective_roi(polygon_a) == polygon_a
    assert get_effective_roi([polygon_a, polygon_b]) == polygon_a
    assert get_effective_roi(None) is None


def test_validate_additional_parameters_accepts_workspace_auto_roi_settings():
    is_valid, errors = validate_additional_parameters(
        "workspace_monitor",
        {
            "auto_roi_enabled": True,
            "auto_roi_source": "reflective_wire",
            "auto_roi_min_conf": 0.45,
            "auto_roi_update_interval": 10,
            "auto_roi_stable_frames": 3,
            "auto_roi_lock": False,
        },
        strict=True,
    )

    assert is_valid is True
    assert errors == []


def test_validate_parameter_value_rejects_invalid_auto_roi_min_conf():
    is_valid, error = validate_parameter_value(
        "auto_roi_min_conf", 1.5, model_type="workspace_monitor"
    )

    assert is_valid is False
    assert error == "auto_roi_min_conf must be <= 1.0, got 1.5"


def test_validate_general_parameters_accepts_oil_cap_detection_settings():
    is_valid, errors = validate_general_parameters(
        "oil_cap_detection",
        {
            "conf_threshold": 0.45,
            "iou_threshold": 0.45,
            "detection_cooldown": 300,
            "track_match_iou_threshold": 0.5,
            "track_stale_timeout": 3.0,
            "violation_min_center_y_ratio": 0.35,
            "violation_labels": ["oil_cap_opened"],
        },
        strict=True,
    )

    assert is_valid is True
    assert errors == []


def test_validate_additional_parameters_accepts_polygon_set_for_oil_cap_detection():
    is_valid, errors = validate_additional_parameters(
        "oil_cap_detection",
        {
            "roi": [
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
            ],
            "max_roi_polygons": 2,
        },
        strict=True,
    )

    assert is_valid is True
    assert errors == []


def test_validate_parameter_value_rejects_non_string_violation_labels():
    is_valid, error = validate_parameter_value(
        "violation_labels",
        ["oil_cap_opened", 123],
        model_type="oil_cap_detection",
    )

    assert is_valid is False
    assert error == "violation_labels must contain only strings"
