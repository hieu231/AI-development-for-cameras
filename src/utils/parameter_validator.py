# src/utils/parameter_validator.py
"""
Parameter Validator - Validate model parameters based on model type
"""

from typing import Dict, Any, Tuple, Optional, List
from src.utils.roi_utils import validate_normalized_roi_input


# Parameter validation rules
PARAMETER_RULES = {
    "conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "iou_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "detection_cooldown": {"type": (int, float), "min": 0},
    "min_detections": {"type": int, "min": 1},
    "max_det": {"type": int, "min": 1},
    "detection_time_threshold": {"type": (int, float), "min": 0},
    "time_threshold": {"type": (int, float), "min": 0.0},
    "cooldown": {"type": (int, float), "min": 0},
    "fire_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "smoke_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "stranger_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "max_workers": {"type": int, "min": 1, "max": 100},
    "ppe_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "face_similarity_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "registered_worker_count": {"type": int, "min": 0, "max": 100},
    "auto_roi_enabled": {"type": bool},
    "auto_roi_source": {"type": str},
    "auto_roi_min_conf": {"type": float, "min": 0.0, "max": 1.0},
    "auto_roi_update_interval": {"type": int, "min": 1, "max": 3600},
    "auto_roi_stable_frames": {"type": int, "min": 1, "max": 120},
    "auto_roi_lock": {"type": bool},
    "min_bbox_area_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_bbox_area_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_bbox_height_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "min_aspect_ratio": {"type": float, "min": 0.0},
    "min_consecutive_frames": {"type": int, "min": 1},
    "person_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "person_iou_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "person_cover_ratio_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "foreground_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "foreground_iou_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "foreground_cover_ratio_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "no_hardhat_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "enable_people_aux_detection": {"type": bool},
    "track_ongoing_incidents": {"type": bool},
    "debug_detection_summary": {"type": bool},
    "debug_summary_interval_seconds": {"type": (int, float), "min": 0.5},
    "people_conf_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "people_match_iou_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "people_model_path": {"type": str},
    "max_motion_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_center_shift_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_area_change_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_area_shrink_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "min_area_growth_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "high_severity_area_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "critical_severity_area_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "global_event_cooldown": {"type": (int, float), "min": 0},
    "continuous_event_interval": {"type": (int, float), "min": 0},
    "incident_hold_seconds": {"type": (int, float), "min": 0},
    "flow_motion_ratio_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "flow_edge_motion_ratio_threshold": {
        "type": float,
        "min": 0.0,
        "max": 1.0,
    },
    "sudden_area_growth_ratio": {"type": float, "min": 0.0, "max": 10.0},
    "min_flow_frames": {"type": int, "min": 1},
    "max_incident_motion_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "max_incident_edge_motion_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_max_area_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_min_red_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_max_orange_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_min_bright_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_max_value_std": {"type": float, "min": 0.0, "max": 1.0},
    "tail_light_min_aspect_ratio": {"type": float, "min": 0.0},
    "tail_light_max_aspect_ratio": {"type": float, "min": 0.0},
    "tail_light_max_motion_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "detection_iou_merge_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "max_raw_detections": {"type": int, "min": 1},
    "max_detections_per_frame": {"type": int, "min": 1},
    "max_roi_polygons": {"type": int, "min": 1, "max": 5},
    "track_match_iou_threshold": {"type": float, "min": 0.0, "max": 1.0},
    "track_stale_timeout": {"type": (int, float), "min": 0.0},
    "violation_min_center_y_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "violation_labels": {"type": list},
    "cgr_conf": {"type": float, "min": 0.0, "max": 1.0},
    "threshold": {"type": int, "min": 0},
    "skeleton": {"type": bool},
    "cig_box": {"type": bool},
    "annotate": {"type": bool},
    "open_cap_max_top_y_ratio": {"type": float, "min": 0.0, "max": 1.0},
    "no_person_repeat_interval": {"type": (int, float), "min": 0.0},
    "skip_when_person_present": {"type": bool},
}


# Model type specific allowed parameters
MODEL_TYPE_GENERAL_PARAMS = {
    "object_detection": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "min_detections",
        "max_det",
    ],
    "helmet_detection": ["iou_threshold", "max_det", "min_detections"],
    "petrolimex_detection_model": [
        "conf_threshold",
        "iou_threshold",
        "person_conf_threshold",
        "no_hardhat_conf_threshold",
        "enable_people_aux_detection",
        "people_conf_threshold",
        "people_match_iou_threshold",
        "people_model_path",
        "detection_cooldown",
    ],
    "alpr": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "min_detections",
        "max_det",
    ],
    "oil_dumping": ["conf_threshold", "detection_time_threshold", "detection_cooldown"],
    "oil_spill": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "foreground_conf_threshold",
        "foreground_iou_threshold",
        "foreground_cover_ratio_threshold",
        "person_conf_threshold",
        "person_iou_threshold",
        "person_cover_ratio_threshold",
        "min_bbox_area_ratio",
        "max_bbox_height_ratio",
        "min_aspect_ratio",
        "max_motion_ratio",
        "max_center_shift_ratio",
        "max_area_change_ratio",
        "max_area_shrink_ratio",
        "min_area_growth_ratio",
        "min_consecutive_frames",
    ],
    "oil_cap_detection": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "track_match_iou_threshold",
        "track_stale_timeout",
        "violation_min_center_y_ratio",
        "violation_labels",
    ],
    "tran_dau": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "foreground_conf_threshold",
        "foreground_iou_threshold",
        "foreground_cover_ratio_threshold",
        "person_conf_threshold",
        "person_iou_threshold",
        "person_cover_ratio_threshold",
        "min_bbox_area_ratio",
        "max_bbox_height_ratio",
        "min_aspect_ratio",
        "min_consecutive_frames",
        "high_severity_area_ratio",
        "critical_severity_area_ratio",
        "global_event_cooldown",
        "continuous_event_interval",
        "track_ongoing_incidents",
        "debug_detection_summary",
        "debug_summary_interval_seconds",
        "incident_hold_seconds",
        "flow_motion_ratio_threshold",
        "flow_edge_motion_ratio_threshold",
        "sudden_area_growth_ratio",
        "min_flow_frames",
        "max_incident_motion_ratio",
        "max_incident_edge_motion_ratio",
    ],
    "people_control": [
        "conf_threshold",
        "time_threshold",
        "detection_cooldown",
        "cooldown",
    ],
    "smoke_fire": [
        "conf_threshold",
        "fire_conf_threshold",
        "smoke_conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "detection_time_threshold",
        "min_bbox_area_ratio",
        "max_bbox_area_ratio",
        "tail_light_max_area_ratio",
        "tail_light_min_red_ratio",
        "tail_light_max_orange_ratio",
        "tail_light_min_bright_ratio",
        "tail_light_max_value_std",
        "tail_light_min_aspect_ratio",
        "tail_light_max_aspect_ratio",
        "tail_light_max_motion_ratio",
        "detection_iou_merge_threshold",
        "max_raw_detections",
        "max_detections_per_frame",
    ],
    "evn_smartech": ["conf_threshold", "stranger_conf_threshold"],
    "workspace_monitor": [
        "conf_threshold",
        "ppe_conf_threshold",
        "face_similarity_threshold",
        "max_workers",
        "detection_cooldown",
    ],
    "smoking_behavior": [
        "roi",
        "max_roi_polygons",
        "conf_threshold",
    ],
    "oil_cap_detection": [
        "conf_threshold",
        "iou_threshold",
        "detection_cooldown",
        "track_match_iou_threshold",
        "track_stale_timeout",
        "violation_min_center_y_ratio",
        "violation_labels",
    ],
}


MODEL_TYPE_ADDITIONAL_PARAMS = {
    "object_detection": [],
    "helmet_detection": [
        "conf_threshold",
        "roi",
        "detection_cooldown",
        "max_roi_polygons",
    ],
    "petrolimex_detection_model": [
        "roi",
        "max_roi_polygons",
        "enable_people_aux_detection",
        "people_conf_threshold",
        "people_match_iou_threshold",
        "people_model_path",
    ],
    "alpr": [],
    "oil_dumping": ["roi", "max_roi_polygons"],
    "oil_spill": ["roi", "max_roi_polygons"],
    "oil_cap_detection": [
        "roi",
        "max_roi_polygons",
        "person_model_path",
        "person_conf_threshold",
        "person_iou_threshold",
        "open_cap_max_top_y_ratio",
        "no_person_repeat_interval",
        "skip_when_person_present",
    ],
    "tran_dau": ["roi"],
    "people_control": ["roi", "max_roi_polygons"],
    "smoke_fire": [
        "roi",
        "max_roi_polygons",
        "conf_threshold",
        "fire_conf_threshold",
        "smoke_conf_threshold",
        "min_bbox_area_ratio",
        "max_bbox_area_ratio",
        "tail_light_max_area_ratio",
        "tail_light_min_red_ratio",
        "tail_light_max_orange_ratio",
        "tail_light_min_bright_ratio",
        "tail_light_max_value_std",
        "tail_light_min_aspect_ratio",
        "tail_light_max_aspect_ratio",
        "tail_light_max_motion_ratio",
        "detection_iou_merge_threshold",
        "max_raw_detections",
        "max_detections_per_frame",
    ],
    "evn_smartech": ["roi", "max_roi_polygons"],
    "workspace_monitor": [
        "roi",
        "max_roi_polygons",
        "roi_edit_mode",
        "auto_roi_enabled",
        "auto_roi_source",
        "auto_roi_min_conf",
        "auto_roi_update_interval",
        "auto_roi_stable_frames",
        "auto_roi_lock",
        "registered_employee_ids",
        "registered_worker_count",
    ],
    "smoking_behavior": [
        "roi",
        "max_roi_polygons",
        "cgr_conf",
        "threshold",
        "skeleton",
        "cig_box",
        "annotate",
    ],
}


def validate_parameter_value(
    param_name: str, param_value: Any, model_type: Optional[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate a single parameter value

    Args:
        param_name: Parameter name
        param_value: Parameter value
        model_type: Model type (optional, used for context-aware validation)

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Special handling for ROI
    if param_name in ["roi", "roi_polygon"]:
        if not isinstance(param_value, list):
            return False, f"{param_name} must be a list"

        return validate_normalized_roi_input(param_value)

    if param_name == "roi_rect":
        if not isinstance(param_value, (list, tuple)) or len(param_value) != 4:
            return False, "roi_rect must be [x1, y1, x2, y2]"
        for val in param_value:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                return False, "roi_rect values must be numbers in [0, 1]"
        return True, None

    if param_name == "violation_labels":
        if not isinstance(param_value, list):
            return False, "violation_labels must be list, got {0}".format(type(param_value).__name__)
        if any(not isinstance(item, str) for item in param_value):
            return False, "violation_labels must contain only strings"
        return True, None

    # Get validation rule
    rule = PARAMETER_RULES.get(param_name)
    if not rule:
        # Unknown parameter - allow it but warn
        return True, None

    # Type check
    expected_type = rule["type"]
    if not isinstance(param_value, expected_type):
        if isinstance(expected_type, tuple):
            type_names = " or ".join(t.__name__ for t in expected_type)
            return (
                False,
                f"{param_name} must be {type_names}, got {type(param_value).__name__}",
            )
        else:
            return (
                False,
                f"{param_name} must be {expected_type.__name__}, got {type(param_value).__name__}",
            )

    # Range check
    if "min" in rule and param_value < rule["min"]:
        return False, f"{param_name} must be >= {rule['min']}, got {param_value}"

    if "max" in rule and param_value > rule["max"]:
        return False, f"{param_name} must be <= {rule['max']}, got {param_value}"

    return True, None


def validate_general_parameters(
    model_type: str, parameters: Dict[str, Any], strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate general parameters for a model type

    Args:
        model_type: Type of model
        parameters: Parameters to validate
        strict: If True, only allow parameters defined for this model type

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    allowed_params = MODEL_TYPE_GENERAL_PARAMS.get(model_type, [])

    for param_name, param_value in parameters.items():
        # Skip model_type itself
        if param_name == "model_type":
            continue

        # Check if parameter is allowed for this model type
        if strict and param_name not in allowed_params:
            errors.append(
                f"Parameter '{param_name}' is not allowed for model type '{model_type}'"
            )
            continue

        # Validate parameter value
        is_valid, error = validate_parameter_value(param_name, param_value)
        if not is_valid:
            errors.append(f"Parameter '{param_name}': {error}")

    return len(errors) == 0, errors


def validate_additional_parameters(
    model_type: str, parameters: Dict[str, Any], strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate additional parameters for a model type

    Args:
        model_type: Type of model
        parameters: Parameters to validate
        strict: If True, only allow parameters defined for this model type

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    allowed_params = MODEL_TYPE_ADDITIONAL_PARAMS.get(model_type, [])

    for param_name, param_value in parameters.items():
        # Check if parameter is allowed for this model type
        if strict and param_name not in allowed_params:
            errors.append(
                f"Parameter '{param_name}' is not allowed as additional parameter for model type '{model_type}'"
            )
            continue

        # Validate parameter value (pass model_type for context-aware validation)
        is_valid, error = validate_parameter_value(
            param_name, param_value, model_type=model_type
        )
        if not is_valid:
            errors.append(f"Parameter '{param_name}': {error}")

    return len(errors) == 0, errors


def get_parameter_documentation(model_type: str) -> Dict[str, Any]:
    """
    Get documentation for parameters of a specific model type

    Args:
        model_type: Type of model

    Returns:
        Dictionary with general and additional parameter lists
    """
    return {
        "model_type": model_type,
        "general_parameters": {
            "allowed": MODEL_TYPE_GENERAL_PARAMS.get(model_type, []),
            "definitions": {
                param: PARAMETER_RULES.get(param, {"type": "any"})
                for param in MODEL_TYPE_GENERAL_PARAMS.get(model_type, [])
            },
        },
        "additional_parameters": {
            "allowed": MODEL_TYPE_ADDITIONAL_PARAMS.get(model_type, []),
            "definitions": {
                **(
                    {
                        "roi": {
                            "type": "array",
                            "description": "Region of Interest as normalized polygon [[x1,y1], [x2,y2], ...]",
                            "example": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
                        }
                    }
                    if "roi" in MODEL_TYPE_ADDITIONAL_PARAMS.get(model_type, [])
                    else {}
                ),
                **(
                    {
                        "max_roi_polygons": {
                            "type": "integer",
                            "min": 1,
                            "max": 5,
                            "default": 1,
                            "description": "Maximum number of ROI polygons allowed for this camera",
                        }
                    }
                    if "max_roi_polygons" in MODEL_TYPE_ADDITIONAL_PARAMS.get(model_type, [])
                    else {}
                ),
            },
        },
    }
