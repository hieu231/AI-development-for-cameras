"""
Parameter Defaults - Define default parameter structures for each model type
"""

from typing import Dict, Any


def get_default_additional_parameters(model_type: str) -> Dict[str, Any]:
    """
    Get default additional parameters for a model type

    Additional parameters are camera-specific and stored in camera_models.additional_parameters

    Args:
        model_type: Type of model (e.g., 'helmet_detection', 'people_control')

    Returns:
        Dictionary of default additional parameters
    """
    # Default ROI: full screen (normalized coordinates)
    # Top-left (0,0), Top-right (1,0), Bottom-right (1,1), Bottom-left (0,1)
    DEFAULT_ROI_FULL_SCREEN = [[0, 0], [1, 0], [1, 1], [0, 1]]

    defaults = {
        # Models that use ROI (Region of Interest)
        "helmet_detection": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "petrolimex_detection_model": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "enable_people_aux_detection": True,
            "people_conf_threshold": 0.5,
            "people_match_iou_threshold": 0.05,
        },
        "people_control": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "oil_dumping": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "oil_spill": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "oil_cap_detection": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "person_model_path": "src/ai_models/model_weights/yolo11n.pt",
            "person_conf_threshold": 0.35,
            "person_iou_threshold": 0.35,
            "open_cap_max_top_y_ratio": 0.6,
            "no_person_repeat_interval": 5.0,
            "skip_when_person_present": False,
        },
        "tran_dau": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "smoke_fire": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "fire_conf_threshold": 0.30,
            "smoke_conf_threshold": 0.25,
            "min_bbox_area_ratio": 0.0008,
            "tail_light_max_area_ratio": 0.012,
            "detection_iou_merge_threshold": 0.75,
            "max_raw_detections": 80,
            "max_detections_per_frame": 25,
        },
        "evn_smartech": {"roi": DEFAULT_ROI_FULL_SCREEN, "max_roi_polygons": 1},
        "workspace_monitor": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "roi_edit_mode": False,
            "auto_roi_enabled": False,
            "auto_roi_source": "reflective_wire",
            "auto_roi_min_conf": 0.35,
            "auto_roi_update_interval": 15,
            "auto_roi_stable_frames": 3,
            "auto_roi_lock": False,
            "registered_employee_ids": [],
            "registered_worker_count": 0,
        },
        "smoking_behavior": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "cgr_conf": 0.4,
            "threshold": 50,
            "skeleton": False,
            "cig_box": False,
            "annotate": True,
        },
        "face_recognition": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
            "annotate": True,
            # Cosine-similarity floor for declaring a face "known". Lower = more
            # permissive matches but more false positives. The model also surfaces
            # the closest profile in the overlay even when below this floor so
            # operators can tune from real data.
            "face_similarity_threshold": 0.45,
            # Cooldowns are in seconds — same face only emits one DB event per
            # window. Tightened to 5s for unknowns so the dashboard sees a
            # fresh entry every few seconds (cosine-sim dedup at 0.55 keeps
            # the SAME person from spamming inside that 5s window).
            "known_cooldown": 30,
            "unknown_cooldown": 5,
        },
        # ALPR + Gemini plate operate on the full frame today, but we still
        # seed roi/max_roi_polygons so the UI shows consistent ROI controls
        # across model types. The models themselves don't crop today — they
        # can be updated to mask non-ROI regions later without needing a DB
        # migration.
        "alpr": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
        },
        "gemini_plate": {
            "roi": DEFAULT_ROI_FULL_SCREEN,
            "max_roi_polygons": 1,
        },
    }

    return defaults.get(model_type, {})


def get_parameter_schema(model_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get complete parameter schema for a model type

    Returns both general and additional parameter definitions

    Args:
        model_type: Type of model

    Returns:
        Dictionary with 'general' and 'additional' parameter schemas
    """
    schemas = {
        "alpr": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_cooldown": {"type": "int", "min": 0, "default": 10},
                "min_detections": {"type": "int", "min": 1, "default": 1},
                "max_det": {"type": "int", "min": 1, "default": 10},
            },
            "additional": {},
        },
        "gemini_plate": {
            "general": {
                "gemini_api_key": {"type": "string", "required": True},
                "model_path": {"type": "string", "required": True},
                "prompt": {"type": "string", "default": "Extract license plate number"},
            },
            "additional": {},
        },
        "helmet_detection": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_cooldown": {"type": "int", "min": 0, "default": 10},
                "min_detections": {"type": "int", "min": 1, "default": 1},
                "max_det": {"type": "int", "min": 1, "default": 50},
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "petrolimex_detection_model": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "person_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                    "optional": True,
                },
                "no_hardhat_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                    "optional": True,
                },
                "enable_people_aux_detection": {
                    "type": "boolean",
                    "default": True,
                    "optional": True,
                },
                "people_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                    "optional": True,
                },
                "people_match_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.05,
                    "optional": True,
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 300,
                    "unit": "seconds",
                },
                "people_model_path": {
                    "type": "string",
                    "default": "src/ai_models/model_weights/yolo11m.pt",
                    "optional": True,
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "oil_dumping": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.4,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_time_threshold": {
                    "type": "int",
                    "min": 0,
                    "default": 15,
                    "unit": "seconds",
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 28800,
                    "unit": "seconds",
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "people_control": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.4,
                },
                "time_threshold": {"type": "float", "min": 0.0, "default": 0.0},
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 300,
                    "unit": "seconds",
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use (dynamic limit)",
                },
            },
        },
        "oil_spill": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 1800,
                    "unit": "seconds",
                },
                "foreground_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "foreground_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "foreground_cover_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.6,
                },
                "person_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "person_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "person_cover_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.6,
                },
                "min_bbox_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.003,
                },
                "max_bbox_height_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.22,
                },
                "min_aspect_ratio": {"type": "float", "min": 0.0, "default": 0.75},
                "max_motion_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.12,
                },
                "max_center_shift_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.08,
                },
                "max_area_change_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.6,
                },
                "max_area_shrink_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.15,
                },
                "min_area_growth_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.08,
                },
                "min_consecutive_frames": {"type": "int", "min": 1, "default": 3},
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "oil_cap_detection": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 300,
                    "unit": "seconds",
                },
                "track_match_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.5,
                },
                "track_stale_timeout": {
                    "type": "float",
                    "min": 0.0,
                    "default": 3.0,
                    "unit": "seconds",
                },
                "violation_min_center_y_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "violation_labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["oil_cap_opened"],
                    "optional": True,
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "tran_dau": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.25,
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.3,
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 60,
                    "unit": "seconds",
                },
                "foreground_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "foreground_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "foreground_cover_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.55,
                },
                "person_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "person_iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "person_cover_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.55,
                },
                "min_bbox_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.002,
                },
                "max_bbox_height_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.28,
                },
                "min_aspect_ratio": {
                    "type": "float",
                    "min": 0.0,
                    "default": 0.55,
                },
                "min_consecutive_frames": {
                    "type": "int",
                    "min": 1,
                    "default": 1,
                },
                "min_flow_frames": {
                    "type": "int",
                    "min": 1,
                    "default": 1,
                },
                "flow_motion_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.04,
                },
                "flow_edge_motion_ratio_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.06,
                },
                "sudden_area_growth_ratio": {
                    "type": "float",
                    "min": 0.0,
                    "default": 0.18,
                },
                "high_severity_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.008,
                },
                "critical_severity_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.02,
                },
                "global_event_cooldown": {
                    "type": "float",
                    "min": 0.0,
                    "default": 3.0,
                    "unit": "seconds",
                },
                "continuous_event_interval": {
                    "type": "float",
                    "min": 0.0,
                    "default": 0.5,
                    "unit": "seconds",
                },
                "track_ongoing_incidents": {
                    "type": "boolean",
                    "default": True,
                },
                "debug_detection_summary": {
                    "type": "boolean",
                    "default": True,
                },
                "debug_summary_interval_seconds": {
                    "type": "float",
                    "min": 0.5,
                    "default": 5.0,
                    "unit": "seconds",
                },
                "incident_hold_seconds": {
                    "type": "float",
                    "min": 0.0,
                    "default": 10.0,
                    "unit": "seconds",
                },
                "max_incident_motion_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.85,
                },
                "max_incident_edge_motion_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.95,
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
            },
        },
        "smoke_fire": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.30,
                },
                "fire_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.30,
                    "description": "Ngưỡng confidence riêng cho class fire",
                },
                "smoke_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.25,
                    "description": "Ngưỡng confidence riêng cho class smoke",
                },
                "iou_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 15,
                    "unit": "seconds",
                },
                "detection_time_threshold": {
                    "type": "float",
                    "min": 0,
                    "default": 0,
                    "unit": "seconds",
                    "description": "Thời gian phát hiện liên tục trước khi trigger event (0=tắt)",
                },
                "min_bbox_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.0008,
                    "description": "Tỷ lệ diện tích bbox tối thiểu để bắt khói/lửa mới xuất hiện",
                },
                "max_bbox_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.60,
                },
                "tail_light_max_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.012,
                    "description": "Bbox fire nhỏ hơn ngưỡng này sẽ được kiểm tra anti-tail-light",
                },
                "tail_light_min_red_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.55,
                },
                "tail_light_max_orange_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.18,
                },
                "tail_light_min_bright_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
                "tail_light_max_value_std": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.10,
                },
                "tail_light_min_aspect_ratio": {
                    "type": "float",
                    "range": [0.0, 10.0],
                    "default": 0.35,
                },
                "tail_light_max_aspect_ratio": {
                    "type": "float",
                    "range": [0.0, 10.0],
                    "default": 3.50,
                },
                "tail_light_max_motion_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.06,
                },
                "detection_iou_merge_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.75,
                    "description": "Gộp bbox cùng class khi IoU >= ngưỡng này",
                },
                "max_raw_detections": {
                    "type": "int",
                    "min": 1,
                    "default": 80,
                    "description": "Giới hạn số bbox YOLO trả về mỗi frame",
                },
                "max_detections_per_frame": {
                    "type": "int",
                    "min": 1,
                    "default": 25,
                    "description": "Giới hạn số bbox được giữ lại sau filter",
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.30,
                },
                "fire_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.30,
                },
                "smoke_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.25,
                },
                "min_bbox_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.0008,
                },
                "tail_light_max_area_ratio": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.012,
                },
                "detection_iou_merge_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.75,
                },
                "max_raw_detections": {
                    "type": "int",
                    "min": 1,
                    "default": 80,
                },
                "max_detections_per_frame": {
                    "type": "int",
                    "min": 1,
                    "default": 25,
                },
            },
        },
        "evn_smartech": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.55,
                },
                "stranger_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                },
            },
            "additional": {
                "roi": {"type": "polygon", "normalized": True, "optional": True}
            },
        },
        "workspace_monitor": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.4,
                },
                "ppe_conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "face_similarity_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.45,
                },
                "max_workers": {"type": "int", "min": 1, "default": 5},
                "detection_cooldown": {
                    "type": "int",
                    "min": 0,
                    "default": 300,
                    "unit": "seconds",
                },
            },
            "additional": {
                "roi": {
                    "type": "polygon_set",
                    "normalized": True,
                    "optional": True,
                    "description": "Single polygon [[x,y],...] or polygon set [[[x,y],...], ...]",
                },
                "max_roi_polygons": {
                    "type": "int",
                    "min": 1,
                    "max": 5,
                    "default": 1,
                    "description": "Maximum number of ROI polygons to use",
                },
                "roi_edit_mode": {
                    "type": "boolean",
                    "default": False,
                    "optional": True,
                },
                "auto_roi_enabled": {
                    "type": "boolean",
                    "default": False,
                    "optional": True,
                },
                "auto_roi_source": {
                    "type": "string",
                    "default": "reflective_wire",
                    "optional": True,
                },
                "auto_roi_min_conf": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.35,
                    "optional": True,
                },
                "auto_roi_update_interval": {
                    "type": "int",
                    "min": 1,
                    "default": 15,
                    "optional": True,
                },
                "auto_roi_stable_frames": {
                    "type": "int",
                    "min": 1,
                    "default": 3,
                    "optional": True,
                },
                "auto_roi_lock": {
                    "type": "boolean",
                    "default": False,
                    "optional": True,
                },
                "registered_employee_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "optional": True,
                },
                "registered_worker_count": {
                    "type": "int",
                    "min": 0,
                    "default": 0,
                    "optional": True,
                },
            },
        },
        "smoking_behavior": {
            "general": {
                "conf_threshold": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.4,
                },
            },
            "additional": {
                "cgr_conf": {
                    "type": "float",
                    "range": [0.0, 1.0],
                    "default": 0.4,
                    "optional": True,
                },
                "threshold": {
                    "type": "int",
                    "min": 0,
                    "default": 50,
                    "optional": True,
                },
                "skeleton": {
                    "type": "boolean",
                    "default": False,
                    "optional": True,
                },
                "cig_box": {
                    "type": "boolean",
                    "default": False,
                    "optional": True,
                },
                "annotate": {
                    "type": "boolean",
                    "default": True,
                    "optional": True,
                },
            },
        },
    }

    return schemas.get(model_type, {"general": {}, "additional": {}})
