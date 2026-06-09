"""
src/core/model_factory.py
Model Factory - Dynamic model loading with standardized interface
Hoàn toàn tương thích với hệ thống PostgreSQL + Realtime WebSocket mới
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.ai_models.base_model import BaseModel, DetectionResult

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating AI model instances"""

    RESERVED_CONSTRUCTOR_KEYS = {
        "model_name",
        "model_path",
        "default_alert_level",
    }

    # Map model types → class path
    MODEL_TYPE_MAP = {
        'object_detection': 'src.ai_models.yolov8m_model.YOLOv8Model',
        'yolo': 'src.ai_models.yolov8m_model.YOLOv8Model',
        'yolov8': 'src.ai_models.yolov8m_model.YOLOv8Model',
        'yolov11': 'src.ai_models.yolov8m_model.YOLOv8Model',
        'helmet_detection': 'src.ai_models.helmet_model.HelmetModel',
        'ppe_helmet': 'src.ai_models.helmet_model.PPEModel',
        'oil_cap_detection': 'src.ai_models.oil_cap_detection_model.OilCapDetectionModel',
        'petrolimex_detection_model': 'src.ai_models.petrolimex_detection_model.PetrolimexDetectionModel',
        # ALPR uses fast-alpr (ONNX) inside ALPRModel — it does NOT take
        # a YOLO weight file. Routing to YOLOv8Model with model_path=None
        # crashed every load with FileNotFoundError("'None' does not
        # exist") so no events were ever produced even though the camera
        # appeared to be running. The class lives at src/ai_models/alpr_model.py.
        'alpr': 'src.ai_models.alpr_model.ALPRModel',
        # gemini_plate uses the Gemini API for plate text extraction. Was
        # previously absent from this map → factory fell back to YOLOv8 →
        # only vehicle bboxes were drawn, no plate text was ever
        # extracted, no events were emitted. Restored here after commit
        # 44c2a65 regressed the map.
        'gemini_plate': 'src.ai_models.gemini_plate_model.GeminiPlateModel',
        'oil_dumping': 'src.ai_models.oil_dumping_model.OilDumpingModel',
        'oil_spill': 'src.ai_models.oil_spill_model.OilSpillModel',
        'tran_dau': 'src.ai_models.tran_dau_model.TranDauModel',
        'people_control': 'src.ai_models.people_control_model.PeopleControlModel',
        'smoke_fire': 'src.ai_models.smoke_fire_model.SmokeFireModel',
        'vehicle_gate': 'src.ai_models.vehicle_gate_model.VehicleGateModel',
        'evn_smartech': 'src.ai_models.evn_and_smartech_detection_model.EvnAndSmartechDetectionModel',
        'workspace_monitor': 'src.ai_models.workspace_monitor_model.WorkspaceMonitorModel',
        'smoking_behavior': 'src.ai_models.smoking_behavior_model.SmokingBehaviorModel',
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    @classmethod
    def _resolve_model_path(cls, model_path: Optional[str]) -> Optional[str]:
        """Resolve relative model paths against project root for stable runtime behavior."""
        if not model_path:
            return model_path

        path_obj = Path(model_path)
        if path_obj.is_absolute() and path_obj.exists():
            return str(path_obj)

        candidate = (cls.PROJECT_ROOT / path_obj).resolve()
        if candidate.exists():
            return str(candidate)

        return model_path

    @staticmethod
    def _is_face_recognition_enabled() -> bool:
        return os.getenv('ENABLE_FACE_RECOGNITION', 'true').lower() == 'true'

    @classmethod
    def _get_model_type_map(cls) -> Dict[str, str]:
        model_map = cls.MODEL_TYPE_MAP.copy()
        if cls._is_face_recognition_enabled():
            model_map['face_recognition'] = 'src.ai_models.face_recognition_model.FaceRecognitionModel'
        return model_map

    @staticmethod
    def _import_class(class_path: str):
        """Dynamically import class from string path"""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except Exception as e:
            logger.error(f"Failed to import class {class_path}: {e}")
            raise

    @classmethod
    def create_model(cls, model_info: Dict[str, Any]) -> Optional[BaseModel]:
        """
        Tạo instance model từ thông tin cấu hình
        Dùng trong thread_manager để load model cho camera
        """
        try:
            model_type = model_info.get('model_type', 'object_detection')
            raw_model_path = model_info.get('model_path')
            model_path = cls._resolve_model_path(raw_model_path)
            parameters = model_info.get('parameters', {})
            additional_params = model_info.get('additional_params', {})
            requires_external_model_path = model_type != 'smoking_behavior'
            if model_path and requires_external_model_path and not os.path.exists(model_path):
                logger.error(
                    "Model file not found for type '%s': %s (resolved from: %s)",
                    model_type,
                    model_path,
                    raw_model_path,
                )
                return None

            # Gộp tham số: camera-specific override model defaults
            conflicting = set(parameters.keys()) & set(additional_params.keys())
            if conflicting:
                logger.warning(f"Parameter conflict in model {model_info.get('name')}: {conflicting}")
                logger.warning("Camera parameters will override model defaults")

            kwargs = {**parameters, **additional_params}
            ignored_reserved = sorted(
                key for key in cls.RESERVED_CONSTRUCTOR_KEYS if key in kwargs
            )
            for key in ignored_reserved:
                kwargs.pop(key, None)
            if ignored_reserved:
                logger.warning(
                    "Ignoring reserved constructor parameters for model %s: %s",
                    model_info.get("name") or model_info.get("model_name") or model_type,
                    ignored_reserved,
                )

            # Lấy class từ map
            class_path = cls._get_model_type_map().get(model_type)
            if not class_path:
                logger.warning(f"Unknown model_type '{model_type}', fallback to YOLOv8")
                class_path = cls.MODEL_TYPE_MAP['yolo']

            model_class = cls._import_class(class_path)
            try:
                instance = model_class(model_path=model_path, **kwargs)
            except TypeError as exc:
                # Some models (e.g. SmokingBehaviorModel) set model_path internally and
                # forward **kwargs to BaseModel, so passing model_path here duplicates it.
                if "multiple values for keyword argument 'model_path'" not in str(exc):
                    raise
                logger.warning(
                    "Model %s rejected explicit model_path, retrying without it.",
                    model_class.__name__,
                )
                instance = model_class(**kwargs)

            logger.info(f"Model loaded: {model_class.__name__} | Type: {model_type} | Path: {model_path}")
            return instance

        except Exception as e:
            logger.error(f"Failed to create model: {e}", exc_info=True)
            return None

    @classmethod
    def get_available_types(cls) -> list:
        """Trả về danh sách model_type đang hỗ trợ"""
        return list(cls._get_model_type_map().keys())

    @staticmethod
    def detect(frame, model_name: str = "all") -> Optional[DetectionResult]:
        """
        Compatibility wrapper for older code that expected a global model_factory.detect(...)
        This should be wired to the appropriate detection pipeline.
        """
        # Trong hệ thống mới, detection được xử lý trong camera thread
        # Endpoint /detect chỉ để test → trả về None hoặc raise
        logger.warning("ModelFactory.detect() is deprecated. Use camera threads for real detection.")
        return None
