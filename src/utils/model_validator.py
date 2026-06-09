"""
Model validation utilities for AI model versioning
Validates model path and ensures model can be loaded
"""
import os
import logging
import shutil
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_WEIGHTS_DIR = Path("model_weights")


class ModelValidator:
    """Validator for AI model files and paths"""

    @staticmethod
    def validate_model_path(model_path: str, model_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if model path exists and can be loaded

        Args:
            model_path: Path to model file
            model_type: Type of model (yolo, ncl, etc.)

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check if file exists
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"

        # Check file extension
        if not model_path.endswith(('.pt', '.pth', '.onnx', '.engine', '.trt')):
            return False, f"Invalid model file extension. Expected .pt, .pth, .onnx, .engine, or .trt"

        # Try to load model based on type
        try:
            if model_type in ['yolo', 'yolov8', 'yolov11', 'object_detection',
                             'helmet_detection', 'petrolimex_detection_model', 'alpr', 'oil_dumping', 'oil_spill',
                             'tran_dau', 'oil_cap_detection']:
                # Validate YOLO model
                return ModelValidator._validate_yolo_model(model_path)
            elif model_type == 'smoking_behavior':
                # Smoking behavior uses OpenVINO ONNX pair (last.onnx + yolov8n-pose.onnx)
                return ModelValidator._validate_smoking_behavior_model(model_path)
            elif model_type == 'ncl' or model_type == 'people_control':
                # Validate NCL model (if you have specific validation)
                return ModelValidator._validate_ncl_model(model_path)
            else:
                # Unknown type, just check file exists
                logger.warning(f"Unknown model type '{model_type}', skipping load validation")
                return True, None

        except Exception as e:
            return False, f"Failed to validate model: {str(e)}"

    @staticmethod
    def _validate_yolo_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate YOLO model by trying to load it"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            # If we can load it, it's valid
            del model  # Clean up
            return True, None
        except Exception as e:
            return False, f"Failed to load YOLO model: {str(e)}"

    @staticmethod
    def _validate_ncl_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate NCL/PyTorch model by trying to load it"""
        try:
            import torch
            # Try to load as PyTorch model
            model = torch.load(model_path, map_location='cpu')
            del model  # Clean up
            return True, None
        except Exception as e:
            return False, f"Failed to load PyTorch model: {str(e)}"

    @staticmethod
    def _validate_smoking_behavior_model(model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate smoking behavior model path and companion pose model."""
        if not model_path.endswith('.onnx'):
            return False, "smoking_behavior model_path must be an .onnx file"

        model_dir = Path(model_path).parent
        pose_model = model_dir / 'yolov8n-pose.onnx'
        if not pose_model.exists():
            return False, f"Missing companion pose model: {pose_model}"

        return True, None

    @staticmethod
    def copy_model_to_weights_dir(source_path: str, model_name: str, version: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Copy model file to model_weights directory with version naming
        Handles duplicate names by appending timestamp

        Args:
            source_path: Original model file path
            model_name: Name of the model
            version: Version string

        Returns:
            Tuple[bool, Optional[str], Optional[str]]: (success, new_path, error_message)
        """
        try:
            # Create model_weights directory if not exists
            MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

            # Get file extension
            ext = Path(source_path).suffix

            # Sanitize model name and version for filename
            safe_model_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in model_name)
            safe_version = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in version)

            # Create target filename
            target_filename = f"{safe_model_name}_v{safe_version}{ext}"
            target_path = MODEL_WEIGHTS_DIR / target_filename

            # If file already exists, append timestamp
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_filename = f"{safe_model_name}_v{safe_version}_{timestamp}{ext}"
                target_path = MODEL_WEIGHTS_DIR / target_filename
                logger.info(f"File exists, using timestamp: {target_filename}")

            # Copy file
            shutil.copy2(source_path, target_path)
            logger.info(f"Copied model from {source_path} to {target_path}")

            return True, str(target_path), None

        except Exception as e:
            return False, None, f"Failed to copy model file: {str(e)}"

    @staticmethod
    def get_model_file_info(model_path: str) -> dict:
        """Get information about a model file"""
        try:
            if not os.path.exists(model_path):
                return {"exists": False}

            stat = os.stat(model_path)
            return {
                "exists": True,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": Path(model_path).suffix
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}
