"""
base_model.py
Base Model Interface chuẩn cho toàn bộ hệ thống AI detection
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
import numpy as np
import logging
import os
from src.utils.alert_levels import AlertLevel, ALERT_LEVEL_MAP, AlertColor

logger = logging.getLogger(__name__)


def resolve_engine_path(
    model_path: str,
    runtime_device: Optional[str] = None,
) -> str:
    """Prefer TensorRT .engine only when runtime device is CUDA."""
    if model_path and model_path.endswith(".pt"):
        if os.getenv("AIBE_DISABLE_TENSORRT", "").lower() == "true":
            logger.info("AIBE_DISABLE_TENSORRT=true: keeping .pt model")
            return model_path

        if os.getenv("AIBE_SKIP_CUDA", "").lower() == "true":
            logger.info("AIBE_SKIP_CUDA=true: skipping TensorRT engine selection")
            return model_path

        effective_device = (runtime_device or "").strip().lower()
        if effective_device and effective_device != "cuda":
            logger.info(
                "Runtime device '%s' is not CUDA, keeping .pt model: %s",
                effective_device,
                model_path,
            )
            return model_path

        if not effective_device:
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.info("CUDA is not available, keeping .pt model: %s", model_path)
                    return model_path
            except Exception:
                return model_path

        engine_path = model_path[:-3] + ".engine"
        if os.path.isfile(engine_path):
            logger.info(
                "TensorRT engine found, using: %s (instead of %s)",
                engine_path,
                model_path,
            )
            return engine_path
    return model_path


@dataclass
class DetectedObject:
    """Thông tin một object được detect"""
    label: str
    confidence: float
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2) hoặc None nếu không có bbox
    extra: Optional[Dict[str, Any]] = None


@dataclass
class AlertInfo:
    """Thông tin cảnh báo (dùng để hiển thị + lưu log)"""
    level: AlertLevel
    color: AlertColor
    message: str
    confidence: float
    detected_objects: List[DetectedObject] = None

    def __post_init__(self):
        if self.detected_objects is None:
            self.detected_objects = []


@dataclass
class DetectionResult:
    """
    Kết quả chuẩn mà mọi model phải trả về
    """
    frame: Optional[np.ndarray]         # Frame đã được vẽ annotation (nếu có)
    event: bool                         # True → có sự kiện cần lưu DB / gửi alert
    metadata: Dict[str, Any]            # Metadata tùy model (có thể chứa AlertInfo)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "metadata": self.metadata,
            # frame không chuyển thành dict (quá nặng), chỉ dùng khi cần hiển thị
        }


class BaseModel(ABC):
    """
    Base class cho mọi AI model trong hệ thống
    """

    _YOLO_TRACK_OPTIONAL_MODULES = {"scipy", "lap"}

    def __init__(
        self,
        model_name: str,
        default_alert_level: AlertLevel = AlertLevel.LOW,
        confidence_threshold: float = 0.7,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            model_name: Tên model (dùng để log, hiển thị)
            default_alert_level: Mức cảnh báo mặc định khi detect thành công
            confidence_threshold: Ngưỡng confidence để coi là detect
            model_path: Đường dẫn tới weight (nếu có)
            **kwargs: Các tham số bổ sung
        """
        self.model_name = model_name
        self.default_alert_level = default_alert_level
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path

        # Device auto-detect
        self.device = self._setup_device()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._yolo_track_available = True

        self.logger.info(
            f"Initialized {self.model_name} | Device: {self.device} | "
            f"Conf thresh: {self.confidence_threshold} | Default level: {self.default_alert_level.name}"
        )

    def _setup_device(self) -> str:
        """Tự động chọn device tốt nhất"""
        # Skip CUDA check entirely if explicitly disabled (useful for Jetson CUDA issues)
        if os.getenv("AIBE_SKIP_CUDA", "").lower() == "true":
            logger.info("AIBE_SKIP_CUDA=true: Skipping CUDA, using CPU")
            return "cpu"
        
        preferred = (os.getenv("AIBE_DEVICE") or "").strip().lower()
        if preferred:
            # Allow forcing device via env (useful for Jetson/PyTorch CUDA mismatch)
            if preferred in {"cpu", "cuda", "mps"}:
                if preferred == "cuda" and not self._cuda_is_usable():
                    logger.warning("AIBE_DEVICE=cuda requested but CUDA is not usable; falling back to CPU")
                    return "cpu"
                return preferred
            logger.warning(f"Unknown AIBE_DEVICE='{preferred}', ignoring (expected cpu|cuda|mps)")

        try:
            import torch
            if torch.cuda.is_available():
                # Some Jetson images end up with an incompatible CUDA-enabled torch wheel
                # (CUDA is visible, but basic kernels fail at runtime). Detect and fallback.
                if self._cuda_is_usable():
                    return "cuda"
                logger.warning("CUDA is available but not usable (kernel execution failed). Falling back to CPU.")
                return "cpu"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _cuda_is_usable(self) -> bool:
        """
        Quick runtime check that CUDA kernels can actually execute.

        We've seen Jetson setups where `torch.cuda.is_available()` is True but
        CUDA conv2d fails with: 'GET was unable to find an engine to execute this computation'
        or 'no kernel image is available for execution on the device'.
        """
        import sys
        import io
        import os
        from contextlib import redirect_stderr, redirect_stdout
        
        # Temporarily suppress CUDA error output
        old_stderr = sys.stderr
        old_stdout = sys.stdout
        
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            
            # Suppress both stderr and stdout during CUDA check
            # CUDA errors can be printed to either stream
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                sys.stdout = devnull
                try:
                    # Minimal conv2d on GPU
                    x = torch.randn(1, 3, 32, 32, device="cuda")
                    conv = torch.nn.Conv2d(3, 8, 3).to("cuda")
                    _ = conv(x)
                    torch.cuda.synchronize()  # Ensure operation completes
                    result = True
                except Exception:
                    result = False
                finally:
                    sys.stderr = old_stderr
                    sys.stdout = old_stdout
            
            if not result:
                # Only log a clean warning after suppressing the noisy error
                logger.warning("CUDA kernel execution failed (likely compute capability mismatch). Falling back to CPU.")
            return result
        except Exception as e:
            sys.stderr = old_stderr
            sys.stdout = old_stdout
            # Extract just the error type/message, not the full stack trace
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            logger.warning(f"CUDA usability check failed: {error_msg}. Falling back to CPU.")
            return False

    def _get_yolo_runtime_kwargs(self, **overrides) -> Dict[str, Any]:
        """Build shared Ultralytics runtime kwargs for fast-path inference."""
        runtime: Dict[str, Any] = {}

        device = self.device if self.device != "mps" else "cpu"
        if device:
            runtime["device"] = device

        raw_imgsz = (os.getenv("AIBE_YOLO_IMGSZ", "640") or "").strip()
        if raw_imgsz:
            try:
                imgsz = int(raw_imgsz)
                if imgsz > 0:
                    runtime["imgsz"] = imgsz
            except ValueError:
                self.logger.warning("Ignoring invalid AIBE_YOLO_IMGSZ='%s'", raw_imgsz)

        raw_stride = (os.getenv("AIBE_YOLO_VID_STRIDE") or "").strip()
        if raw_stride:
            try:
                stride = int(raw_stride)
                if stride > 1:
                    runtime["vid_stride"] = stride
            except ValueError:
                self.logger.warning("Ignoring invalid AIBE_YOLO_VID_STRIDE='%s'", raw_stride)

        tracker = (os.getenv("AIBE_YOLO_TRACKER", "bytetrack.yaml") or "").strip()
        if tracker:
            runtime["tracker"] = tracker

        use_half = os.getenv("AIBE_YOLO_HALF", "true").lower() == "true"
        if use_half and device == "cuda":
            runtime["half"] = True

        runtime.update(overrides)
        return runtime

    def _get_yolo_predict_kwargs(self, **overrides) -> Dict[str, Any]:
        """Build YOLO kwargs that are valid for predict() fallback."""
        runtime = self._get_yolo_runtime_kwargs(**overrides)
        runtime.pop("persist", None)
        runtime.pop("tracker", None)
        return runtime

    def _get_missing_yolo_track_dependency(self, exc: BaseException) -> Optional[str]:
        """Return the missing tracker dependency name if this is a recoverable import error."""
        module_name = getattr(exc, "name", None)
        if module_name in self._YOLO_TRACK_OPTIONAL_MODULES:
            return module_name

        error_text = str(exc).lower()
        for candidate in self._YOLO_TRACK_OPTIONAL_MODULES:
            if f"no module named '{candidate}'" in error_text:
                return candidate
        return None

    def _run_yolo_track(self, model, source, **kwargs):
        """Run Ultralytics track() with shared runtime acceleration knobs."""
        if not self._yolo_track_available:
            return model.predict(
                source=source,
                **self._get_yolo_predict_kwargs(**kwargs),
            )

        runtime_kwargs = self._get_yolo_runtime_kwargs(**kwargs)
        try:
            return model.track(source=source, **runtime_kwargs)
        except ModuleNotFoundError as exc:
            missing_dependency = self._get_missing_yolo_track_dependency(exc)
            if missing_dependency is None:
                raise

            self._yolo_track_available = False
            self.logger.warning(
                "YOLO tracker dependency '%s' is missing. Falling back to predict() without persistent tracking.",
                missing_dependency,
            )
            return model.predict(
                source=source,
                **self._get_yolo_predict_kwargs(**kwargs),
            )
        except np.linalg.LinAlgError:
            # BoT-SORT Kalman filter can hit a non-positive-definite
            # covariance matrix when the tracked distribution degenerates
            # (typically after several consecutive identical-bbox frames on
            # a static false positive). The exception corrupts the Ultralytics
            # predictor's internal tracker list — the next call on the same
            # YOLO instance re-raises even when we ask for a plain predict()
            # instead of track(). So we:
            #   1. Reset the predictor's tracker state so subsequent
            #      track() calls start fresh,
            #   2. Try a plain predict() as a one-shot fallback for THIS
            #      frame, with its own guard,
            #   3. If the fallback also fails, drop the frame (return an
            #      empty results list) rather than crash the per-camera
            #      processor loop.
            self.logger.warning(
                "Tracker Kalman filter numerical error — resetting tracker "
                "state and falling back to predict() for this frame."
            )
            try:
                predictor = getattr(model, "predictor", None)
                if predictor is not None:
                    trackers = getattr(predictor, "trackers", None)
                    if trackers:
                        for tracker in trackers:
                            reset = getattr(tracker, "reset", None)
                            if callable(reset):
                                try:
                                    reset()
                                except Exception:
                                    pass
                    setattr(predictor, "trackers", None)
            except Exception:
                pass

            try:
                return model.predict(
                    source=source,
                    **self._get_yolo_predict_kwargs(**kwargs),
                )
            except Exception as predict_exc:
                self.logger.warning(
                    "predict() fallback after tracker error also failed "
                    "(%s) — dropping this frame.",
                    predict_exc,
                )
                return []

    @abstractmethod
    def process_frame(self, frame: np.ndarray, **kwargs) -> DetectionResult:
        """
        Xử lý một frame duy nhất

        Args:
            frame: Frame gốc (BGR, numpy array)
            **kwargs: Tham số tùy model (ROI, custom thresholds, ...)

        Returns:
            DetectionResult chuẩn hóa
        """
        pass

    # ──────────────────────────────────────────────────────────────
    # Helper methods (có thể override nếu cần)
    # ──────────────────────────────────────────────────────────────
    def _create_alert_info(
        self,
        message: str,
        confidence: float,
        detected_objects: Optional[List[DetectedObject]] = None,
        level: Optional[AlertLevel] = None,
    ) -> AlertInfo:
        """Tạo AlertInfo chuẩn"""
        alert_level = level or self.default_alert_level
        return AlertInfo(
            level=alert_level,
            color=ALERT_LEVEL_MAP.get(alert_level, AlertColor.GRAY),
            message=message,
            confidence=confidence,
            detected_objects=detected_objects or [],
        )

    def _should_trigger_event(self, confidence: float) -> bool:
        """Kiểm tra có vượt ngưỡng confidence không"""
        return confidence >= self.confidence_threshold

    def _build_metadata(
        self,
        alert_info: Optional[AlertInfo] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Tạo metadata chuẩn để nhét vào DetectionResult"""
        metadata: Dict[str, Any] = {
            "model": self.model_name,
            "timestamp": __import__("time").time(),
            "severity": AlertLevel.LOW.value,
        }
        if alert_info:
            # Chuyển AlertInfo sang dict và đảm bảo enum → string để JSON/JSONB không bị lỗi
            alert_dict = asdict(alert_info)
            level = alert_dict.get("level")
            color = alert_dict.get("color")

            # AlertLevel → dùng name (LOW/HIGH) cho websocket & client
            if isinstance(level, AlertLevel):
                alert_dict["level"] = level.name
                metadata["severity"] = level.value

            # AlertColor → dùng value ("#FF0000"...) để frontend dùng trực tiếp
            if isinstance(color, AlertColor):
                alert_dict["color"] = color.value

            metadata["alert"] = alert_dict
        if extra:
            metadata.update(extra)
        return metadata

    # ──────────────────────────────────────────────────────────────
    # Resource management
    # ──────────────────────────────────────────────────────────────
    def cleanup(self):
        """Dọn dẹp GPU memory"""
        active_logger = getattr(self, "logger", logger)
        try:
            import gc
            import torch

            gc.collect()
            device = getattr(self, "device", None)
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception as e:
            active_logger.warning(f"Cleanup error: {e}")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
