from queue import Queue
from threading import Event
from uuid import uuid4

import cv2
import numpy as np

from src.ai_models.base_model import DetectionResult
from src.core.thread_manager import CameraThread, ProcessorThread, ThreadManager


def test_camera_thread_stops_retrying_on_vss_rate_limit(monkeypatch):
    camera_thread = CameraThread(uuid4(), "http://example.com/stream", Queue(), Event())
    calls = {"count": 0}

    def fake_create_capture():
        calls["count"] += 1
        raise RuntimeError(
            "VSS server đang giới hạn đăng nhập (Login too frequently). Vui lòng đợi 120s rồi thử lại."
        )

    monkeypatch.setattr(camera_thread, "_create_capture", fake_create_capture)
    monkeypatch.setattr("src.core.thread_manager.time.sleep", lambda *_args: None)

    camera_thread._capture_frames()

    assert calls["count"] == 1
    assert camera_thread.startup_error is not None
    assert "giới hạn đăng nhập" in camera_thread.startup_error


def test_thread_manager_reports_startup_failure_without_timeout_log(monkeypatch, caplog):
    camera_id = uuid4()

    class DummyCaptureThread:
        def __init__(self):
            self.connected_event = Event()
            self.startup_error = "VSS server đang giới hạn đăng nhập (Login too frequently). Vui lòng đợi 120s rồi thử lại."
            self.thread = None

    class DummyManager:
        def __init__(self, *_args, **_kwargs):
            self.capture_thread = DummyCaptureThread()

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr("src.core.thread_manager.CameraManager", DummyManager)
    monkeypatch.setattr("src.core.thread_manager.uses_vss_backend", lambda _url: True)

    manager = ThreadManager()

    assert manager.add_camera(camera_id, "http://example.com/stream", show_display=False) is False
    assert "giới hạn đăng nhập" in (manager.get_last_start_error(camera_id) or "")
    assert f"Camera {camera_id}: Non-retryable error, removed:" in caplog.text
    assert f"Camera {camera_id}: Connection timeout, removing..." not in caplog.text


def test_camera_thread_does_not_mark_http_flv_connected_before_first_frame(monkeypatch):
    camera_thread = CameraThread(uuid4(), "http://example.com/stream", Queue(), Event())

    class DummyHTTPFLVCapture:
        def __init__(self):
            self.read_calls = 0

        def isOpened(self):
            return True

        def set(self, *_args, **_kwargs):
            return False

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 25.0
            return 0.0

        def signals_connection_on_open(self):
            return True

        def read(self):
            self.read_calls += 1
            camera_thread.stop_event.set()
            return False, None

        def release(self):
            return None

    monkeypatch.setattr(camera_thread, "_create_capture", lambda: DummyHTTPFLVCapture())
    monkeypatch.setattr("src.core.thread_manager.time.sleep", lambda *_args: None)

    camera_thread._capture_frames()

    assert camera_thread.connected_event.is_set() is False
    assert camera_thread.startup_error is None


def test_processor_thread_publishes_people_statuses_to_stream(monkeypatch):
    camera_id = uuid4()
    frame_queue = Queue()
    stop_event = Event()
    frame_queue.put(np.zeros((32, 32, 3), dtype=np.uint8))
    captured = {}

    class DummyProcessor:
        def __init__(self):
            self.models = {"petrolimex_detection_model": object()}

        def process_frame(self, frame):
            return {
                "petrolimex_detection_model": DetectionResult(
                    frame=frame.copy(),
                    event=False,
                    metadata={
                        "detections": [],
                        "people_statuses": [
                            {
                                "bbox": [1, 2, 10, 20],
                                "display_name": "person_positive",
                                "class_name": "person",
                                "confidence": 0.88,
                                "track_id": 99,
                            }
                        ],
                    },
                )
            }

        def compose_annotations(self, frame, _results):
            return frame.copy()

    monkeypatch.setattr(
        "src.core.thread_manager.SingleThreadProcessor",
        lambda _camera_id: DummyProcessor(),
    )
    monkeypatch.setattr("src.core.thread_manager.cv2.destroyWindow", lambda *_args: None)
    monkeypatch.setattr("src.core.thread_manager.cv2.waitKey", lambda *_args: 0)

    def fake_update(_camera_id, _frame, _timestamp, base_frame=None, detections=None):
        captured["detections"] = detections
        stop_event.set()

    monkeypatch.setattr("src.core.thread_manager.frame_buffer_manager.update", fake_update)

    processor_thread = ProcessorThread(
        camera_id,
        frame_queue,
        stop_event,
        show_display=False,
    )
    processor_thread._process_frames()

    assert captured["detections"] == [
        {
            "bbox": [1, 2, 10, 20],
            "label": "person_positive",
            "confidence": 0.88,
            "track_id": 99,
            "model_name": "petrolimex_detection_model",
        }
    ]


def test_processor_thread_close_display_window_only_destroys_created_window(monkeypatch):
    camera_id = uuid4()
    calls = {"destroy": 0, "wait": 0}

    monkeypatch.setattr(
        "src.core.thread_manager.SingleThreadProcessor",
        lambda _camera_id: type("DummyProcessor", (), {"models": {}})(),
    )
    monkeypatch.setattr(
        "src.core.thread_manager.cv2.destroyWindow",
        lambda *_args: calls.__setitem__("destroy", calls["destroy"] + 1),
    )
    monkeypatch.setattr(
        "src.core.thread_manager.cv2.waitKey",
        lambda *_args: calls.__setitem__("wait", calls["wait"] + 1) or 0,
    )

    processor_thread = ProcessorThread(
        camera_id,
        Queue(),
        Event(),
        show_display=False,
    )

    processor_thread._close_display_window()
    assert calls == {"destroy": 0, "wait": 0}

    processor_thread._window_created = True
    processor_thread._close_display_window()

    assert calls == {"destroy": 1, "wait": 1}
    assert processor_thread._window_created is False
