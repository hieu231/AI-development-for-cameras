"""
Unit tests for FrameBufferManager
"""
import time
import threading
import numpy as np
import pytest
from uuid import uuid4

from src.core.frame_buffer import FrameBufferManager


@pytest.fixture
def buf():
    """Return a fresh FrameBufferManager instance (bypass singleton for tests)."""
    mgr = object.__new__(FrameBufferManager)
    mgr._buffers = {}
    mgr._map_lock = threading.Lock()
    return mgr


@pytest.fixture
def cam_id():
    return uuid4()


def _make_frame(value: int = 0, h: int = 100, w: int = 100):
    """Create a simple BGR frame filled with *value*."""
    return np.full((h, w, 3), value, dtype=np.uint8)


class TestUpdateAndGet:
    def test_basic_round_trip(self, buf, cam_id):
        frame = _make_frame(42)
        buf.update(cam_id, frame, time.time())

        result = buf.get(cam_id)
        assert result is not None
        got_frame, got_ts = result
        assert np.array_equal(got_frame, frame)

    def test_get_returns_copy(self, buf, cam_id):
        frame = _make_frame(10)
        buf.update(cam_id, frame, time.time())

        result = buf.get(cam_id)
        got_frame, _ = result
        # Mutating returned frame should not change buffer
        got_frame[:] = 255
        result2 = buf.get(cam_id)
        assert np.array_equal(result2[0], frame)

    def test_get_nonexistent_camera(self, buf):
        assert buf.get(uuid4()) is None


class TestOverwrite:
    def test_latest_frame_wins(self, buf, cam_id):
        old_frame = _make_frame(1)
        new_frame = _make_frame(2)

        buf.update(cam_id, old_frame, time.time())
        buf.update(cam_id, new_frame, time.time())

        result = buf.get(cam_id)
        assert result is not None
        assert np.array_equal(result[0], new_frame)


class TestIsOnline:
    def test_online_when_fresh(self, buf, cam_id):
        buf.update(cam_id, _make_frame(), time.time())
        assert buf.is_online(cam_id, timeout=3.0) is True

    def test_offline_when_stale(self, buf, cam_id):
        buf.update(cam_id, _make_frame(), time.time() - 5.0)
        assert buf.is_online(cam_id, timeout=3.0) is False

    def test_offline_when_missing(self, buf):
        assert buf.is_online(uuid4()) is False


class TestRemove:
    def test_remove_clears_buffer(self, buf, cam_id):
        buf.update(cam_id, _make_frame(), time.time())
        buf.remove(cam_id)
        assert buf.get(cam_id) is None

    def test_remove_nonexistent_is_safe(self, buf):
        buf.remove(uuid4())  # should not raise


class TestThreadSafety:
    def test_concurrent_writes(self, buf, cam_id):
        """Multiple threads writing simultaneously should not crash."""
        errors = []

        def writer(value: int):
            try:
                for _ in range(100):
                    buf.update(cam_id, _make_frame(value), time.time())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Buffer should still be readable
        result = buf.get(cam_id)
        assert result is not None


class TestGetFrame:
    def test_convenience_method(self, buf, cam_id):
        frame = _make_frame(77)
        buf.update(cam_id, frame, time.time())
        got = buf.get_frame(cam_id)
        assert got is not None
        assert np.array_equal(got, frame)

    def test_returns_none_when_empty(self, buf):
        assert buf.get_frame(uuid4()) is None


class TestListCameras:
    def test_list_cameras(self, buf):
        id1, id2 = uuid4(), uuid4()
        buf.update(id1, _make_frame(), time.time())
        buf.update(id2, _make_frame(), time.time())
        cams = buf.list_cameras()
        assert set(cams) == {id1, id2}
