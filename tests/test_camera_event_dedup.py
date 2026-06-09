import threading
from uuid import uuid4

from src.core.camera_single_thread import SingleThreadProcessor


def _build_processor_for_dedup():
    processor = SingleThreadProcessor.__new__(SingleThreadProcessor)
    processor.camera_id = uuid4()
    processor._recent_event_lock = threading.Lock()
    processor._recent_event_cache = {}
    processor._EVENT_DUPLICATE_WINDOW_SECONDS = 2.0
    processor._EVENT_DUPLICATE_CACHE_TTL_SECONDS = 30.0
    processor._EVENT_DUPLICATE_BBOX_QUANTIZATION_PX = 32
    return processor


def test_event_dedup_key_prefers_track_id_over_bbox_jitter():
    processor = _build_processor_for_dedup()
    model_id = uuid4()

    metadata_a = {
        "eventType": "Phat hien khoi/chay",
        "violation": "SMOKE_FIRE:FIRE",
        "track_id": 1001,
        "bbox": [10, 10, 60, 60],
    }
    metadata_b = {
        "eventType": "Phat hien khoi/chay",
        "violation": "SMOKE_FIRE:FIRE",
        "track_id": 1001,
        "bbox": [24, 20, 74, 70],
    }

    key_a = processor._build_event_dedup_key(model_id, metadata_a)
    key_b = processor._build_event_dedup_key(model_id, metadata_b)

    assert key_a == key_b


def test_event_dedup_key_quantizes_bbox_when_track_missing():
    processor = _build_processor_for_dedup()
    model_id = uuid4()

    metadata_a = {
        "type": "Xam nhap trai phep khu vuc",
        "violation": "PEOPLE_CONTROL:ROI:0",
        "bbox": [10, 10, 50, 70],
    }
    metadata_b = {
        "type": "Xam nhap trai phep khu vuc",
        "violation": "PEOPLE_CONTROL:ROI:0",
        "bbox": [14, 12, 54, 72],
    }

    key_a = processor._build_event_dedup_key(model_id, metadata_a)
    key_b = processor._build_event_dedup_key(model_id, metadata_b)

    assert key_a == key_b


def test_reserve_recent_event_slot_blocks_same_track_with_bbox_jitter():
    processor = _build_processor_for_dedup()
    model_id = uuid4()

    metadata_a = {
        "eventType": "Phuong tien ra/vao khu vuc",
        "violation": "VEHICLE",
        "track_id": 7,
        "bbox": [100, 80, 180, 150],
    }
    metadata_b = {
        "eventType": "Phuong tien ra/vao khu vuc",
        "violation": "VEHICLE",
        "track_id": 7,
        "bbox": [112, 84, 192, 154],
    }

    first_allowed, _ = processor._reserve_recent_event_slot(model_id, metadata_a)
    second_allowed, _ = processor._reserve_recent_event_slot(model_id, metadata_b)

    assert first_allowed is True
    assert second_allowed is False


def test_reserve_recent_event_slot_allows_different_tracks():
    processor = _build_processor_for_dedup()
    model_id = uuid4()

    metadata_a = {
        "eventType": "Phat hien tran dau",
        "violation": "SUDDEN_OIL_FLOW",
        "track_id": 1,
        "bbox": [20, 20, 80, 80],
    }
    metadata_b = {
        "eventType": "Phat hien tran dau",
        "violation": "SUDDEN_OIL_FLOW",
        "track_id": 2,
        "bbox": [24, 24, 84, 84],
    }

    first_allowed, _ = processor._reserve_recent_event_slot(model_id, metadata_a)
    second_allowed, _ = processor._reserve_recent_event_slot(model_id, metadata_b)

    assert first_allowed is True
    assert second_allowed is True
