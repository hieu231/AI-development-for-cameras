from src.core.object_tracker import ObjectTracker, RecentViolationDeduplicator


def test_object_tracker_blocks_same_violation_even_after_save_cooldown(monkeypatch):
    now = {"value": 100.0}
    monkeypatch.setattr("src.core.object_tracker.time.time", lambda: now["value"])

    tracker = ObjectTracker(reset_interval=1800, save_cooldown=2.0)

    assert tracker.should_record_event(101, "FIRE") is True

    now["value"] = 101.0
    assert tracker.should_record_event(101, "FIRE") is False

    now["value"] = 103.5
    assert tracker.should_record_event(101, "FIRE") is False


def test_object_tracker_allows_new_violation_after_save_cooldown(monkeypatch):
    now = {"value": 200.0}
    monkeypatch.setattr("src.core.object_tracker.time.time", lambda: now["value"])

    tracker = ObjectTracker(reset_interval=1800, save_cooldown=2.0)

    assert tracker.should_record_event(202, "SMOKE") is True

    now["value"] = 201.0
    assert tracker.should_record_event(202, "FIRE") is False

    now["value"] = 202.5
    assert tracker.should_record_event(202, "FIRE") is True


def test_object_tracker_allows_same_violation_after_track_expires(monkeypatch):
    now = {"value": 300.0}
    monkeypatch.setattr("src.core.object_tracker.time.time", lambda: now["value"])

    tracker = ObjectTracker(reset_interval=10, save_cooldown=2.0)

    assert tracker.should_record_event(303, "NO_VEST") is True

    now["value"] = 311.0
    assert tracker.should_record_event(303, "NO_VEST") is True


def test_recent_violation_deduplicator_blocks_same_track_in_window():
    deduplicator = RecentViolationDeduplicator(window_seconds=2.0, iou_threshold=0.5)

    assert (
        deduplicator.is_recent_duplicate(
            "FIRE",
            10,
            (10, 10, 30, 30),
            current_time=100.0,
        )
        is False
    )

    deduplicator.remember_event(
        "FIRE",
        10,
        (10, 10, 30, 30),
        current_time=100.0,
    )
    assert (
        deduplicator.is_recent_duplicate(
            "FIRE",
            10,
            (12, 12, 32, 32),
            current_time=101.0,
        )
        is True
    )


def test_recent_violation_deduplicator_matches_by_iou_when_track_jitters():
    deduplicator = RecentViolationDeduplicator(window_seconds=2.0, iou_threshold=0.5)
    deduplicator.remember_event(
        "SMOKE",
        1001,
        (20, 20, 60, 60),
        current_time=200.0,
    )

    assert (
        deduplicator.is_recent_duplicate(
            "SMOKE",
            2002,
            (22, 22, 62, 62),
            current_time=201.0,
        )
        is True
    )


def test_recent_violation_deduplicator_expires_after_window():
    deduplicator = RecentViolationDeduplicator(window_seconds=1.0, iou_threshold=0.5)
    deduplicator.remember_event(
        "NO_VEST",
        55,
        (5, 5, 25, 35),
        current_time=300.0,
    )

    assert (
        deduplicator.is_recent_duplicate(
            "NO_VEST",
            55,
            (5, 5, 25, 35),
            current_time=301.2,
        )
        is False
    )
