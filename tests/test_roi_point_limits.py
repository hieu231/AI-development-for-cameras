from src.utils.roi_utils import validate_normalized_roi


def _create_polygon(point_count: int) -> list[list[float]]:
    return [
        [round(index / max(point_count, 1), 6), round(((index % 2) + 1) / 4, 6)]
        for index in range(point_count)
    ]


def test_validate_normalized_roi_accepts_minimum_point_count():
    is_valid, error = validate_normalized_roi(_create_polygon(3))

    assert is_valid is True
    assert error is None


def test_validate_normalized_roi_accepts_maximum_point_count():
    is_valid, error = validate_normalized_roi(_create_polygon(30))

    assert is_valid is True
    assert error is None


def test_validate_normalized_roi_rejects_more_than_thirty_points():
    is_valid, error = validate_normalized_roi(_create_polygon(31))

    assert is_valid is False
    assert error == "ROI must have at most 30 points"
