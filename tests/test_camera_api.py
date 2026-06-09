from fastapi import HTTPException
from starlette import status

from src.api.camera import _camera_start_http_error, _normalize_camera_rtsp_url


class DummyCamera:
    protocol = "VSS"
    vss_base_url = "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
    vss_username = "TEST1"
    vss_password = "4de93544234adffbb681ed60ffcfb941"
    vss_device_id = "HAN3-20-7209"
    vss_channel = "2"
    rtsp_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=2&stream=1&wnum=1&panel=0&buffer=2000"
    )


def test_normalize_camera_rtsp_url_builds_vss_realvideo_when_protocol_vss():
    camera_data = {
        "protocol": "VSS",
        "vss_base_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
        "vss_username": "TEST1",
        "vss_password": "4de93544234adffbb681ed60ffcfb941",
        "vss_device_id": "HAN3-20-7209",
        "vss_channel": "1",
        "rtsp_url": "",
    }

    _normalize_camera_rtsp_url(camera_data)

    assert camera_data["rtsp_url"] == (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1&wnum=1&panel=0&buffer=2000"
    )


def test_normalize_camera_rtsp_url_converts_direct_http_flv_to_realvideo():
    camera_data = {
        "rtsp_url": "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1",
        "vss_base_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
        "vss_username": "TEST1",
        "vss_password": "4de93544234adffbb681ed60ffcfb941",
        "vss_device_id": "HAN3-20-7209",
        "vss_channel": "1",
    }

    _normalize_camera_rtsp_url(camera_data)

    assert camera_data["protocol"] == "VSS"
    assert camera_data["rtsp_url"] == (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1&wnum=1&panel=0&buffer=2000"
        "&token=abc123"
    )


def test_normalize_camera_rtsp_url_rejects_direct_http_flv_without_vss_fields():
    camera_data = {
        "rtsp_url": "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1",
    }

    try:
        _normalize_camera_rtsp_url(camera_data)
    except HTTPException as exc:
        assert exc.status_code == 422
        assert "Direct HTTP-FLV URLs cannot be stored" in exc.detail
    else:
        raise AssertionError("Expected HTTPException for direct HTTP-FLV URL without VSS metadata")


def test_normalize_camera_rtsp_url_uses_existing_vss_fields_on_update():
    camera_data = {
        "rtsp_url": "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_2_1",
    }

    _normalize_camera_rtsp_url(camera_data, existing_camera=DummyCamera())

    assert camera_data["protocol"] == "VSS"
    assert camera_data["rtsp_url"] == (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=2&stream=1&wnum=1&panel=0&buffer=2000"
        "&token=abc123"
    )


def test_normalize_camera_rtsp_url_preserves_realvideo_token_when_rebuilding():
    camera_data = {
        "protocol": "VSS",
        "rtsp_url": (
            "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
            "?token=abc123&deviceId=HAN3-20-7209&chs=1&stream=1"
        ),
        "vss_base_url": "http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
        "vss_username": "TEST1",
        "vss_password": "4de93544234adffbb681ed60ffcfb941",
        "vss_device_id": "HAN3-20-7209",
        "vss_channel": "1",
    }

    _normalize_camera_rtsp_url(camera_data)

    assert "token=abc123" in camera_data["rtsp_url"]


def test_camera_start_http_error_maps_rate_limit_to_429(monkeypatch):
    class DummyThreadManager:
        def get_last_start_error(self, _camera_id):
            return "VSS server đang giới hạn đăng nhập (Login too frequently). Vui lòng đợi 120s rồi thử lại."

    monkeypatch.setattr("src.api.camera.thread_manager", DummyThreadManager())

    exc = _camera_start_http_error("camera-id")

    assert exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS