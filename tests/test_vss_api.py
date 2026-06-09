from types import SimpleNamespace
from uuid import uuid4

from fastapi import HTTPException

from src.api.vss import VSSBuildStreamRequest, build_stream_url, resolve_vss_url, VSSResolveRequest


def test_build_stream_url_uses_non_forced_token_resolution(monkeypatch):
    calls = []

    def fake_resolve_vss_token(url, force_login=False):
        calls.append((url, force_login))
        return "cached-token"

    monkeypatch.setattr("src.api.vss.resolve_vss_token", fake_resolve_vss_token)

    response = build_stream_url(
        VSSBuildStreamRequest(
            base_url="http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
            username="TEST1",
            password="4de93544234adffbb681ed60ffcfb941",
            channel="1",
            device_id="HAN3-20-7203",
        )
    )

    assert response.token == "cached-token"
    assert response.offer_url is None
    assert response.play_url is None
    assert calls and calls[0][1] is False


def test_build_stream_url_maps_rate_limit_to_429(monkeypatch):
    def fake_resolve_vss_token(*_args, **_kwargs):
        raise RuntimeError("VSS server đang giới hạn đăng nhập (Login too frequently). Vui lòng đợi 120s rồi thử lại.")

    monkeypatch.setattr("src.api.vss.resolve_vss_token", fake_resolve_vss_token)

    try:
        build_stream_url(
            VSSBuildStreamRequest(
                base_url="http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
                username="TEST1",
                password="4de93544234adffbb681ed60ffcfb941",
                channel="1",
                device_id="HAN3-20-7203",
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected HTTPException for VSS rate-limit")


def test_build_stream_url_returns_webrtc_play_url_when_camera_exists(monkeypatch):
    camera_id = uuid4()

    def fake_resolve_vss_token(url, force_login=False):
        return "cached-token"

    class DummyQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            return SimpleNamespace(id=camera_id)

    class DummyDb:
        def query(self, _model):
            return DummyQuery()

    request = SimpleNamespace(base_url="http://localhost:8668/")

    monkeypatch.setattr("src.api.vss.resolve_vss_token", fake_resolve_vss_token)

    response = build_stream_url(
        VSSBuildStreamRequest(
            base_url="http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
            username="TEST1",
            password="4de93544234adffbb681ed60ffcfb941",
            channel="1",
            device_id="HAN3-20-7203",
        ),
        request=request,
        db=DummyDb(),
    )

    assert response.offer_url == "http://localhost:8668/api/webrtc/offer"
    assert response.play_url == f"http://localhost:8668/api/webrtc/play/{camera_id}"


def test_build_stream_url_returns_null_play_url_when_camera_missing(monkeypatch):
    def fake_resolve_vss_token(url, force_login=False):
        return "cached-token"

    class DummyQuery:
        def filter(self, *_args, **_kwargs):
            return self

        def first(self):
            return None

    class DummyDb:
        def query(self, _model):
            return DummyQuery()

    request = SimpleNamespace(base_url="http://localhost:8668/")

    monkeypatch.setattr("src.api.vss.resolve_vss_token", fake_resolve_vss_token)

    response = build_stream_url(
        VSSBuildStreamRequest(
            base_url="http://203.171.17.183:9966/vss/apiPage/RealVideo.html",
            username="TEST1",
            password="4de93544234adffbb681ed60ffcfb941",
            channel="1",
            device_id="HAN3-20-7203",
        ),
        request=request,
        db=DummyDb(),
    )

    assert response.offer_url == "http://localhost:8668/api/webrtc/offer"
    assert response.play_url is None


def test_resolve_vss_url_maps_rate_limit_to_429(monkeypatch):
    def fake_resolve_vss_token(*_args, **_kwargs):
        raise RuntimeError("Login too frequently")

    monkeypatch.setattr("src.api.vss.resolve_vss_token", fake_resolve_vss_token)

    try:
        resolve_vss_url(
            VSSResolveRequest(
                url="http://203.171.17.183:9966/vss/apiPage/RealVideo.html?deviceId=HAN3-20-7203&chs=1&stream=1",
                force_login=False,
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("Expected HTTPException for VSS rate-limit")