import subprocess

import requests
import pytest

from src.core import capture_backends
from src.core.capture_backends import (
    HOWEN_WS_VIDEO_FRAME_TYPES,
    _prepare_howen_video_payload,
    HTTPFLVCapture,
    build_refreshable_vss_source_url,
    build_vss_flv_url,
    create_vss_capture,
    resolve_vss_token,
    uses_vss_backend,
)


def test_build_vss_flv_url_from_realvideo_page():
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=abc123&deviceId=HAN3-20-7209&chs=1&stream=1&wnum=1&panel=0&buffer=2000"
    )

    assert build_vss_flv_url(url) == (
        "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1"
    )


def test_build_vss_flv_url_uses_first_channel_when_multiple():
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=abc123&deviceId=HAN3-20-7209&chs=1_2_3&stream=0"
    )

    assert build_vss_flv_url(url) == (
        "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_0"
    )


def test_direct_http_flv_url_passthrough():
    url = "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1"
    assert build_vss_flv_url(url) == url


def test_build_refreshable_vss_source_url_from_http_flv():
    flv_url = "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1"
    base_url = "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"

    assert build_refreshable_vss_source_url(
        flv_url,
        base_url=base_url,
        username="TEST1",
        password_md5="4de93544234adffbb681ed60ffcfb941",
    ) == (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1&wnum=1&panel=0&buffer=2000&token=abc123"
    )


def test_backend_detection_for_vss_sources():
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=abc123&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    flv_url = "http://203.171.17.183:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1"
    rtsp_url = "rtsp://admin:password@192.168.1.10:554/stream"

    assert uses_vss_backend(realvideo_url) is True
    assert uses_vss_backend(flv_url) is True
    assert uses_vss_backend(rtsp_url) is False


def test_resolve_vss_token_from_login(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")
    monkeypatch.setenv("VSS_PREFER_LOGIN_TOKEN", "true")

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"token": "fresh-token-123"}}

        @property
        def text(self):
            return ""

    def fake_post(login_url, json, timeout):
        assert login_url == "http://203.171.17.183:9966/vss/user/apiLogin.action"
        assert json == {"username": "TEST1", "password": "4de93544234adffbb681ed60ffcfb941"}
        assert timeout > 0
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    assert resolve_vss_token(url, force_login=True) == "fresh-token-123"


def test_build_vss_flv_url_uses_login_when_token_missing(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?deviceId=HAN3-20-7209&chs=2&stream=1"
    )

    monkeypatch.setattr(
        "src.core.capture_backends.resolve_vss_token",
        lambda _url, force_login=False: "login-token-xyz",
    )
    monkeypatch.setattr(
        "src.core.capture_backends.resolve_vss_stream_host",
        lambda _url, device_id: "203.171.17.183",
    )

    assert build_vss_flv_url(url) == (
        "http://203.171.17.183:33122/flvRouter.php?live?login-token-xyz_HAN3-20-7209_2_1"
    )


def test_resolve_vss_token_uses_url_token_when_login_not_preferred(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=static-token&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.delenv("VSS_USERNAME", raising=False)
    monkeypatch.delenv("VSS_PASSWORD_MD5", raising=False)
    monkeypatch.setenv("VSS_PREFER_LOGIN_TOKEN", "false")

    assert resolve_vss_token(url) == "static-token"


def test_resolve_vss_token_uses_url_token_without_login_when_not_forced(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=static-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_PREFER_LOGIN_TOKEN", "true")
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})
    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})

    def fail_post(*_args, **_kwargs):
        raise AssertionError("requests.post should not be called when a URL token is available")

    monkeypatch.setattr(requests, "post", fail_post)

    assert resolve_vss_token(url) == "static-token"


def test_resolve_vss_token_force_login_uses_url_token_during_rate_limit(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=static-token&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")

    login_url = "http://203.171.17.183:9966/vss/user/apiLogin.action"
    cache_key = (login_url, "TEST1")
    future = 10**10

    monkeypatch.setattr(capture_backends, "_VSS_RATE_LIMITED_UNTIL", future)
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {cache_key: ("cached-token", future)})

    assert resolve_vss_token(url, force_login=True) == "static-token"


def test_resolve_vss_token_force_login_still_raises_during_rate_limit_without_url_token(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")

    future = 10**10

    monkeypatch.setattr(capture_backends, "_VSS_RATE_LIMITED_UNTIL", future)
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})

    with pytest.raises(RuntimeError, match="giới hạn đăng nhập"):
        resolve_vss_token(url, force_login=True)


def test_resolve_vss_token_force_login_uses_cached_token_during_rate_limit_when_url_token_missing(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")

    login_url = "http://203.171.17.183:9966/vss/user/apiLogin.action"
    cache_key = (login_url, "TEST1")
    future = 10**10

    monkeypatch.setattr(capture_backends, "_VSS_RATE_LIMITED_UNTIL", future)
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {cache_key: ("cached-token", future)})

    assert resolve_vss_token(url, force_login=True) == "cached-token"


def test_resolve_vss_token_non_forced_uses_cached_token_during_rate_limit(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=static-token&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")

    login_url = "http://203.171.17.183:9966/vss/user/apiLogin.action"
    cache_key = (login_url, "TEST1")
    future = 10**10

    monkeypatch.setattr(capture_backends, "_VSS_RATE_LIMITED_UNTIL", future)
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {cache_key: ("cached-token", future)})

    assert resolve_vss_token(url, force_login=False) == "cached-token"


def test_resolve_vss_token_logs_cache_hit(monkeypatch, caplog):
    caplog.set_level("INFO")
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=static-token&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")

    login_url = "http://203.171.17.183:9966/vss/user/apiLogin.action"
    cache_key = (login_url, "TEST1")
    future = 10**10

    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {cache_key: ("cached-token", future)})

    assert resolve_vss_token(url, force_login=False) == "cached-token"
    assert "VSS token cache hit" in caplog.text


def test_resolve_vss_token_logs_cache_miss_and_store(monkeypatch, caplog):
    caplog.set_level("INFO")
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setenv("VSS_USERNAME", "TEST1")
    monkeypatch.setenv("VSS_PASSWORD_MD5", "4de93544234adffbb681ed60ffcfb941")
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"token": "fresh-token-123"}}

        @property
        def text(self):
            return ""

    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: DummyResponse())

    assert resolve_vss_token(url, force_login=True) == "fresh-token-123"
    assert "VSS token cache miss" in caplog.text
    assert "VSS token cached" in caplog.text


def test_build_vss_flv_url_uses_resolved_gt_host(monkeypatch):
    url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=abc123&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    monkeypatch.setattr(
        "src.core.capture_backends.resolve_vss_stream_host",
        lambda _url, device_id: "refueller.petrolimexaviation.com",
    )

    assert build_vss_flv_url(url) == (
        "http://refueller.petrolimexaviation.com:33122/flvRouter.php?live?abc123_HAN3-20-7209_1_1"
    )


def test_build_vss_flv_url_rejects_malformed_vss_host():
    url = (
        "http://http:9966/vss/apiPage/RealVideo.html"
        "?token=abc123&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    with pytest.raises(ValueError, match="malformed"):
        build_vss_flv_url(url)


def test_http_flv_capture_raises_rate_limit_after_auth_failure(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    def fake_build_vss_flv_url(_url, force_login=False):
        if force_login:
            raise RuntimeError(
                "VSS server đang giới hạn đăng nhập (Login too frequently). Vui lòng đợi 120s rồi thử lại."
            )
        return "http://203.171.17.183:33122/flvRouter.php?live?stale-token_HAN3-20-7209_1_1"

    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(HTTPFLVCapture, "_probe_stream", lambda self: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")))
    monkeypatch.setattr(HTTPFLVCapture, "release", lambda self: None)

    with pytest.raises(RuntimeError, match="giới hạn đăng nhập"):
        HTTPFLVCapture(realvideo_url)


def test_http_flv_capture_does_not_force_refresh_after_probe_timeout(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    force_login_calls = {"count": 0}

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    def fake_build_vss_flv_url(_url, force_login=False):
        if force_login:
            force_login_calls["count"] += 1
        return "http://203.171.17.183:33122/flvRouter.php?live?stale-token_HAN3-20-7209_1_1"

    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})
    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_RECENT_CHANNEL_TOKENS", {})
    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(
        HTTPFLVCapture,
        "_probe_stream",
        lambda self: (_ for _ in ()).throw(subprocess.TimeoutExpired(["ffprobe"], 20)),
    )
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert capture._output_mode == "mjpeg"
    assert force_login_calls["count"] == 0
    capture.release()


def test_http_flv_capture_falls_back_to_mjpeg_when_ffprobe_missing(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    def fake_resolve_binary(binary_name, _env_var_name):
        if binary_name == "ffprobe":
            return None
        return "ffmpeg"

    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_RECENT_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", fake_resolve_binary)
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert capture._output_mode == "mjpeg"
    assert capture.get(7) == 0.0
    capture.release()


def test_http_flv_capture_uses_reusable_token_path_when_no_active_channel_peer(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    force_login_calls = []

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    def fake_build_vss_flv_url(_url, force_login=False):
        force_login_calls.append(force_login)
        return "http://203.171.17.183:33122/flvRouter.php?live?fresh-token_HAN3-20-7209_1_1"

    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_RECENT_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})
    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(HTTPFLVCapture, "_probe_stream", lambda self: (1920, 1080, 25.0))
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert force_login_calls == [False]
    capture.release()


def test_http_flv_capture_reuses_token_when_same_channel_is_active(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    force_login_calls = []

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    session_key = (
        "http://203.171.17.183:9966/vss/user/apiLogin.action",
        "TEST1",
        "HAN3-20-7209",
        "1",
        "1",
    )

    def fake_build_vss_flv_url(_url, force_login=False):
        force_login_calls.append(force_login)
        return "http://203.171.17.183:33122/flvRouter.php?live?shared-token_HAN3-20-7209_1_1"

    monkeypatch.setattr(
        capture_backends,
        "_VSS_ACTIVE_CHANNEL_TOKENS",
        {session_key: {"token": "shared-token", "owners": {"camera-1"}}},
    )
    monkeypatch.setattr(capture_backends, "_VSS_RECENT_CHANNEL_TOKENS", {})
    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(HTTPFLVCapture, "_probe_stream", lambda self: (1920, 1080, 25.0))
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert force_login_calls == [False]
    capture.release()


def test_http_flv_capture_uses_reusable_token_path_after_last_channel_peer_stops(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    force_login_calls = []

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    session_key = (
        "http://203.171.17.183:9966/vss/user/apiLogin.action",
        "TEST1",
        "HAN3-20-7209",
        "1",
        "1",
    )

    def fake_build_vss_flv_url(_url, force_login=False):
        force_login_calls.append(force_login)
        return "http://203.171.17.183:33122/flvRouter.php?live?recent-token_HAN3-20-7209_1_1"

    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {})
    monkeypatch.setattr(
        capture_backends,
        "_VSS_RECENT_CHANNEL_TOKENS",
        {session_key: ("recent-token", 10**10)},
    )
    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(HTTPFLVCapture, "_probe_stream", lambda self: (1920, 1080, 25.0))
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert force_login_calls == [False]
    capture.release()


def test_http_flv_capture_uses_cached_token_path_when_no_active_peer_even_if_cached(monkeypatch):
    realvideo_url = (
        "http://203.171.17.183:9966/vss/apiPage/RealVideo.html"
        "?token=stale-token&username=TEST1&password=4de93544234adffbb681ed60ffcfb941"
        "&deviceId=HAN3-20-7209&chs=1&stream=1"
    )
    force_login_calls = []

    class DummyProcess:
        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return None

    login_url = "http://203.171.17.183:9966/vss/user/apiLogin.action"
    cache_key = (login_url, "TEST1")

    def fake_build_vss_flv_url(_url, force_login=False):
        force_login_calls.append(force_login)
        return "http://203.171.17.183:33122/flvRouter.php?live?cached-token_HAN3-20-7209_1_1"

    monkeypatch.setattr(capture_backends, "_VSS_ACTIVE_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_RECENT_CHANNEL_TOKENS", {})
    monkeypatch.setattr(capture_backends, "_VSS_TOKEN_CACHE", {cache_key: ("cached-token", 10**10)})
    monkeypatch.setattr("src.core.capture_backends.build_vss_flv_url", fake_build_vss_flv_url)
    monkeypatch.setattr("src.core.capture_backends._resolve_binary", lambda *_args: "ffmpeg")
    monkeypatch.setattr(HTTPFLVCapture, "_probe_stream", lambda self: (1920, 1080, 25.0))
    monkeypatch.setattr(HTTPFLVCapture, "_start_process", lambda self: setattr(self, "_process", DummyProcess()))

    capture = HTTPFLVCapture(realvideo_url)

    assert capture.isOpened() is True
    assert force_login_calls == [False]
    capture.release()


def test_create_vss_capture_auto_prefers_websocket(monkeypatch):
    calls = []

    class DummyCapture:
        pass

    monkeypatch.setenv("VSS_CAPTURE_BACKEND", "auto")
    monkeypatch.setattr("src.core.capture_backends.HowenWebSocketCapture", lambda url: calls.append(("ws", url)) or DummyCapture())
    monkeypatch.setattr("src.core.capture_backends.HTTPFLVCapture", lambda url: calls.append(("flv", url)) or DummyCapture())

    capture = create_vss_capture("http://203.171.17.183:9966/vss/apiPage/RealVideo.html?deviceId=HAN3-20-7203&chs=1&stream=1")

    assert isinstance(capture, DummyCapture)
    assert calls == [("ws", "http://203.171.17.183:9966/vss/apiPage/RealVideo.html?deviceId=HAN3-20-7203&chs=1&stream=1")]


def test_create_vss_capture_auto_falls_back_to_http_flv(monkeypatch):
    calls = []

    class DummyCapture:
        pass

    monkeypatch.setenv("VSS_CAPTURE_BACKEND", "auto")

    def fail_ws(url):
        calls.append(("ws", url))
        raise RuntimeError("ws unavailable")

    monkeypatch.setattr("src.core.capture_backends.HowenWebSocketCapture", fail_ws)
    monkeypatch.setattr("src.core.capture_backends.HTTPFLVCapture", lambda url: calls.append(("flv", url)) or DummyCapture())

    capture = create_vss_capture("http://203.171.17.183:9966/vss/apiPage/RealVideo.html?deviceId=HAN3-20-7203&chs=1&stream=1")

    assert isinstance(capture, DummyCapture)
    assert calls[-1][0] == "flv"


def test_prepare_howen_video_payload_normalizes_short_annex_b_prefixes():
    assert _prepare_howen_video_payload(b"\x00\x01\x64\x00") == b"\x00\x00\x00\x01\x64\x00"
    assert _prepare_howen_video_payload(b"\x00\x00\x01\x67\x64") == b"\x00\x00\x00\x01\x67\x64"


def test_howen_default_video_frame_types_include_type_3():
    assert 3 in HOWEN_WS_VIDEO_FRAME_TYPES