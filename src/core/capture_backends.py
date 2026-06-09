import json
import logging
import os
import select
import shutil
import struct
import subprocess
import threading
import time
from fractions import Fraction
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import src.core.opencv_config  # noqa: F401  — MUST precede `import cv2`
import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

HTTP_FLV_RW_TIMEOUT_US = int(os.getenv("HTTP_FLV_RW_TIMEOUT_US", "15000000"))
HTTP_FLV_PROBE_TIMEOUT_SEC = int(os.getenv("HTTP_FLV_PROBE_TIMEOUT_SEC", "20"))
DEFAULT_HTTP_FLV_FPS = 25.0
HTTP_FLV_PIPE_READ_SIZE = int(os.getenv("HTTP_FLV_PIPE_READ_SIZE", str(64 * 1024)))
HTTP_FLV_MJPEG_BUFFER_LIMIT = int(
    os.getenv("HTTP_FLV_MJPEG_BUFFER_LIMIT", str(8 * 1024 * 1024))
)
HTTP_FLV_READ_TIMEOUT_SEC = float(os.getenv("HTTP_FLV_READ_TIMEOUT_SEC", "1.0"))
VSS_LOGIN_TIMEOUT_SEC = int(os.getenv("VSS_LOGIN_TIMEOUT_SEC", "15"))
VSS_TOKEN_CACHE_TTL_SEC = int(os.getenv("VSS_TOKEN_CACHE_TTL_SEC", "1500"))
VSS_DEVICE_QUERY_TIMEOUT_SEC = int(os.getenv("VSS_DEVICE_QUERY_TIMEOUT_SEC", "15"))
VSS_PREFER_LOGIN_TOKEN = os.getenv("VSS_PREFER_LOGIN_TOKEN", "true").lower() == "true"
VSS_STREAM_PORT_HTTP = int(os.getenv("VSS_STREAM_PORT_HTTP", "33122"))
VSS_STREAM_PORT_HTTPS = int(os.getenv("VSS_STREAM_PORT_HTTPS", "9987"))
FFMPEG_HW_DECODE = os.getenv("FFMPEG_HW_DECODE", "auto").strip().lower()
FFMPEG_HW_DECODE_TIMEOUT_SEC = float(os.getenv("FFMPEG_HW_DECODE_TIMEOUT_SEC", "3.0"))

_TOKEN_CACHE_LOCK = threading.Lock()
_VSS_TOKEN_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_VSS_ACTIVE_CHANNEL_LOCK = threading.Lock()
_VSS_ACTIVE_CHANNEL_TOKENS: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
_VSS_RECENT_CHANNEL_TOKENS: dict[tuple[str, str, str, str, str], tuple[str, float]] = {}

# Global login lock: ensures only one thread does a VSS login at a time.
# Other threads wait and then use the cached token.
_VSS_LOGIN_LOCK = threading.Lock()
_VSS_LAST_LOGIN_TIME: float = 0.0
VSS_LOGIN_COOLDOWN_SEC = float(os.getenv("VSS_LOGIN_COOLDOWN_SEC", "3"))

# Rate-limit tracking: when VSS says "Login too frequently", we record a
# cooldown timestamp so that subsequent calls fail fast instead of making
# the rate-limit situation worse.
_VSS_RATE_LIMITED_UNTIL: float = 0.0
VSS_RATE_LIMIT_BACKOFF_SEC = float(os.getenv("VSS_RATE_LIMIT_BACKOFF_SEC", "120"))
VSS_RELEASED_TOKEN_GRACE_SEC = float(os.getenv("VSS_RELEASED_TOKEN_GRACE_SEC", "180"))

_FFMPEG_CAPS_LOCK = threading.Lock()
_FFMPEG_CAPS_CACHE: dict[str, tuple[bool, frozenset[str]]] = {}
_FFMPEG_HW_DECODE_MODE_WARNED = False

_CUVID_DECODER_BY_CODEC = {
    "av1": "av1_cuvid",
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "mpeg1video": "mpeg1_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
}


def _resolve_binary(binary_name: str, env_var_name: str) -> str | None:
    configured_path = os.getenv(env_var_name)
    if configured_path:
        if os.path.exists(configured_path):
            logger.info(f"{binary_name}: Using configured path {configured_path}")
            return configured_path
        else:
            logger.warning(
                f"{binary_name}: Configured path does not exist: {configured_path}"
            )

    found_path = shutil.which(binary_name)
    if found_path:
        logger.info(f"{binary_name}: Found at {found_path}")
    else:
        logger.warning(
            f"{binary_name}: Not found in PATH. Install via: apt-get install {binary_name} (Linux) or brew install {binary_name} (macOS)"
        )
    return found_path


def _wants_nvidia_hw_decode() -> bool:
    global _FFMPEG_HW_DECODE_MODE_WARNED
    if FFMPEG_HW_DECODE in {"0", "false", "off", "none", "cpu", "disable"}:
        return False
    if FFMPEG_HW_DECODE in {"", "1", "true", "auto", "cuda", "nvidia", "nvdec"}:
        return True
    if not _FFMPEG_HW_DECODE_MODE_WARNED:
        logger.warning(
            "Unknown FFMPEG_HW_DECODE=%s (expected auto|cuda|nvidia|nvdec|off|cpu). "
            "Defaulting to auto.",
            FFMPEG_HW_DECODE,
        )
        _FFMPEG_HW_DECODE_MODE_WARNED = True
    return True


def _get_ffmpeg_decode_caps(ffmpeg_path: str) -> tuple[bool, frozenset[str]]:
    with _FFMPEG_CAPS_LOCK:
        cached = _FFMPEG_CAPS_CACHE.get(ffmpeg_path)
    if cached is not None:
        return cached

    cuda_hwaccel = False
    cuvid_decoders: set[str] = set()

    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_HW_DECODE_TIMEOUT_SEC,
            check=False,
        )
        hwaccels_output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        cuda_hwaccel = "\ncuda\n" in f"\n{hwaccels_output}\n"
    except Exception as exc:  # pragma: no cover - best-effort probe
        logger.debug("FFmpeg hwaccels probe failed for %s: %s", ffmpeg_path, exc)

    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=FFMPEG_HW_DECODE_TIMEOUT_SEC,
            check=False,
        )
        decoders_output = f"{result.stdout or ''}\n{result.stderr or ''}"
        for line in decoders_output.splitlines():
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            decoder_name = tokens[1].strip().lower()
            if decoder_name.endswith("_cuvid"):
                cuvid_decoders.add(decoder_name)
    except Exception as exc:  # pragma: no cover - best-effort probe
        logger.debug("FFmpeg decoders probe failed for %s: %s", ffmpeg_path, exc)

    caps = (cuda_hwaccel, frozenset(cuvid_decoders))
    with _FFMPEG_CAPS_LOCK:
        _FFMPEG_CAPS_CACHE[ffmpeg_path] = caps
    return caps


def _build_ffmpeg_hwdecode_input_args(
    ffmpeg_path: str,
    codec_name: str | None,
    *,
    source_name: str,
) -> list[str]:
    if not _wants_nvidia_hw_decode():
        return []

    cuda_hwaccel, cuvid_decoders = _get_ffmpeg_decode_caps(ffmpeg_path)
    if not cuda_hwaccel:
        logger.debug(
            "%s: FFmpeg has no CUDA hwaccel support. Using CPU decode.",
            source_name,
        )
        return []

    args = ["-hwaccel", "cuda"]
    normalized_codec = (codec_name or "").strip().lower()
    decoder_name = _CUVID_DECODER_BY_CODEC.get(normalized_codec)
    if decoder_name and decoder_name in cuvid_decoders:
        args.extend(["-c:v", decoder_name])
        logger.info(
            "%s: enabling NVIDIA hardware decode (decoder=%s).",
            source_name,
            decoder_name,
        )
    elif normalized_codec:
        logger.info(
            "%s: enabling CUDA hwaccel (no cuvid decoder mapping for codec=%s).",
            source_name,
            normalized_codec,
        )
    else:
        logger.info("%s: enabling CUDA hwaccel.", source_name)

    return args


def is_vss_realvideo_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.path.lower().endswith("/vss/apipage/realvideo.html")


def is_http_flv_url(url: str) -> bool:
    parts = urlsplit(url)
    return parts.path.lower().endswith(
        "/flvrouter.php"
    ) and parts.query.lower().startswith("live?")


def uses_vss_backend(url: str) -> bool:
    return is_vss_realvideo_url(url) or is_http_flv_url(url)


def _extract_http_flv_stream_parts(url: str) -> tuple[str, str, str, str]:
    if not is_http_flv_url(url):
        raise ValueError(f"Not a VSS HTTP-FLV URL: {url}")

    stream_spec = urlsplit(url).query[5:]
    try:
        token, device_id, channel, stream = stream_spec.rsplit("_", 3)
    except ValueError as exc:
        raise ValueError(f"Malformed VSS HTTP-FLV URL: {url}") from exc

    if not token or not device_id or not channel or not stream:
        raise ValueError(f"Malformed VSS HTTP-FLV URL: {url}")

    return token, device_id, channel, stream


def build_vss_realvideo_url(
    base_url: str,
    *,
    username: str,
    password_md5: str,
    device_id: str,
    channel: str,
    stream: str = "1",
    token: str | None = None,
) -> str:
    if not is_vss_realvideo_url(base_url):
        raise ValueError(f"Not a VSS RealVideo URL: {base_url}")

    parts = urlsplit(base_url)
    query = parse_qs(parts.query, keep_blank_values=True)
    query["username"] = [username]
    query["password"] = [password_md5]
    query["deviceId"] = [device_id]
    query["chs"] = [channel]
    query["stream"] = [stream]
    query.setdefault("wnum", ["1"])
    query.setdefault("panel", ["0"])
    query.setdefault("buffer", ["2000"])
    if token:
        query["token"] = [token]

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(query, doseq=True),
            parts.fragment,
        )
    )


def build_refreshable_vss_source_url(
    url: str,
    *,
    base_url: str | None = None,
    username: str | None = None,
    password_md5: str | None = None,
    device_id: str | None = None,
    channel: str | None = None,
    stream: str | None = None,
) -> str:
    if not is_http_flv_url(url):
        return url

    if not base_url or not username or not password_md5:
        return url

    token, parsed_device_id, parsed_channel, parsed_stream = (
        _extract_http_flv_stream_parts(url)
    )
    return build_vss_realvideo_url(
        base_url,
        username=username,
        password_md5=password_md5,
        device_id=device_id or parsed_device_id,
        channel=channel or parsed_channel,
        stream=stream or parsed_stream,
        token=token,
    )


def build_vss_ws_params(url: str, force_login: bool = False) -> dict:
    """Extract WebSocket connection parameters from a VSS RealVideo URL.
    Returns dict with keys: ws_url, token, device_id, channel, stream.
    """
    if not is_vss_realvideo_url(url):
        raise ValueError(f"Not a VSS RealVideo URL: {url}")

    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)

    token = resolve_vss_token(url, force_login=force_login)
    device_id = _get_required_query_value(query, "deviceId")
    channels = _get_required_query_value(query, "chs")
    stream = _get_required_query_value(query, "stream")
    channel = channels.split("_")[0]

    stream_host = resolve_vss_stream_host(url, device_id)
    ws_scheme = "wss" if parts.scheme.lower() == "https" else "ws"
    port = (
        VSS_STREAM_PORT_HTTPS
        if parts.scheme.lower() == "https"
        else VSS_STREAM_PORT_HTTP
    )
    ws_url = f"{ws_scheme}://{stream_host}:{port}/stream"

    return {
        "ws_url": ws_url,
        "token": token,
        "device_id": device_id,
        "channel": channel,
        "stream": stream,
    }


def _build_vss_channel_session_key(url: str) -> tuple[str, str, str, str, str] | None:
    if not is_vss_realvideo_url(url):
        return None

    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    username, _password_md5 = _get_vss_credentials(url)
    if not username:
        return None

    login_url = _build_vss_login_url(url)
    device_id = _get_required_query_value(query, "deviceId")
    channels = _get_required_query_value(query, "chs")
    stream = _get_required_query_value(query, "stream")
    channel = channels.split("_")[0]
    return (login_url, username, device_id, channel, stream)


def _get_active_vss_channel_token(url: str) -> str | None:
    session_key = _build_vss_channel_session_key(url)
    if session_key is None:
        return None

    with _VSS_ACTIVE_CHANNEL_LOCK:
        entry = _VSS_ACTIVE_CHANNEL_TOKENS.get(session_key)
        if not entry:
            logger.info("VSS active channel token miss (session_key=%s)", session_key)
            return None
        token = str(entry["token"])
        owners = entry.get("owners", set())
        logger.info(
            "VSS active channel token hit (session_key=%s, owners=%d, token=%s)",
            session_key,
            len(owners),
            _mask_vss_token(token),
        )
        return token


def _get_recent_vss_channel_token(url: str) -> str | None:
    session_key = _build_vss_channel_session_key(url)
    if session_key is None:
        return None

    with _VSS_ACTIVE_CHANNEL_LOCK:
        entry = _VSS_RECENT_CHANNEL_TOKENS.get(session_key)
        if not entry:
            logger.info("VSS recent channel token miss (session_key=%s)", session_key)
            return None
        token, expires_at = entry
        if expires_at <= time.time():
            _VSS_RECENT_CHANNEL_TOKENS.pop(session_key, None)
            logger.info(
                "VSS recent channel token stale (session_key=%s, token=%s)",
                session_key,
                _mask_vss_token(token),
            )
            return None
        logger.info(
            "VSS recent channel token hit (session_key=%s, token=%s)",
            session_key,
            _mask_vss_token(token),
        )
        return token


def _get_cached_vss_token_for_url(url: str, force_login: bool = False) -> str | None:
    if not is_vss_realvideo_url(url):
        return None

    username, _password_md5 = _get_vss_credentials(url)
    if not username:
        return None

    login_url = _build_vss_login_url(url)
    cache_key = (login_url, username)
    cache_context = (
        f"login_url={login_url}, username={username}, force_login={force_login}"
    )

    with _TOKEN_CACHE_LOCK:
        cached_entry = _VSS_TOKEN_CACHE.get(cache_key)
        if cached_entry is not None:
            cached_token, expires_at = cached_entry
            if expires_at > time.time():
                logger.info(
                    "VSS token cache hit (%s, token=%s)",
                    cache_context,
                    _mask_vss_token(cached_token),
                )
                return cached_token
            logger.info(
                "VSS token cache stale (%s, token=%s)",
                cache_context,
                _mask_vss_token(cached_token),
            )
    logger.info("VSS token cache miss (%s)", cache_context)
    return None


def _build_vss_flv_url_from_session_token(url: str, token: str) -> str:
    if not is_vss_realvideo_url(url):
        raise ValueError(f"Not a VSS RealVideo URL: {url}")

    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    device_id = _get_required_query_value(query, "deviceId")
    channels = _get_required_query_value(query, "chs")
    stream = _get_required_query_value(query, "stream")
    channel = channels.split("_")[0]
    stream_host = resolve_vss_stream_host(url, device_id)
    port = (
        VSS_STREAM_PORT_HTTPS
        if parts.scheme.lower() == "https"
        else VSS_STREAM_PORT_HTTP
    )
    return f"{parts.scheme}://{stream_host}:{port}/flvRouter.php?live?{token}_{device_id}_{channel}_{stream}"


def _register_active_vss_channel_token(url: str, token: str, owner_id: str) -> None:
    session_key = _build_vss_channel_session_key(url)
    if session_key is None:
        return

    with _VSS_ACTIVE_CHANNEL_LOCK:
        _VSS_RECENT_CHANNEL_TOKENS.pop(session_key, None)
        entry = _VSS_ACTIVE_CHANNEL_TOKENS.setdefault(
            session_key, {"token": token, "owners": set()}
        )
        entry["token"] = token
        owners = entry.setdefault("owners", set())
        owners.add(owner_id)
        logger.info(
            "VSS active channel token registered (session_key=%s, owners=%d, token=%s)",
            session_key,
            len(owners),
            _mask_vss_token(token),
        )


def _unregister_active_vss_channel_token(url: str, owner_id: str) -> None:
    session_key = _build_vss_channel_session_key(url)
    if session_key is None:
        return

    with _VSS_ACTIVE_CHANNEL_LOCK:
        entry = _VSS_ACTIVE_CHANNEL_TOKENS.get(session_key)
        if not entry:
            return
        owners = entry.get("owners", set())
        owners.discard(owner_id)
        if owners:
            logger.info(
                "VSS active channel token owner removed (session_key=%s, owners=%d)",
                session_key,
                len(owners),
            )
            return
        _VSS_ACTIVE_CHANNEL_TOKENS.pop(session_key, None)
        _VSS_RECENT_CHANNEL_TOKENS.pop(session_key, None)
        logger.info("VSS active channel token cleared (session_key=%s)", session_key)


def build_vss_flv_url(url: str, force_login: bool = False) -> str:
    logger.info(f"build_vss_flv_url: input url={url}, force_login={force_login}")

    if is_http_flv_url(url):
        logger.info(
            f"build_vss_flv_url: URL is already HTTP-FLV format, returning as-is"
        )
        return url
    if not is_vss_realvideo_url(url):
        logger.info(
            f"build_vss_flv_url: URL is not VSS RealVideo format, returning as-is"
        )
        return url

    parts = urlsplit(url)
    logger.info(
        f"build_vss_flv_url: parsed scheme={parts.scheme}, hostname={parts.hostname}, port={parts.port}, path={parts.path}"
    )

    query = parse_qs(parts.query, keep_blank_values=True)
    try:
        token = resolve_vss_token(url, force_login=force_login)
        device_id = _get_required_query_value(query, "deviceId")
        channels = _get_required_query_value(query, "chs")
        stream = _get_required_query_value(query, "stream")
    except Exception as e:
        logger.error(f"build_vss_flv_url: Failed to extract required query params: {e}")
        raise

    channel = channels.split("_")[0]
    if "_" in channels:
        logger.warning(
            "VSS RealVideo URL contains multiple channels (%s); using first channel %s for AI ingest",
            channels,
            channel,
        )

    stream_host = resolve_vss_stream_host(url, device_id)
    port = (
        VSS_STREAM_PORT_HTTPS
        if parts.scheme.lower() == "https"
        else VSS_STREAM_PORT_HTTP
    )
    result_url = f"{parts.scheme}://{stream_host}:{port}/flvRouter.php?live?{token}_{device_id}_{channel}_{stream}"
    logger.info(f"build_vss_flv_url: final FLV URL={result_url}")
    return result_url


def _get_required_query_value(query: dict, key: str) -> str:
    values = query.get(key)
    if not values or not values[0]:
        raise ValueError(f"Missing required query parameter: {key}")
    return values[0]


def _get_optional_query_value(query: dict, key: str) -> str | None:
    values = query.get(key)
    if not values or not values[0]:
        return None
    return values[0]


def _build_vss_login_url(url: str) -> str:
    parts = urlsplit(url)

    # Validate parsed components
    if not parts.hostname or not parts.scheme:
        logger.error(
            f"VSS URL parse failed: url={url}, scheme={parts.scheme}, hostname={parts.hostname}, path={parts.path}"
        )
        raise ValueError(
            f"VSS RealVideo URL is malformed. Expected: https://host:port/vss/apipage/realvideo.html?... Got: {url}"
        )

    # Check if hostname is accidentally a scheme (e.g., 'http' instead of 'vss.host')
    if parts.hostname in ("http", "https", "ftp"):
        logger.error(
            f"VSS URL hostname parsed as scheme: url={url}, hostname={parts.hostname}"
        )
        raise ValueError(
            f"VSS URL is malformed. Hostname should not be a scheme. Got: {url}"
        )

    port = parts.port or (443 if parts.scheme.lower() == "https" else 9966)
    login_url = f"{parts.scheme}://{parts.hostname}:{port}/vss/user/apiLogin.action"
    logger.debug(f"Built VSS login URL: {login_url}")
    return login_url


def _build_vss_query_gt_url(url: str) -> str:
    parts = urlsplit(url)

    # Validate parsed components
    if not parts.hostname or not parts.scheme:
        logger.error(
            f"VSS URL parse failed: url={url}, scheme={parts.scheme}, hostname={parts.hostname}, path={parts.path}"
        )
        raise ValueError(
            f"VSS RealVideo URL is malformed. Expected: https://host:port/vss/apipage/realvideo.html?... Got: {url}"
        )

    # Check if hostname is accidentally a scheme (e.g., 'http' instead of 'vss.host')
    if parts.hostname in ("http", "https", "ftp"):
        logger.error(
            f"VSS URL hostname parsed as scheme: url={url}, hostname={parts.hostname}"
        )
        raise ValueError(
            f"VSS URL is malformed. Hostname should not be a scheme. Got: {url}"
        )

    port = parts.port or (443 if parts.scheme.lower() == "https" else 9966)
    query_url = (
        f"{parts.scheme}://{parts.hostname}:{port}/vss/vehicle/queryGtOfDevice.action"
    )
    logger.debug(f"Built VSS device query URL: {query_url}")
    return query_url


def resolve_vss_stream_host(url: str, device_id: str) -> str:
    parts = urlsplit(url)
    if not parts.hostname or not parts.scheme:
        logger.error(
            f"resolve_vss_stream_host: URL parse failed: scheme={parts.scheme}, hostname={parts.hostname}, path={parts.path}"
        )
        raise ValueError("VSS RealVideo URL is missing host or scheme")

    query_gt_url = _build_vss_query_gt_url(url)
    logger.debug(
        f"resolve_vss_stream_host: Querying device {device_id} at {query_gt_url}"
    )

    try:
        response = requests.post(
            query_gt_url,
            json={"deviceNo": device_id},
            timeout=VSS_DEVICE_QUERY_TIMEOUT_SEC,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") or {}
        node_id = str(data.get("nodeID", ""))
        gt_address = str(data.get("gtAddress", "")).strip()
        logger.debug(
            f"resolve_vss_stream_host: Device {device_id} nodeID={node_id}, gtAddress={gt_address}"
        )

        if node_id != "0" and gt_address:
            result_host = gt_address.split(":", 1)[0]
            logger.info(
                f"resolve_vss_stream_host: Using resolved gtAddress host: {result_host}"
            )
            return result_host
    except Exception as exc:
        logger.warning(
            "Failed to resolve gtAddress for VSS device %s: %s", device_id, exc
        )

    logger.info(
        f"resolve_vss_stream_host: Falling back to original hostname: {parts.hostname}"
    )
    return parts.hostname


def _get_vss_credentials(url: str) -> tuple[str | None, str | None]:
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    username = _get_optional_query_value(query, "username") or os.getenv("VSS_USERNAME")
    password_md5 = _get_optional_query_value(query, "password") or os.getenv(
        "VSS_PASSWORD_MD5"
    )
    return username, password_md5


# Phrases long enough for safe substring matching
_VSS_ERROR_PHRASES = frozenset(
    {
        "login too frequently",
        "too many requests",
        "login failed",
        "invalid password",
        "invalid username",
        "access denied",
        "unauthorized",
        "forbidden",
        "expired",
    }
)
# Short words only checked via exact match (to avoid "ok" matching inside "token")
_VSS_ERROR_EXACT = frozenset(
    {
        "error",
        "fail",
        "failure",
        "false",
    }
)
# Words that indicate success (not valid as tokens, but not errors either)
_VSS_SUCCESS_EXACT = frozenset(
    {
        "success",
        "ok",
        "true",
    }
)


def _is_valid_token(value: str) -> bool:
    """Return True only if *value* looks like a genuine VSS session token."""
    if not value or len(value) > 200:
        return False
    # Real tokens are hex/alphanumeric strings; reject anything with spaces
    if " " in value:
        return False
    value_lower = value.lower()
    # Exact-match reject for short words (errors and success indicators are not valid tokens)
    if value_lower in _VSS_ERROR_EXACT or value_lower in _VSS_SUCCESS_EXACT:
        return False
    # Substring reject for longer phrases
    for phrase in _VSS_ERROR_PHRASES:
        if phrase in value_lower:
            return False
    return True


def _extract_token_from_payload(payload):
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped if _is_valid_token(stripped) else None

    if isinstance(payload, dict):
        for key, value in payload.items():
            if (
                key.lower() in {"token", "tokenid", "access_token"}
                and isinstance(value, str)
                and value.strip()
            ):
                candidate = value.strip()
                if _is_valid_token(candidate):
                    return candidate
                logger.warning(
                    "VSS login returned invalid token value for key '%s': %s",
                    key,
                    candidate,
                )
                return None
        for value in payload.values():
            token = _extract_token_from_payload(value)
            if token:
                return token

    if isinstance(payload, list):
        for item in payload:
            token = _extract_token_from_payload(item)
            if token:
                return token

    return None


VSS_LOGIN_MAX_RETRIES = int(os.getenv("VSS_LOGIN_MAX_RETRIES", "2"))
VSS_LOGIN_RETRY_DELAY_SEC = int(os.getenv("VSS_LOGIN_RETRY_DELAY_SEC", "30"))


def _check_vss_error_response(payload) -> str | None:
    """Return an error message if the VSS JSON response indicates failure, else None."""
    if not isinstance(payload, dict):
        return None

    # Success indicators for the msg/message field
    _SUCCESS_MSG = {"success", "ok"}

    # VSS APIs typically signal errors via result/code/status fields
    result = payload.get("result", payload.get("code", payload.get("status")))
    if result is not None and str(result).lower() not in {
        "0",
        "1",
        "200",
        "true",
        "ok",
        "success",
    }:
        msg = payload.get("msg", payload.get("message", payload.get("error", "")))
        msg_str = str(msg or result)
        # Don't flag as error if msg itself indicates success
        if msg_str.lower().strip() in _SUCCESS_MSG:
            return None
        return msg_str

    # Also check if the response body itself is an error message string
    msg = payload.get("msg", payload.get("message", ""))
    if isinstance(msg, str) and msg:
        msg_lower = msg.lower()
        # Skip if msg is a success indicator
        if msg_lower in _SUCCESS_MSG:
            return None
        if msg_lower in _VSS_ERROR_EXACT:
            return msg
        for phrase in _VSS_ERROR_PHRASES:
            if phrase in msg_lower:
                return msg
    return None


def _raise_vss_rate_limit_error(remaining: int, error_msg: str | None = None) -> None:
    detail = (
        f"VSS server đang giới hạn đăng nhập. Vui lòng đợi {remaining}s rồi thử lại."
    )
    if error_msg:
        detail = f"VSS server đang giới hạn đăng nhập ({error_msg}). Vui lòng đợi {remaining}s rồi thử lại."
    raise RuntimeError(detail)


def _is_vss_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "giới hạn đăng nhập" in message
        or "login too frequently" in message
        or "too many requests" in message
    )


def _should_force_vss_token_refresh(first_error: Exception) -> bool:
    if _is_vss_rate_limit_error(first_error):
        return False

    if isinstance(first_error, subprocess.TimeoutExpired):
        return False

    message = str(first_error).lower()
    if "timed out" in message or "timeout" in message:
        return False
    if "no video stream found in http-flv source" in message:
        return False
    if "unable to determine http-flv stream resolution" in message:
        return False

    auth_hints = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "expired",
        "invalid token",
        "invalid session",
        "access denied",
        "login failed",
    )
    return any(hint in message for hint in auth_hints)


def _should_fallback_to_mjpeg(first_error: Exception) -> bool:
    if _is_vss_rate_limit_error(first_error):
        return False

    if isinstance(first_error, subprocess.TimeoutExpired):
        return True

    message = str(first_error).lower()
    auth_hints = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "expired",
        "invalid token",
        "invalid session",
        "access denied",
        "login failed",
    )
    if any(hint in message for hint in auth_hints):
        return False

    probe_hints = (
        "timed out",
        "timeout",
        "ffprobe failed",
        "no video stream found in http-flv source",
        "unable to determine http-flv stream resolution",
        "ffprobe is unavailable",
    )
    return any(hint in message for hint in probe_hints)


def _mask_vss_token(token: str | None) -> str:
    if not token:
        return "(empty)"
    if len(token) <= 8:
        return token
    return f"{token[:4]}...{token[-4:]}"


def _get_force_login_rate_limit_fallback(
    token: str | None, cached_token: str | None
) -> tuple[str | None, str | None]:
    if token:
        return token, "URL"
    if cached_token:
        return cached_token, "cached"
    return None, None


def resolve_vss_token(url: str, force_login: bool = False) -> str:
    # --- Temporary override: skip all login logic when a fixed token is set ---
    fixed_token = os.getenv("VSS_FIXED_TOKEN", "").strip()
    if fixed_token:
        logger.info(
            "VSS_FIXED_TOKEN is set, skipping login and returning fixed token: %s",
            _mask_vss_token(fixed_token),
        )
        return fixed_token

    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    raw_token = _get_optional_query_value(query, "token")
    token = raw_token if (raw_token and _is_valid_token(raw_token)) else None
    if raw_token and not token:
        logger.warning(
            "URL token '%s' is not a valid session token, ignoring", raw_token
        )
    username, password_md5 = _get_vss_credentials(url)

    if not force_login:
        active_channel_token = _get_active_vss_channel_token(url)
        if active_channel_token:
            logger.info(
                "VSS token resolved from active channel session for %s (token=%s)",
                url,
                _mask_vss_token(active_channel_token),
            )
            return active_channel_token

        cached = _get_cached_vss_token_for_url(url, force_login=False)
        if cached:
            return cached

        if token:
            logger.info(
                "VSS token resolved directly from URL for %s (token=%s)",
                url,
                _mask_vss_token(token),
            )
            return token

    should_use_login = bool(username and password_md5) and (
        force_login or VSS_PREFER_LOGIN_TOKEN or not token
    )

    if token and not should_use_login:
        return token

    if not username or not password_md5:
        if token:
            return token
        raise ValueError(
            "Missing VSS token and login credentials. Set VSS_USERNAME and VSS_PASSWORD_MD5 or include token in RealVideo URL"
        )

    login_url = _build_vss_login_url(url)
    cache_key = (login_url, username)

    # Fast-path: cache check (always, even for force_login)
    cached = _get_cached_vss_token_for_url(url, force_login=force_login)
    if cached and not force_login:
        return cached

    # --- Rate-limit guard: if we recently got rate-limited, don't try login ---
    global _VSS_RATE_LIMITED_UNTIL, _VSS_LAST_LOGIN_TIME
    now = time.time()
    if now < _VSS_RATE_LIMITED_UNTIL:
        remaining = int(_VSS_RATE_LIMITED_UNTIL - now)
        if force_login:
            fallback, fallback_source = _get_force_login_rate_limit_fallback(
                token, cached
            )
            if fallback:
                logger.warning(
                    "VSS forced login skipped during rate-limit (%ds remaining), using %s token instead",
                    remaining,
                    fallback_source,
                )
                return fallback
            _raise_vss_rate_limit_error(remaining)
        # Use any available token (cached or URL) instead of hitting the API
        fallback = cached or token
        if fallback:
            logger.info(
                "VSS login skipped (rate-limited, %ds remaining), using %s token",
                remaining,
                "cached" if cached else "URL",
            )
            return fallback
        _raise_vss_rate_limit_error(remaining)

    # Acquire the global login lock so only one thread performs the actual
    # HTTP login at a time.
    with _VSS_LOGIN_LOCK:
        # Re-check cache under the login lock
        cached = _get_cached_vss_token_for_url(url, force_login=force_login)
        if cached and not force_login:
            logger.debug("VSS token cache hit after lock wait")
            return cached

        # Re-check rate-limit under the lock (another thread may have set it)
        now = time.time()
        if now < _VSS_RATE_LIMITED_UNTIL:
            remaining = int(_VSS_RATE_LIMITED_UNTIL - now)
            if force_login:
                fallback, fallback_source = _get_force_login_rate_limit_fallback(
                    token, cached
                )
                if fallback:
                    logger.warning(
                        "VSS forced login skipped after lock during rate-limit (%ds remaining), using %s token instead",
                        remaining,
                        fallback_source,
                    )
                    return fallback
                _raise_vss_rate_limit_error(remaining)
            fallback = cached or token
            if fallback:
                logger.info(
                    "VSS login skipped after lock (rate-limited, %ds remaining)",
                    remaining,
                )
                return fallback
            _raise_vss_rate_limit_error(remaining)

        # Enforce cooldown between consecutive login API calls
        elapsed = now - _VSS_LAST_LOGIN_TIME
        if elapsed < VSS_LOGIN_COOLDOWN_SEC:
            wait = VSS_LOGIN_COOLDOWN_SEC - elapsed
            logger.debug("VSS login cooldown: waiting %.1fs", wait)
            time.sleep(wait)

        last_error_msg = ""
        for attempt in range(1, VSS_LOGIN_MAX_RETRIES + 1):
            _VSS_LAST_LOGIN_TIME = time.time()
            try:
                response = requests.post(
                    login_url,
                    json={"username": username, "password": password_md5},
                    timeout=VSS_LOGIN_TIMEOUT_SEC,
                )
                response.raise_for_status()
            except Exception as exc:
                last_error_msg = str(exc)
                logger.warning("VSS login request failed: %s", exc)
                if token:
                    return token
                raise RuntimeError(f"VSS login request failed: {exc}")

            # --- Check for error response before extracting token ---
            token_from_json = None
            try:
                json_payload = response.json()
                error_msg = _check_vss_error_response(json_payload)
                if error_msg:
                    last_error_msg = error_msg
                    is_rate_limit = (
                        "too frequently" in error_msg.lower()
                        or "too many" in error_msg.lower()
                    )
                    if is_rate_limit:
                        # Set global rate-limit flag to prevent cascading retries
                        _VSS_RATE_LIMITED_UNTIL = (
                            time.time() + VSS_RATE_LIMIT_BACKOFF_SEC
                        )
                        logger.warning(
                            "VSS login rate-limited (%s), backing off for %ds",
                            error_msg,
                            int(VSS_RATE_LIMIT_BACKOFF_SEC),
                        )
                        forced_fallback, forced_fallback_source = (
                            _get_force_login_rate_limit_fallback(token, cached)
                            if force_login
                            else (None, None)
                        )
                        if forced_fallback:
                            logger.warning(
                                "VSS forced login rate-limited, using %s token instead of retrying",
                                forced_fallback_source,
                            )
                            return forced_fallback
                        if attempt < VSS_LOGIN_MAX_RETRIES:
                            delay = VSS_LOGIN_RETRY_DELAY_SEC
                            logger.warning(
                                "VSS login retry %d/%d in %ds",
                                attempt,
                                VSS_LOGIN_MAX_RETRIES,
                                delay,
                            )
                            time.sleep(delay)
                            continue
                        # Exhausted retries
                        if force_login:
                            _raise_vss_rate_limit_error(
                                int(VSS_RATE_LIMIT_BACKOFF_SEC), error_msg
                            )
                        fallback = cached or token
                        if fallback:
                            logger.warning(
                                "VSS login rate-limited after %d retries, using %s token",
                                attempt,
                                "cached" if cached else "URL",
                            )
                            return fallback
                        _raise_vss_rate_limit_error(
                            int(VSS_RATE_LIMIT_BACKOFF_SEC), error_msg
                        )
                    # Non-rate-limit error
                    if token:
                        logger.warning(
                            "VSS login error (%s), falling back to URL token", error_msg
                        )
                        return token
                    raise RuntimeError(f"VSS login failed: {error_msg}")
                token_from_json = _extract_token_from_payload(json_payload)
            except ValueError:
                pass

            # Validate that extracted value is a real token
            raw_fallback = response.text.strip()
            token_value = token_from_json or (
                raw_fallback if _is_valid_token(raw_fallback) else None
            )

            if not token_value:
                if token:
                    logger.warning(
                        "VSS login returned no valid token (body=%s), falling back to URL token",
                        raw_fallback[:120],
                    )
                    return token
                raise RuntimeError(
                    f"VSS login response is not a valid session token: {raw_fallback[:120]}"
                )

            # Success! Cache the token and clear rate-limit flag
            _VSS_RATE_LIMITED_UNTIL = 0.0
            with _TOKEN_CACHE_LOCK:
                _VSS_TOKEN_CACHE[cache_key] = (
                    token_value,
                    time.time() + VSS_TOKEN_CACHE_TTL_SEC,
                )
            logger.info(
                "VSS token cached (%s, token=%s, ttl=%ss)",
                f"login_url={login_url}, username={username}, force_login={force_login}",
                _mask_vss_token(token_value),
                VSS_TOKEN_CACHE_TTL_SEC,
            )
            return token_value

        # All retries exhausted
        fallback = cached or token
        if fallback:
            logger.warning(
                "VSS login retries exhausted (%s), using %s token",
                last_error_msg,
                "cached" if cached else "URL",
            )
            return fallback
        raise RuntimeError(
            f"VSS login failed after {VSS_LOGIN_MAX_RETRIES} retries: {last_error_msg}"
        )


def _parse_fps(rate_value: str) -> float:
    if not rate_value or rate_value == "0/0":
        return DEFAULT_HTTP_FLV_FPS

    try:
        fps = float(Fraction(rate_value))
    except (ValueError, ZeroDivisionError):
        return DEFAULT_HTTP_FLV_FPS

    if fps <= 0 or fps > 120:
        return DEFAULT_HTTP_FLV_FPS
    return fps


class HTTPFLVCapture:
    """Capture frames from a VSS HTTP-FLV source via a local FFmpeg subprocess."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._active_channel_owner_id = f"httpflv:{id(self)}"
        self._registered_active_channel = False
        self.stream_url = self._build_initial_stream_url(source_url)
        self._ffmpeg_path = _resolve_binary("ffmpeg", "FFMPEG_PATH")
        self._ffprobe_path = _resolve_binary("ffprobe", "FFPROBE_PATH")
        self._process = None
        self._stderr_thread = None
        self._stderr_stop = threading.Event()
        self._opened = False
        self._output_mode = "rawvideo"
        self._mjpeg_buffer = bytearray()
        self.width = 0
        self.height = 0
        self.fps = DEFAULT_HTTP_FLV_FPS
        self._video_codec: str | None = None
        self._tried_alternate_stream = False
        self._input_format = "live_flv"  # try enhanced FLV (HEVC) first

        if self._ffmpeg_path is None:
            error_msg = "HTTP-FLV capture requires ffmpeg to be installed. Set FFMPEG_PATH env var or install via: apt-get install ffmpeg"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        if self._ffprobe_path is None:
            logger.warning(
                "HTTP-FLV capture proceeding without ffprobe for %s; falling back to MJPEG pipe mode",
                self.stream_url,
            )

        try:
            self._initialize_stream()
            self._opened = self._process is not None and self._process.poll() is None
            self._register_active_channel_token_if_needed()
        except Exception as exc:
            # If the server itself is unreachable, skip ALL fallback chains.
            # There's no point trying token refresh, alternate streams, or MJPEG
            # when the server won't accept connections at all.
            if self._is_server_unreachable_error(exc):
                logger.error(
                    "Failed to initialize HTTP-FLV capture for %s: %s (server unreachable, skipping fallbacks)",
                    self.stream_url,
                    exc,
                )
                self.release()
                return
            if self._try_refresh_stream_url(exc):
                return
            if self._try_alternate_stream_variant(exc):
                return
            if self._try_start_with_mjpeg_fallback(exc):
                return
            logger.error(
                "Failed to initialize HTTP-FLV capture for %s: %s", self.stream_url, exc
            )
            self.release()

    @staticmethod
    def _is_server_unreachable_error(exc: Exception) -> bool:
        """Return True if the error indicates the server itself is unreachable.

        When the VSS server is down (connection refused, network unreachable,
        DNS failure, etc.) there is no point trying token refresh, alternate
        stream variants, or MJPEG fallback — they will all fail the same way.
        """
        msg = str(exc).lower()
        unreachable_hints = (
            "connection refused",
            "network is unreachable",
            "no route to host",
            "name or service not known",
            "nodename nor servname",
            "host is unreachable",
            "errno 111",  # ECONNREFUSED
            "errno 113",  # EHOSTUNREACH
            "errno 101",  # ENETUNREACH
        )
        return any(hint in msg for hint in unreachable_hints)

    def _build_initial_stream_url(self, source_url: str) -> str:
        active_channel_token = _get_active_vss_channel_token(source_url)
        if active_channel_token:
            logger.info(
                "HTTP-FLV capture reusing active VSS channel token for %s: %s",
                source_url,
                _mask_vss_token(active_channel_token),
            )
            realvideo_url = build_vss_realvideo_url(
                source_url,
                username=_get_vss_credentials(source_url)[0] or "",
                password_md5=_get_vss_credentials(source_url)[1] or "",
                device_id=_get_required_query_value(
                    parse_qs(urlsplit(source_url).query, keep_blank_values=True),
                    "deviceId",
                ),
                channel=_get_required_query_value(
                    parse_qs(urlsplit(source_url).query, keep_blank_values=True), "chs"
                ).split("_")[0],
                stream=_get_required_query_value(
                    parse_qs(urlsplit(source_url).query, keep_blank_values=True),
                    "stream",
                ),
                token=active_channel_token,
            )
            return build_vss_flv_url(realvideo_url, force_login=False)

        if is_vss_realvideo_url(source_url):
            logger.info(
                "HTTP-FLV capture found no active VSS channel peer; building stream URL with reusable token path for %s",
                source_url,
            )
        return build_vss_flv_url(source_url, force_login=False)

    def _register_active_channel_token_if_needed(self) -> None:
        if self._registered_active_channel:
            return
        if not is_vss_realvideo_url(self.source_url):
            return
        token, _device_id, _channel, _stream = _extract_http_flv_stream_parts(
            self.stream_url
        )
        _register_active_vss_channel_token(
            self.source_url, token, self._active_channel_owner_id
        )
        self._registered_active_channel = True

    def _try_refresh_stream_url(self, first_error: Exception) -> bool:
        if not is_vss_realvideo_url(self.source_url):
            return False

        username, password_md5 = _get_vss_credentials(self.source_url)
        if not username or not password_md5:
            return False

        if not _should_force_vss_token_refresh(first_error):
            logger.info(
                "Skipping forced VSS token refresh after non-auth capture failure for %s: %s",
                self.stream_url,
                first_error,
            )
            return False

        try:
            refreshed_stream_url = build_vss_flv_url(self.source_url, force_login=True)
            if refreshed_stream_url == self.stream_url:
                return False

            logger.info(
                "Retrying HTTP-FLV capture with refreshed VSS token after initial failure: %s",
                first_error,
            )
            self.stream_url = refreshed_stream_url
            self._initialize_stream()
            self._opened = self._process is not None and self._process.poll() is None
            self._register_active_channel_token_if_needed()
            return self._opened
        except Exception as retry_exc:
            logger.error(
                "Retry with refreshed VSS token failed for %s: %s",
                self.stream_url,
                retry_exc,
            )
            self.release()
            if _is_vss_rate_limit_error(retry_exc):
                raise retry_exc
            return False

    def _try_alternate_stream_variant(self, first_error: Exception) -> bool:
        if self._tried_alternate_stream:
            return False
        if not is_vss_realvideo_url(self.source_url):
            return False

        if isinstance(first_error, subprocess.TimeoutExpired):
            should_try = True
        else:
            message = str(first_error).lower()
            should_try = (
                "no video stream found in http-flv source" in message
                or "unable to determine http-flv stream resolution" in message
                or "timed out" in message
                or "timeout" in message
            )
        if not should_try:
            return False

        original_stream_url = self.stream_url
        try:
            query = parse_qs(urlsplit(self.source_url).query, keep_blank_values=True)
            current_stream = _get_required_query_value(query, "stream")
            alt_stream = "0" if current_stream == "1" else "1"
            channel = _get_required_query_value(query, "chs").split("_")[0]
            device_id = _get_required_query_value(query, "deviceId")
            username, password_md5 = _get_vss_credentials(self.source_url)

            if not username or not password_md5:
                return False

            token = None
            if is_http_flv_url(self.stream_url):
                try:
                    token, _dev, _ch, _st = _extract_http_flv_stream_parts(
                        self.stream_url
                    )
                except Exception:
                    token = None

            self._tried_alternate_stream = True
            realvideo_alt = build_vss_realvideo_url(
                self.source_url,
                username=username,
                password_md5=password_md5,
                device_id=device_id,
                channel=channel,
                stream=alt_stream,
                token=token,
            )
            alt_flv = build_vss_flv_url(realvideo_alt, force_login=False)

            logger.warning(
                "HTTP-FLV trying alternate stream=%s after probe failure on stream=%s for %s",
                alt_stream,
                current_stream,
                self.stream_url,
            )

            self.release()
            self.stream_url = alt_flv
            self._initialize_stream()
            self._opened = self._process is not None and self._process.poll() is None
            if self._opened:
                self._register_active_channel_token_if_needed()
            return self._opened
        except Exception as alt_exc:
            logger.warning(
                "HTTP-FLV alternate stream fallback failed for %s: %s",
                self.stream_url,
                alt_exc,
            )
            # Clean up any partially-started process and restore original stream URL
            # so subsequent fallbacks (MJPEG) use the correct URL
            self.release()
            self.stream_url = original_stream_url
            return False

    def _initialize_stream(self) -> None:
        self._output_mode = "rawvideo"
        self._mjpeg_buffer.clear()
        probe_result = self._probe_stream()
        if len(probe_result) == 3:
            # Backward-compatible with tests/mocks that return (w, h, fps).
            self.width, self.height, self.fps = probe_result
            self._video_codec = None
        else:
            self.width, self.height, self.fps, self._video_codec = probe_result
        self._start_process()

    def _try_start_with_mjpeg_fallback(self, first_error: Exception) -> bool:
        if not _should_fallback_to_mjpeg(first_error):
            return False

        logger.warning(
            "HTTP-FLV probe failed for %s (%s); falling back to MJPEG pipe mode",
            self.stream_url,
            first_error,
        )
        # Try both live_flv (HEVC-capable) and standard flv formats
        for fmt in ("live_flv", "flv"):
            self.release()
            self._output_mode = "mjpeg"
            self._mjpeg_buffer.clear()
            self.width = 0
            self.height = 0
            self.fps = DEFAULT_HTTP_FLV_FPS
            self._video_codec = None
            self._input_format = fmt
            logger.info(
                "HTTP-FLV MJPEG fallback: trying input format '%s' for %s",
                fmt,
                self.stream_url,
            )
            self._start_process()
            self._opened = self._process is not None and self._process.poll() is None
            if self._opened:
                # Give ffmpeg a moment to validate the stream
                import time

                time.sleep(0.5)
                self._opened = (
                    self._process is not None and self._process.poll() is None
                )
            if self._opened:
                logger.info(
                    "HTTP-FLV MJPEG fallback succeeded with input format '%s' for %s",
                    fmt,
                    self.stream_url,
                )
                self._register_active_channel_token_if_needed()
                return True
            logger.warning(
                "HTTP-FLV MJPEG fallback failed with input format '%s' for %s",
                fmt,
                self.stream_url,
            )
        return False

    def _probe_stream(self) -> tuple[int, int, float, str | None]:
        if self._ffprobe_path is None:
            raise RuntimeError("ffprobe is unavailable")

        # Try live_flv format first (supports HEVC), then fall back to standard flv
        last_error: Exception | None = None
        for fmt in ("live_flv", "flv"):
            try:
                result = self._run_ffprobe(fmt)
                self._input_format = fmt
                return result
            except RuntimeError as exc:
                last_error = exc
                logger.debug(
                    "HTTP-FLV ffprobe with format '%s' failed for %s: %s",
                    fmt,
                    self.stream_url,
                    exc,
                )
        raise last_error  # type: ignore[misc]

    def _run_ffprobe(self, input_format: str) -> tuple[int, int, float, str | None]:
        cmd = [
            self._ffprobe_path,
            "-v",
            "error",
            "-headers",
            "User-Agent: curl/8.5.0\r\n",
            "-f",
            input_format,
            "-rw_timeout",
            str(HTTP_FLV_RW_TIMEOUT_US),
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            self.stream_url,
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=HTTP_FLV_PROBE_TIMEOUT_SEC,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            raise RuntimeError(
                stderr or f"ffprobe failed for HTTP-FLV stream (format={input_format})"
            )

        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError(
                f"No video stream found in HTTP-FLV source (format={input_format})"
            )

        video_stream = streams[0]
        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Unable to determine HTTP-FLV stream resolution")

        fps = _parse_fps(
            video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
        )
        codec_name = (video_stream.get("codec_name") or "").strip().lower() or None
        return width, height, fps, codec_name

    def _start_process(self) -> None:
        assert self._ffmpeg_path is not None

        cmd = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-headers",
            "User-Agent: curl/8.5.0\r\n",
            "-f",
            self._input_format,
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-rw_timeout",
            str(HTTP_FLV_RW_TIMEOUT_US),
        ]
        cmd.extend(
            _build_ffmpeg_hwdecode_input_args(
                self._ffmpeg_path,
                self._video_codec,
                source_name="HTTP-FLV capture",
            )
        )
        cmd.extend(["-i", self.stream_url, "-an"])
        if self._output_mode == "mjpeg":
            cmd.extend(
                [
                    "-c:v",
                    "mjpeg",
                    "-q:v",
                    "5",
                    "-f",
                    "image2pipe",
                    "pipe:1",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:v",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-f",
                    "rawvideo",
                    "pipe:1",
                ]
            )
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=10**7,
        )
        self._stderr_stop.clear()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return

        while not self._stderr_stop.is_set():
            line = self._process.stderr.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="replace").strip()
            if message:
                logger.info("HTTP-FLV ffmpeg: %s", message)

    def isOpened(self) -> bool:
        return (
            self._opened and self._process is not None and self._process.poll() is None
        )

    def signals_connection_on_open(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened() or self._process is None or self._process.stdout is None:
            return False, None

        if self._output_mode == "mjpeg":
            return self._read_mjpeg_frame()

        return self._read_rawvideo_frame()

    def _read_rawvideo_frame(self) -> tuple[bool, np.ndarray | None]:
        expected_size = self.width * self.height * 3
        if expected_size <= 0:
            self._opened = False
            return False, None

        raw_buffer = bytearray(expected_size)
        raw_view = memoryview(raw_buffer)
        offset = 0

        while offset < expected_size:
            chunk = self._read_stdout_chunk(expected_size - offset)
            if chunk is None:
                return False, None
            if not chunk:
                self._opened = False
                return False, None
            raw_view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)

        frame = (
            np.frombuffer(raw_buffer, dtype=np.uint8)
            .reshape((self.height, self.width, 3))
            .copy()
        )
        return True, frame

    def _read_mjpeg_frame(self) -> tuple[bool, np.ndarray | None]:
        while (
            self.isOpened()
            and self._process is not None
            and self._process.stdout is not None
        ):
            start = self._mjpeg_buffer.find(b"\xff\xd8")
            if start != -1:
                end = self._mjpeg_buffer.find(b"\xff\xd9", start + 2)
                if end != -1:
                    jpeg_bytes = bytes(self._mjpeg_buffer[start : end + 2])
                    del self._mjpeg_buffer[: end + 2]
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        continue
                    self.height, self.width = frame.shape[:2]
                    return True, frame
                if start > 0:
                    del self._mjpeg_buffer[:start]

            chunk = self._read_stdout_chunk(HTTP_FLV_PIPE_READ_SIZE)
            if chunk is None:
                return False, None

            if not chunk:
                self._opened = False
                return False, None

            self._mjpeg_buffer.extend(chunk)
            if len(self._mjpeg_buffer) > HTTP_FLV_MJPEG_BUFFER_LIMIT:
                start = self._mjpeg_buffer.find(b"\xff\xd8")
                if start > 0:
                    del self._mjpeg_buffer[:start]
                if len(self._mjpeg_buffer) > HTTP_FLV_MJPEG_BUFFER_LIMIT:
                    del self._mjpeg_buffer[:-HTTP_FLV_MJPEG_BUFFER_LIMIT]

        self._opened = False
        return False, None

    def _read_stdout_chunk(self, size: int) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            self._opened = False
            return b""

        stdout = self._process.stdout
        if os.name != "nt":
            try:
                ready, _, _ = select.select([stdout], [], [], HTTP_FLV_READ_TIMEOUT_SEC)
            except (OSError, ValueError):
                ready = [stdout]
            if not ready:
                return None

        reader = getattr(stdout, "read1", None)
        try:
            if callable(reader):
                return reader(size)
            return stdout.read(size)
        except ValueError:
            # ffmpeg pipe may already be closed while capture thread is still reading.
            self._opened = False
            return b""
        except OSError:
            self._opened = False
            return b""

    def release(self) -> None:
        self._opened = False
        self._stderr_stop.set()
        if self._registered_active_channel:
            _unregister_active_vss_channel_token(
                self.source_url, self._active_channel_owner_id
            )
            self._registered_active_channel = False

        process = self._process
        self._process = None
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1)
        self._stderr_thread = None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, _prop_id: int, _value: float) -> bool:
        return False


# ---------------------------------------------------------------------------
# VSS capture backend selection
# ---------------------------------------------------------------------------
# Controls which transport is used for AI frame capture:
#   auto  – (default) quick-probe the FLV URL; if ffprobe finds a video
#           stream, use HTTPFLVCapture; otherwise fall back to WebSocket.
#           The probe uses a short 5 s timeout so fallback is fast.
#   flv   – HTTP-FLV only (requires ffprobe + ffmpeg, consistent when the
#           VSS gateway serves standard FLV/H.264)
#   ws    – Howen WebSocket only (proprietary protocol, handles HEVC natively)
VSS_CAPTURE_BACKEND = os.getenv("VSS_CAPTURE_BACKEND", "auto").strip().lower()

# Quick-probe timeout for auto mode (seconds).  This must be SHORT so that
# auto mode falls back to WebSocket quickly when FLV is unsupported.
_VSS_AUTO_PROBE_TIMEOUT = int(os.getenv("VSS_AUTO_PROBE_TIMEOUT", "8"))


def _quick_flv_probe(source_url: str) -> bool:
    """Run a fast ffprobe to check whether the FLV URL has a parseable video stream.

    Returns True if at least one video stream is found, False otherwise.
    This is intentionally lightweight — it does NOT trigger token refresh,
    alternate-stream fallback, or MJPEG retry.
    """
    ffprobe_path = _resolve_binary("ffprobe", "FFPROBE_PATH")
    if ffprobe_path is None:
        logger.debug("_quick_flv_probe: ffprobe not available, skipping FLV check")
        return False

    # Build the FLV URL from the source (RealVideo) URL
    try:
        flv_url = build_vss_flv_url(source_url, force_login=False)
    except Exception as exc:
        logger.debug("_quick_flv_probe: could not build FLV URL: %s", exc)
        return False

    for fmt in ("live_flv", "flv"):
        try:
            cmd = [
                ffprobe_path,
                "-v",
                "error",
                "-headers",
                "User-Agent: curl/8.5.0\r\n",
                "-f",
                fmt,
                "-rw_timeout",
                str(min(HTTP_FLV_RW_TIMEOUT_US, 5_000_000)),  # cap at 5 s
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "json",
                flv_url,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_VSS_AUTO_PROBE_TIMEOUT,
            )
            if result.returncode == 0:
                import json as _json

                payload = _json.loads(result.stdout or "{}")
                streams = payload.get("streams") or []
                if streams:
                    logger.info(
                        "_quick_flv_probe: FLV probe OK (format=%s, codec=%s) for %s",
                        fmt,
                        streams[0].get("codec_name", "?"),
                        flv_url,
                    )
                    return True
        except subprocess.TimeoutExpired:
            logger.debug(
                "_quick_flv_probe: probe timed out (format=%s) for %s", fmt, flv_url
            )
        except Exception as exc:
            logger.debug("_quick_flv_probe: probe failed (format=%s): %s", fmt, exc)

    logger.info("_quick_flv_probe: no video stream found via FLV for %s", source_url)
    return False


def create_vss_capture(source_url: str):
    """Create a capture backend for a VSS camera stream.

    The backend is chosen via the ``VSS_CAPTURE_BACKEND`` env var:
    - ``auto`` – quick-probe FLV; use it if probe succeeds, else WebSocket
    - ``flv``  – use HTTPFLVCapture (FFmpeg → pipe, consistent real-time)
    - ``ws``   – use HowenWebSocketCapture (handles HEVC natively)
    """
    backend = (os.getenv("VSS_CAPTURE_BACKEND", VSS_CAPTURE_BACKEND) or "auto").strip().lower()

    if backend == "flv":
        logger.info("VSS capture backend: HTTP-FLV (forced) for %s", source_url)
        return HTTPFLVCapture(source_url)

    if backend == "ws":
        logger.info("VSS capture backend: WebSocket (forced) for %s", source_url)
        return HowenWebSocketCapture(source_url)

    # auto: quick-probe FLV, fall back to WebSocket
    logger.info("VSS capture backend: auto — probing FLV for %s", source_url)
    if _quick_flv_probe(source_url):
        logger.info(
            "VSS capture backend: FLV probe succeeded, using HTTPFLVCapture for %s",
            source_url,
        )
        return HTTPFLVCapture(source_url)

    logger.info(
        "VSS capture backend: FLV probe failed, using WebSocket for %s", source_url
    )
    try:
        return HowenWebSocketCapture(source_url)
    except Exception as exc:
        logger.warning(
            "VSS capture backend: WebSocket failed for %s, falling back to HTTP-FLV: %s",
            source_url,
            exc,
        )
        return HTTPFLVCapture(source_url)


# ---------------------------------------------------------------------------
# Howen WebSocket capture backend
# ---------------------------------------------------------------------------

HOWEN_WS_CONNECT_TIMEOUT = int(os.getenv("HOWEN_WS_CONNECT_TIMEOUT", "15"))
HOWEN_WS_READ_TIMEOUT_SEC = float(os.getenv("HOWEN_WS_READ_TIMEOUT_SEC", "1.0"))
HOWEN_WS_FIRST_FRAME_TIMEOUT_SEC = float(
    os.getenv("HOWEN_WS_FIRST_FRAME_TIMEOUT_SEC", "30")
)
# After receiving the first WS video frame, how long to wait for ffmpeg to
# actually produce decoded output on its stdout pipe.  If ffmpeg produces
# nothing in this window the stream is likely encrypted or in an unsupported
# format, and we raise an error immediately rather than waiting ~30 s.
HOWEN_WS_DECODE_TIMEOUT_SEC = float(os.getenv("HOWEN_WS_DECODE_TIMEOUT_SEC", "12.0"))
HOWEN_WS_FRAME_HEADER_LEN = (
    12  # frame_type(2) + time_offset(2) + data_len(4) + timestamp(4)
)
HOWEN_WS_MSG_HEADER_LEN = 8  # magic(1) + ver(1) + action(2) + payload_len(4)
# Codec strategy for piping Howen NAL data through ffmpeg.
# Default is hevc because production Howen feeds currently decode reliably on
# HEVC only. Set HOWEN_WS_CODEC=auto to retry h264->hevc, or force h264/hevc.
HOWEN_WS_CODEC = os.getenv("HOWEN_WS_CODEC", "hevc").strip().lower()
HOWEN_WS_ACTION = os.getenv("HOWEN_WS_ACTION", "3000")
HOWEN_WS_VIDEO_FRAME_TYPES = {
    int(item.strip())
    for item in os.getenv("HOWEN_WS_VIDEO_FRAME_TYPES", "1,2,3").split(",")
    if item.strip().isdigit()
}


def _get_howen_codec_candidates() -> list[str]:
    if HOWEN_WS_CODEC in {"", "auto"}:
        return ["hevc", "h264"]
    if HOWEN_WS_CODEC in {"h264", "hevc"}:
        return [HOWEN_WS_CODEC]
    logger.warning(
        "HowenWSCapture: unsupported HOWEN_WS_CODEC=%s, using as-is", HOWEN_WS_CODEC
    )
    return [HOWEN_WS_CODEC]


def _build_howen_play_command(
    device_id: str,
    channel: str,
    stream: str = "1",
) -> bytes:
    """Build the binary Howen WebSocket play command.

    Protocol: 8-byte header + JSON payload.
    Header: [0x48, version=1, action_code=2 (LE uint16), json_length (LE uint32)]

    Uses action 3000 (live video) by default.  The sessionID is a
    millisecond-precision timestamp, matching the browser player behaviour.
    """
    session_id = str(int(time.time() * 1000))

    json_payload = json.dumps(
        {
            "action": HOWEN_WS_ACTION,
            "payload": {
                "sessionID": session_id,
                "deviceID": device_id,
                "channel": channel,
                "stream": stream,
                "bufferTimeMs": "2000",
            },
        }
    )
    json_bytes = json_payload.encode("utf-8")
    header = struct.pack("<BBHI", 0x48, 1, 2, len(json_bytes))
    return header + json_bytes


def _prepare_howen_video_payload(nal_data: bytes) -> bytes:
    if nal_data.startswith(b"\x00\x00\x00\x01"):
        return nal_data
    if nal_data.startswith(b"\x00\x00\x01"):
        return b"\x00" + nal_data
    if nal_data.startswith(b"\x00\x01"):
        return b"\x00\x00" + nal_data
    return nal_data


class HowenWebSocketCapture:
    """Capture frames from a VSS camera via the Howen WebSocket protocol.

    Connects to ws://host:port/stream, sends a play command, receives
    proprietary media frames, extracts H.264 NAL units, and pipes them
    through ffmpeg for decoding into raw BGR frames.
    """

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._ffmpeg_path = _resolve_binary("ffmpeg", "FFMPEG_PATH")
        self._process: subprocess.Popen | None = None
        self._ws = None
        self._ws_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._opened = False
        self.width = 0
        self.height = 0
        self.fps = DEFAULT_HTTP_FLV_FPS
        self._ws_params: dict | None = None
        self._first_frame_event = threading.Event()
        self._init_error: str | None = None
        self._active_codec = "h264"
        self._active_stream_type = "1"

        if self._ffmpeg_path is None:
            raise RuntimeError(
                "Howen WebSocket capture requires ffmpeg. "
                "Set FFMPEG_PATH env var or install via: apt-get install ffmpeg"
            )

        try:
            self._ws_params = build_vss_ws_params(source_url)
            logger.info(
                "HowenWSCapture: connecting to %s for device %s",
                self._ws_params["ws_url"],
                self._ws_params["device_id"],
            )
            self._start()
        except Exception as exc:
            logger.error("HowenWSCapture: init failed: %s", exc)
            self.release()
            raise

    def _start(self) -> None:
        """Connect WebSocket, start ffmpeg, begin receiving frames."""
        try:
            import websocket as ws_lib
        except ImportError:
            raise RuntimeError(
                "websocket-client package is required. Install via: pip install websocket-client"
            )

        params = self._ws_params
        assert params is not None
        codec_candidates = _get_howen_codec_candidates()
        stream = params.get("stream", "1")
        last_error: Exception | None = None

        for attempt_idx, codec in enumerate(codec_candidates, 1):
            self._cleanup_attempt()
            self._stop_event.clear()
            self._first_frame_event = threading.Event()
            self._init_error = None
            self.width = 0
            self.height = 0
            self._active_codec = codec
            self._active_stream_type = stream

            try:
                logger.info(
                    "HowenWSCapture: starting ffmpeg codec=%s (attempt %d/%d)",
                    codec,
                    attempt_idx,
                    len(codec_candidates),
                )
                assert self._ffmpeg_path is not None
                ffmpeg_cmd = [
                    self._ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "info",
                    "-f",
                    codec,
                ]
                ffmpeg_cmd.extend(
                    _build_ffmpeg_hwdecode_input_args(
                        self._ffmpeg_path,
                        codec,
                        source_name="HowenWSCapture",
                    )
                )
                ffmpeg_cmd.extend(
                    [
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "-f",
                        "rawvideo",
                        "pipe:1",
                    ]
                )
                self._process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**7,
                )

                self._stderr_thread = threading.Thread(
                    target=self._drain_ffmpeg_stderr, daemon=True
                )
                self._stderr_thread.start()

                self._ws = ws_lib.WebSocket()
                self._ws.connect(params["ws_url"], timeout=HOWEN_WS_CONNECT_TIMEOUT)
                self._ws.settimeout(HOWEN_WS_READ_TIMEOUT_SEC)
                logger.info(
                    "HowenWSCapture: WebSocket connected to %s", params["ws_url"]
                )

                play_cmd = _build_howen_play_command(
                    params["device_id"],
                    params["channel"],
                    stream,
                )
                self._ws.send(play_cmd, opcode=0x2)
                logger.info(
                    "HowenWSCapture: play command sent (%d bytes) device=%s ch=%s stream=%s",
                    len(play_cmd),
                    params["device_id"],
                    params["channel"],
                    stream,
                )

                self._ws_thread = threading.Thread(
                    target=self._ws_recv_loop, daemon=True
                )
                self._ws_thread.start()

                if not self._first_frame_event.wait(
                    timeout=HOWEN_WS_FIRST_FRAME_TIMEOUT_SEC
                ):
                    if self._init_error:
                        raise RuntimeError(f"HowenWSCapture: {self._init_error}")
                    raise RuntimeError(
                        f"HowenWSCapture: no video frames within {HOWEN_WS_FIRST_FRAME_TIMEOUT_SEC}s "
                        f"(codec={codec})"
                    )

                if self.width <= 0 or self.height <= 0:
                    raise RuntimeError("HowenWSCapture: could not determine resolution")

                # ffmpeg may receive metadata NALs (type 50) before actual video.
                # Don't gate on select() decode timeout — let _capture_frames
                # handle retry via its consecutive_failures mechanism.

                self._opened = True
                logger.info(
                    "HowenWSCapture: streaming %dx%d codec=%s",
                    self.width,
                    self.height,
                    codec,
                )
                return

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "HowenWSCapture: attempt %d/%d (codec=%s) failed: %s",
                    attempt_idx,
                    len(codec_candidates),
                    codec,
                    exc,
                )
                self._cleanup_attempt()
                if attempt_idx < len(codec_candidates):
                    continue
                raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("HowenWSCapture: failed to initialize stream")

    def _cleanup_attempt(self) -> None:
        self._opened = False
        self._stop_event.set()

        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

        process = self._process
        self._process = None
        if process is not None:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass

        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2)
        self._ws_thread = None

        if (
            getattr(self, "_stderr_thread", None) is not None
            and self._stderr_thread.is_alive()
        ):
            self._stderr_thread.join(timeout=1)
        self._stderr_thread = None

    def _ws_recv_loop(self) -> None:
        """Receive WebSocket messages and pipe video data to ffmpeg stdin."""
        try:
            from websocket import WebSocketTimeoutException
        except Exception:  # pragma: no cover

            class WebSocketTimeoutException(Exception):
                pass

        frames_received = 0
        audio_frames = 0
        msgs_received = 0
        seen_frame_types = set()
        try:
            while not self._stop_event.is_set():
                try:
                    opcode, data = self._ws.recv_data(control_frame=True)
                except WebSocketTimeoutException:
                    continue
                except Exception as recv_err:
                    if not self._stop_event.is_set():
                        logger.warning("HowenWSCapture: recv_data error: %s", recv_err)
                    break

                msgs_received += 1
                if msgs_received <= 5:
                    logger.info(
                        "HowenWSCapture: msg #%d opcode=%d len=%d",
                        msgs_received,
                        opcode,
                        len(data),
                    )

                if opcode == 0x8:  # close
                    logger.warning("HowenWSCapture: WebSocket closed by server")
                    break
                if opcode == 0x9:  # ping
                    self._ws.pong(data)
                    continue
                if opcode not in (0x1, 0x2):
                    continue
                if len(data) < HOWEN_WS_MSG_HEADER_LEN:
                    continue

                magic = data[0]
                if magic != 0x48:
                    continue

                action_code = struct.unpack_from("<H", data, 2)[0]

                if action_code == 2:
                    # Session confirmation
                    try:
                        json_str = (
                            data[HOWEN_WS_MSG_HEADER_LEN:]
                            .decode("utf-8")
                            .rstrip("\x00")
                        )
                        resp = json.loads(json_str)
                        logger.info("HowenWSCapture: session response: %s", resp)
                        if int(resp.get("error", -1)) != 0:
                            self._init_error = f"Session rejected: {resp}"
                            self._first_frame_event.set()
                            return
                    except Exception as e:
                        logger.warning("HowenWSCapture: session parse error: %s", e)

                elif action_code == 1000:
                    # Media frame
                    media_data = data[HOWEN_WS_MSG_HEADER_LEN:]
                    if len(media_data) < HOWEN_WS_FRAME_HEADER_LEN:
                        continue

                    frame_type = struct.unpack_from("<H", media_data, 0)[0]
                    seen_frame_types.add(frame_type)

                    if frame_type not in HOWEN_WS_VIDEO_FRAME_TYPES:
                        audio_frames += 1
                        continue

                    nal_data = _prepare_howen_video_payload(
                        media_data[HOWEN_WS_FRAME_HEADER_LEN:]
                    )
                    if not nal_data:
                        continue

                    frames_received += 1
                    if frames_received <= 5:
                        logger.info(
                            "HowenWSCapture: NAL #%d ftype=%d size=%d start=%s",
                            frames_received,
                            frame_type,
                            len(nal_data),
                            nal_data[:16].hex(),
                        )

                    # Pipe ALL video frames to ffmpeg — let it handle NAL parsing
                    if self._process and self._process.stdin:
                        try:
                            self._process.stdin.write(nal_data)
                            self._process.stdin.flush()
                        except (BrokenPipeError, OSError):
                            logger.warning("HowenWSCapture: ffmpeg stdin broken")
                            break

                    if frames_received == 1:
                        logger.info(
                            "HowenWSCapture: first video NAL received, waiting for ffmpeg to detect resolution..."
                        )

                        # Don't set _first_frame_event here — let _drain_ffmpeg_stderr
                        # detect the real resolution from ffmpeg's "Video: ... WxH" output.
                        # Schedule a fallback in case ffmpeg never logs resolution.
                        def _resolution_fallback():
                            import time as _t

                            _t.sleep(8)
                            if not self._first_frame_event.is_set():
                                if self.width <= 0:
                                    self.width = 1920
                                if self.height <= 0:
                                    self.height = 1080
                                logger.warning(
                                    "HowenWSCapture: resolution detection timed out, using default %dx%d",
                                    self.width,
                                    self.height,
                                )
                                self._first_frame_event.set()

                        threading.Thread(
                            target=_resolution_fallback, daemon=True
                        ).start()

        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("HowenWSCapture: recv error: %s", exc)
        finally:
            logger.info(
                "HowenWSCapture: recv loop ended (video=%d, audio=%d, msgs=%d)",
                frames_received,
                audio_frames,
                msgs_received,
            )
            if not self._first_frame_event.is_set():
                self._first_frame_event.set()

    def _drain_ffmpeg_stderr(self) -> None:
        if self._process is None or self._process.stderr is None:
            return
        while not self._stop_event.is_set():
            line = self._process.stderr.readline()
            if not line:
                break
            message = line.decode("utf-8", errors="replace").strip()
            if not message:
                continue

            # Parse resolution from "Stream #0:0: Video: hevc ... 1920x1080" line
            if "Video:" in message and "x" in message:
                try:
                    parts = message.split()
                    for p in parts:
                        if "x" in p and p.replace("x", "").replace(",", "").isdigit():
                            dims = p.replace(",", "").split("x")
                            if len(dims) == 2:
                                w, h = int(dims[0]), int(dims[1])
                                if 100 < w < 8000 and 100 < h < 8000:
                                    self.width = w
                                    self.height = h
                                    logger.info(
                                        "HowenWSCapture: detected resolution %dx%d",
                                        w,
                                        h,
                                    )
                                    self._first_frame_event.set()
                except Exception:
                    pass

            # Only log warnings/errors, skip info-level noise
            is_warning = any(
                kw in message.lower()
                for kw in ("warn", "error", "invalid", "fail", "skip")
            )
            if is_warning or "Video:" in message:
                logger.warning("HowenWS ffmpeg: %s", message)

    def isOpened(self) -> bool:
        return (
            self._opened and self._process is not None and self._process.poll() is None
        )

    def signals_connection_on_open(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened() or self._process is None or self._process.stdout is None:
            return False, None

        expected_size = self.width * self.height * 3
        if expected_size <= 0:
            self._opened = False
            return False, None

        raw_buffer = bytearray(expected_size)
        raw_view = memoryview(raw_buffer)
        offset = 0

        while offset < expected_size:
            chunk = self._read_stdout_chunk(expected_size - offset)
            if chunk is None:
                return False, None
            if not chunk:
                self._opened = False
                return False, None
            raw_view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)

        frame = (
            np.frombuffer(raw_buffer, dtype=np.uint8)
            .reshape((self.height, self.width, 3))
            .copy()
        )
        return True, frame

    def _read_stdout_chunk(self, size: int) -> bytes | None:
        if self._process is None or self._process.stdout is None:
            self._opened = False
            return b""

        stdout = self._process.stdout
        if os.name != "nt":
            try:
                ready, _, _ = select.select([stdout], [], [], HOWEN_WS_READ_TIMEOUT_SEC)
            except (OSError, ValueError):
                ready = [stdout]
            if not ready:
                return None

        reader = getattr(stdout, "read1", None)
        try:
            if callable(reader):
                return reader(size)
            return stdout.read(size)
        except (ValueError, OSError):
            self._opened = False
            return b""

    def release(self) -> None:
        self._cleanup_attempt()

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0

    def set(self, _prop_id: int, _value: float) -> bool:
        return False
