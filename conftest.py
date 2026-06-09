"""
Pytest configuration.

This repository contains a few interactive/demo scripts prefixed with `test_`
(`test_stream.py`, `test_websocket.py`) that are not intended to be collected as
unit tests. They run network/GUI code at import time and can fail in CI or
headless environments.
"""

collect_ignore = [
    "test_stream.py",
    "test_websocket.py",
    "test_fire_smoke_pt.py",
    "test_live_view.py",
    "test_petrolimex_video.py",
    "test_smoke_fire_only.py",
    "test_smoke_fire_stream.py",
]



