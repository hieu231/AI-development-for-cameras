"""FastAPI Server"""

import logging
import os
import asyncio
import socket
import subprocess
import sys
from contextlib import asynccontextmanager

# Configure logging so module-level loggers (e.g. thread_manager's graceful
# startup progress) actually reach stdout / docker logs. Without this, the
# root logger defaults to WARNING — which made the graceful camera bring-up
# look like it was never running.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.exc import OperationalError
from src.database.get_db import engine, Base
from src.configs.database_config import config as db_config
from src.api import camera
from src.api import location
from src.api import camera_spec
from src.api import ai_model
from src.api import event
from src.api import performance
from src.api import parameters
from src.api import system
from src.api import websocket
from src.api import stream
from src.api import webrtc as webrtc_api
from src.api import vss as vss_api
from src.core.mqtt_bridge import mqtt_bridge
from src.core.websocket_manager import websocket_manager
from src.utils.evidence_path import resolve_writable_evidence_base_path

# Import models để SQLAlchemy nhận biết các bảng
from src.models import camera as camera_model
from src.models import location as location_model
from src.models import camera_spec as camera_spec_model
from src.models import ai_model as ai_model_model
from src.models import camera_model as camera_model_association


# Conditionally import face recognition API if enabled
enable_face_recognition = os.getenv("ENABLE_FACE_RECOGNITION", "true").lower() == "true"
if enable_face_recognition:
    from src.api import face_recognition
    from src.models import face_profile as face_profile_model
# Evidence image base directory (can be configured via env)
EVIDENCE_IMAGE_DIR = resolve_writable_evidence_base_path()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler – replaces deprecated @app.on_event('startup').
    Registers the main event loop for websocket_manager so background threads
    can use asyncio.run_coroutine_threadsafe safely.

    Also triggers a graceful bring-up of every camera whose ``status=True``
    in the database, so a service restart never leaves an online camera
    dark. The bring-up runs in a background thread because it can take
    tens of seconds on a 20-camera edge node (RTSP handshake timeouts,
    TensorRT engine deserialization), and we want /health to go green
    immediately instead of blocking behind the last camera's ffprobe.
    """
    websocket_manager.register_event_loop(asyncio.get_running_loop())
    mqtt_bridge.start()

    # Graceful camera startup — lazy import so this hook stays decoupled
    # from thread_manager's initialization order.
    if os.getenv("GRACEFUL_CAMERA_STARTUP", "true").lower() == "true":
        from src.core.thread_manager import thread_manager as _tm

        async def _graceful_start() -> None:
            try:
                summary = await asyncio.to_thread(_tm.load_all_active_cameras)
                if summary.get("failed"):
                    for cid, reason in summary.get("failures", []):
                        logger.warning(
                            "Graceful startup: camera %s did not come back up: %s",
                            cid,
                            reason,
                        )
            except Exception:
                logger.exception("Graceful camera startup failed")

        asyncio.create_task(_graceful_start())

    try:
        yield
    finally:
        mqtt_bridge.stop()


# Tạo FastAPI app
app = FastAPI(
    title="Petrolimex AI Backend API",
    description="API cho hệ thống quản lý camera và AI models",
    version="1.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def initialize_database() -> None:
    """Create tables at startup and surface DB config issues clearly."""
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        raise RuntimeError(
            "Database connection failed for "
            f"{db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}. "
            "Check DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, and DB_NAME."
        ) from exc


# Tạo bảng database khi khởi động
initialize_database()

# Include API routers
app.include_router(camera.router, prefix="/api")
app.include_router(location.router, prefix="/api")
app.include_router(camera_spec.router, prefix="/api")
app.include_router(ai_model.router, prefix="/api")
app.include_router(event.router, prefix="/api")
app.include_router(performance.router, prefix="/api")
app.include_router(parameters.router)
app.include_router(system.router, prefix="/api")
app.include_router(stream.router)  # Video streaming endpoints
app.include_router(webrtc_api.router)  # WebRTC signaling endpoints
app.include_router(vss_api.router, prefix="/api")  # VSS camera stream endpoints
app.include_router(websocket.router)  # WebSocket endpoints (no /api prefix)

# Include face recognition router if enabled
if enable_face_recognition:
    app.include_router(face_recognition.router)

# Serve evidence images as static files
app.mount(
    "/evidence_image",
    StaticFiles(directory=str(EVIDENCE_IMAGE_DIR), check_dir=False),
    name="evidence_image",
)

# Backward-compatible alias for clients that still request media-prefixed evidence URLs.
app.mount(
    "/api/media/evidence_image",
    StaticFiles(directory=str(EVIDENCE_IMAGE_DIR), check_dir=False),
    name="api_media_evidence_image",
)


@app.get("/api/media/evidence_image/{relative_path:path}")
def get_evidence_image_via_api_media(relative_path: str):
    """Backward-compatible media endpoint for evidence images."""
    base_dir = EVIDENCE_IMAGE_DIR.resolve()
    candidate = (base_dir / relative_path).resolve()

    # Prevent path traversal outside the evidence directory.
    if base_dir not in candidate.parents and candidate != base_dir:
        raise HTTPException(status_code=404, detail="Not Found")

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Not Found")

    return FileResponse(str(candidate))


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Petrolimex AI Backend API is running!",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/help")
def get_help():
    """Help endpoint"""
    return {
        "message": "Petrolimex AI Backend API is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "/": "Root endpoint",
            "/health": "Health check endpoint",
            "/help": "Help endpoint",
            "/api/cameras/help": "Camera API help",
            "/api/locations/help": "Location API help",
            "/api/ai_models/help": "AI Model API help",
            "/api/events/help": "Event API help",
            "/api/performance/help": "Performance API help",
            "/api/system/summary": "System summary",
            "/ws/events": "WebSocket for real-time events (all cameras)",
            "/ws/events/{camera_id}": "WebSocket for real-time events (specific camera)",
            "/ws/stats": "WebSocket connection statistics",
        },
        "websocket_demo": "websocket_demo.html",
    }


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port: int) -> bool:
    """Kill any process using the specified port"""
    try:
        # Try using lsof first
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"Killed process {pid} on port {port}")
                    except subprocess.CalledProcessError:
                        pass
            return True

        # Fallback: try using fuser
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"Freed port {port} using fuser")
            return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    return False


if __name__ == "__main__":
    import uvicorn

    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8668"))
    host = os.getenv("HOST", "0.0.0.0")

    # Check if port is in use and try to free it
    if is_port_in_use(port):
        print(f"Port {port} is already in use. Attempting to free it...")
        if kill_process_on_port(port):
            # Wait a moment for the port to be released
            import time

            time.sleep(1)
            if is_port_in_use(port):
                print(
                    f"ERROR: Port {port} is still in use after attempting to free it."
                )
                print(
                    f"Please manually kill the process using: lsof -ti:{port} | xargs kill -9"
                )
                sys.exit(1)
        else:
            print(f"ERROR: Port {port} is in use and could not be freed automatically.")
            print(
                f"Please manually kill the process using: lsof -ti:{port} | xargs kill -9"
            )
            sys.exit(1)

    # Hot-reload is DEV-ONLY. In production on the Jetson (the target for the
    # Docker image) the watcher process aggressively re-imports modules on any
    # .py touch, which:
    #   - kills camera capture threads mid-read while FFmpeg is still holding
    #     the RTSP socket (leaking VideoCapture handles and CUDA contexts)
    #   - double-loads YOLO weights, doubling VRAM usage
    #   - causes the "sometimes it crashes completely" symptom we saw in logs
    #
    # Control via RELOAD=true|false (default: false). For local dev, run with
    # RELOAD=true python -m src.server
    reload_enabled = os.getenv("RELOAD", "false").lower() == "true"

    if reload_enabled:
        uvicorn.run(
            "src.server:app",
            host=host,
            port=port,
            reload=True,
            reload_excludes=[
                "*.pyc", "__pycache__", "*.ipynb", "*.md", ".git", ".venv",
                "*.log", "*.db", "*.sqlite", "test_*", "src/test/*",
                "src/core/__pycache__/*", "src/models/__pycache__/*",
                "src/api/__pycache__/*", "src/ai_models/__pycache__/*",
                "*.pt", "*.weights", "*.onnx",
            ],
            reload_includes=["*.py"],
            reload_dirs=[
                "src/api", "src/core", "src/models", "src/ai_models",
                "src/face_recognition",
            ],
        )
    else:
        uvicorn.run(
            "src.server:app",
            host=host,
            port=port,
            reload=False,
            workers=int(os.getenv("UVICORN_WORKERS", "1")),
        )
