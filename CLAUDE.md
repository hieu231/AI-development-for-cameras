# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Petrolimex AI Backend** is a multi-camera AI detection system for Petrolimex gas stations running on NVIDIA Jetson edge devices. It performs real-time video surveillance using 10+ specialized AI models to detect violations and anomalies (helmet violations, oil spills, smoke/fire, license plates, faces, etc.), stores evidence images, and provides a comprehensive REST API for management.

**Tech Stack**: FastAPI + PyTorch + Ultralytics YOLO + OpenCV + PostgreSQL (pgvector) + FaceNet

**Target Platform**: NVIDIA Jetson with CUDA 12.8

## Development Commands

### Setup & Installation
```bash
# Install dependencies (requires Python 3.12.6 exactly)
uv sync

# Setup environment variables
cp .env.example .env
# Edit .env with database credentials and configuration
```

### Running the Application
```bash
# Start the FastAPI server (MAIN APPLICATION)
python -m src.server
# API docs: http://localhost:8668/docs
# Server runs on port 8668 with hot-reload enabled

# Test camera processing and AI inference
python -m src.main_run

# Test RTSP/HTTP stream connectivity
python test_stream.py
```

### Database & Model Management
```bash
# Sync AI models to database (required on first setup)
python scripts/sync_models_to_db.py --sync

# List models in database
python scripts/sync_models_to_db.py --list

# Update model parameters
python scripts/update_model_parameters.py --model-type helmet_detection --conf 0.7 --iou 0.3

# Delete all models from database
python scripts/delete_all_models.py
```

### Testing
```bash
# Run tests
pytest

# Test specific stream
python test_stream.py  # Interactive prompt for stream URL

# Test WebSocket real-time event broadcasting
python test_websocket.py  # Listen to all events
python test_websocket.py --camera-id <camera-uuid>  # Listen to specific camera
```

## Architecture Overview

### System Architecture

The system follows a **layered architecture with event-driven processing**:

```
FastAPI REST API (Port 8668)
        ↓
API Route Handlers (9 routers, 50+ endpoints)
        ↓
ThreadManager (Singleton orchestrator)
        ↓
Per-Camera: CameraThread (capture) + SingleThreadProcessor (AI inference)
        ↓
ModelFactory → AI Models (10+ types) → ObjectTracker
        ↓
EventStorageService → PostgreSQL + File System
```

### Key Components

**ThreadManager** ([src/core/thread_manager.py](src/core/thread_manager.py))
- Singleton orchestrator managing all camera processing lifecycle
- Starts/stops camera threads dynamically via API calls
- Maintains camera_id → (CameraThread, SingleThreadProcessor) mapping
- Thread-safe operations with proper cleanup

**CameraThread** ([src/core/thread_manager.py](src/core/thread_manager.py))
- Captures frames from RTSP/HTTP/HTTPS/WebSocket streams using OpenCV
- Automatic reconnection with exponential backoff (max 5 retries)
- FPS limiting (default 5 FPS, configurable via `FPS_LIMIT` env var)
- Pushes frames to queue for processing thread
- Auto-detects stream type and sets appropriate OpenCV backend

**SingleThreadProcessor** ([src/core/camera_single_thread.py](src/core/camera_single_thread.py))
- Consumes frames from queue and runs AI inference pipeline
- Loads AI models dynamically via ModelFactory based on camera configuration
- Runs all assigned models sequentially on each frame
- Delegates event storage to EventStorageService
- Frame preprocessing and ROI handling

**ModelFactory** ([src/core/model_factory.py](src/core/model_factory.py))
- Factory pattern for dynamic AI model instantiation
- Maps `model_type` string → concrete model class
- Validates model parameters before instantiation
- Handles model weight loading and device assignment

**AI Models** ([src/ai_models/](src/ai_models/))
- All models inherit from `BaseModel` ([src/ai_models/base_model.py](src/ai_models/base_model.py))
- **Standardized interface**: `process_frame(frame) → DetectionResult`
- **DetectionResult** format: `{frame: np.ndarray | None, event: bool, metadata: dict}`
  - `event=True` triggers database save
  - `metadata` contains detection details (objects, confidence, tracking_ids, etc.)
- **Object Tracking**: Most models use tracking with 30-minute reset window to reduce false positives
- Auto-detects best device (CUDA > MPS > CPU)

**Available Model Types**:
- `helmet_detection` - Helmet and vest violation detection
- `oil_spill` - Oil spill detection
- `smoke_fire` - Smoke and fire detection
- `oil_dumping` - Illegal oil dumping detection
- `people_control` - Crowd density and people counting
- `alpr` - License plate recognition (GPU-accelerated)
- `face_recognition` - Face detection and biometric matching
- `yolov8` - General YOLOv8 object detection
- `yolov11` - General YOLOv11 object detection

**EventStorageService** ([src/core/event_storage_service.py](src/core/event_storage_service.py))
- Saves events to PostgreSQL database
- Stores evidence images with structured paths: `evidence_image/{YYYY-MM-DD}/{camera-uuid}/{HH-MM-SS.mmm.jpg}`
- Singleton pattern
- Handles database transactions and file I/O

**PerformanceMonitor** ([src/core/performance_monitor.py](src/core/performance_monitor.py))
- Background daemon thread collecting system metrics every 10 minutes
- Tracks CPU, memory, GPU usage, camera FPS, model inference times
- Stores metrics in `performance` table

### Database Models

Located in [src/models/](src/models/):
- `Camera` - Camera configuration (UUID, name, RTSP URL, location, status)
- `Location` - Physical location hierarchy
- `AiModel` - AI model definitions (name, version, model_path, parameters)
- `CameraModel` - Many-to-many junction table (camera ↔ AI models)
- `Event` - Detected events with evidence image paths and metadata
- `FaceProfile` - Face recognition profiles with pgvector embeddings
- `Performance` - System performance metrics
- `CameraSpec` - Camera specifications and configuration
- `Parameters` - System parameters

All models use:
- `UUIDMixin` - UUID primary keys
- `TimestampMixin` - Automatic `created_at` and `updated_at` timestamps

### API Structure

Located in [src/api/](src/api/), all routes prefixed with `/api`:

- `/api/cameras/*` - Camera CRUD, start/stop processing, status
- `/api/ai_models/*` - AI model management, camera-model assignments
- `/api/events/*` - Event queries, filtering, export
- `/api/locations/*` - Location hierarchy management
- `/api/camera_specs/*` - Camera specifications
- `/api/performance/*` - System metrics and monitoring
- `/api/face_recognition/*` - Face profile management, registration, matching
- `/api/system/*` - System summary and health
- `/api/stream/camera/{id}/mjpeg` - MJPEG video stream with annotated frames
- `/api/stream/camera/{id}/status` - Frame buffer online/offline status
- `/api/webrtc/offer` - WebRTC SDP offer/answer exchange (POST)
- `/api/webrtc/connections` - List/close active WebRTC connections
- `/api/webrtc/status` - WebRTC subsystem health
- `/parameters/*` - System parameters (no /api prefix)
- `/ws/events` - WebSocket for real-time event broadcasting (all cameras)
- `/ws/events/{camera_id}` - WebSocket for real-time event broadcasting (specific camera)
- `/ws/stats` - WebSocket connection statistics

Interactive API documentation available at: http://localhost:8668/docs

### Face Recognition System

Located in [src/face_recognition/](src/face_recognition/):
- **FaceDetector** - MTCNN for face detection and alignment
- **FaceEmbedder** - InceptionResnetV1 for 512-dimensional embeddings
- **Repository pattern** - Database access for face profiles
- Uses pgvector for efficient similarity search
- Configurable via `ENABLE_FACE_RECOGNITION` env var (default: true)

### WebSocket Real-Time Event Broadcasting

Located in [src/core/websocket_manager.py](src/core/websocket_manager.py) and [src/api/websocket.py](src/api/websocket.py):

**WebSocketManager** - Singleton connection manager
- Manages active WebSocket connections for real-time event broadcasting
- Supports subscribing to all events or specific camera events
- Automatic cleanup of disconnected clients
- Thread-safe broadcasting from camera processing threads

**WebSocket Endpoints**:
- `ws://localhost:8668/ws/events` - Subscribe to all camera events
- `ws://localhost:8668/ws/events/{camera_id}` - Subscribe to specific camera events
- `GET /ws/stats` - Get WebSocket connection statistics

**Message Format**:
```json
{
  "event_id": "uuid",
  "camera_id": "uuid",
  "camera_name": "string",
  "model_id": "uuid",
  "model_name": "string",
  "model_type": "string",
  "time": "ISO datetime",
  "metadata": {},
  "image_path": "string"
}
```

**Integration**:
- Events are automatically broadcasted after successful database save in `SingleThreadProcessor.save_event()`
- Uses `asyncio.run_coroutine_threadsafe()` to bridge synchronous processing threads with async WebSocket
- Supports ping/pong for connection keep-alive

**Testing**:
```bash
# Listen to all events
python test_websocket.py

# Listen to specific camera
python test_websocket.py --camera-id <camera-uuid>
```

### Video Streaming System (WebRTC / RTSP / MJPEG)

**FrameBufferManager** ([src/core/frame_buffer.py](src/core/frame_buffer.py))
- Singleton per-camera latest-frame store
- Thread-safe via per-camera locks
- Timestamp tracking with offline detection (>3s = stale)
- Bridge between AI processing and all output pipelines

**WebRTCManager** ([src/core/webrtc_manager.py](src/core/webrtc_manager.py))
- Creates `PeerConnection` per client via HTTP signaling
- `FrameVideoTrack` reads annotated frames from `FrameBufferManager`
- Configurable FPS (5/10/24/45/60, default 24)
- ICE configuration via `WEBRTC_STUN_SERVER` / `WEBRTC_TURN_SERVER` env vars
- Requires `aiortc` library

**RTSPOutputManager** ([src/core/rtsp_output.py](src/core/rtsp_output.py))
- Background thread pipes BGR frames to FFmpeg subprocess
- Auto-detects NVENC on Jetson, falls back to libx264 ultrafast
- Pushes to local RTSP server (MediaMTX) at `RTSP_OUTPUT_URL`
- **Auto-starts/stops** with camera lifecycle via `CameraManager`
- Disabled by default (`RTSP_OUTPUT_ENABLED=false`)
- When enabled, each camera publishes to `rtsp://{RTSP_OUTPUT_URL}/{camera_id}`
- Consumable by FFmpeg, VLC, OBS, or any RTSP client

**WebRTC Client** ([webrtc_client.html](webrtc_client.html))
- Browser test page for WebRTC streaming
- Enter camera UUID and server URL, click Start to view annotated video

## Configuration

### Environment Variables (.env)

**Database**:
- `DB_HOST` - PostgreSQL host
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password

**Performance**:
- `FPS_LIMIT` - Max FPS per camera (default: 5, optimized for Jetson)
- `PERFORMANCE_MONITOR_INTERVAL` - Metrics collection interval in seconds (default: 600)

**Features**:
- `ENABLE_FACE_RECOGNITION` - Enable/disable face recognition (default: true)

**Storage**:
- `EVIDENCE_IMAGE_BASE_PATH` - Base path for evidence images (default: evidence_image)
- `EVIDENCE_RETENTION_DAYS` - Evidence retention period (default: 30)

**Video Streaming**:
- `WEBRTC_ENABLED` - Enable WebRTC endpoints (default: true)
- `WEBRTC_STUN_SERVER` - STUN server URL (default: stun:stun.l.google.com:19302)
- `WEBRTC_TURN_SERVER` - TURN server URL (default: empty)
- `WEBRTC_TURN_USERNAME` / `WEBRTC_TURN_PASSWORD` - TURN credentials
- `WEBRTC_DEFAULT_FPS` - Default WebRTC stream FPS (default: 24)
- `RTSP_OUTPUT_ENABLED` - Enable RTSP output workers (default: false)
- `RTSP_OUTPUT_URL` - MediaMTX base URL (default: rtsp://localhost:8554)
- `RTSP_ENCODER` - Encoder selection: auto/h264_nvenc/libx264 (default: auto)

### Model Configuration

AI models are configured in database via [scripts/sync_models_to_db.py](scripts/sync_models_to_db.py):

```python
{
    "name": "Helmet Detection",
    "version": "1.0.0",
    "model_path": "src/ai_models/model_weights/atld_92.pt",
    "parameters": {
        "model_type": "helmet_detection",
        "conf_threshold": 0.7,
        "iou_threshold": 0.3,
        "enable_tracking": True,
        "tracking_reset_minutes": 30
    }
}
```

## Important Patterns & Conventions

### Adding a New AI Model

1. Create new model class in [src/ai_models/](src/ai_models/) inheriting from `BaseModel`
2. Implement `process_frame(frame, **kwargs) → DetectionResult`
3. Register model type in `ModelFactory` ([src/core/model_factory.py](src/core/model_factory.py))
4. Add model configuration to `MODELS_CONFIG` in [scripts/sync_models_to_db.py](scripts/sync_models_to_db.py)
5. Run `python scripts/sync_models_to_db.py --sync` to load into database
6. Assign model to cameras via API: `POST /api/ai_models/{model_id}/cameras/{camera_id}`

### Threading Model

- **Main thread**: FastAPI async event loop
- **Daemon thread**: PerformanceMonitor (background metrics collection)
- **Per-camera threads**:
  - CameraThread (capture from RTSP stream)
  - SingleThreadProcessor (AI inference pipeline)
- All threads managed by ThreadManager singleton
- Proper cleanup with stop_event signals and thread.join()

### Object Tracking

Most AI models include object tracking to:
- Reduce duplicate events (same object detected in consecutive frames)
- Track object movement across frames
- Auto-reset tracking after 30 minutes of inactivity (configurable)

Tracking IDs included in event metadata for correlation.

### Error Handling

- RTSP connection failures: Auto-reconnect with exponential backoff (max 5 retries)
- Frame processing errors: Logged but don't crash threads
- Database errors: Rolled back with error logging
- Model loading failures: Graceful degradation, skip problematic models

## Database Setup

Requires PostgreSQL 12+ with pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Tables are auto-created on server startup via SQLAlchemy:
```python
Base.metadata.create_all(bind=engine)
```

## Deployment on Jetson

1. Install Python 3.12.6 (exact version required)
2. Install CUDA 12.8 drivers and toolkit
3. Install PostgreSQL and enable pgvector extension
4. Clone repository and run `uv sync`
5. Configure `.env` with database credentials
6. Sync AI models: `python scripts/sync_models_to_db.py --sync`
7. Place model weight files (.pt, .onnx, .engine) in [src/ai_models/model_weights/](src/ai_models/model_weights/)
8. Start server: `python -m src.server`

## Future Work

- TensorRT optimization for faster inference on Jetson
- Multi-location deployment with centralized management
- Advanced analytics and reporting dashboards

## File Structure

```
src/
├── server.py              # FastAPI main application
├── main_run.py            # Test runner
├── ai_models/             # 10+ AI model implementations
│   ├── base_model.py      # BaseModel interface
│   ├── helmet_model.py
│   ├── oil_spill_model.py
│   └── model_weights/     # .pt, .onnx, .engine files
├── api/                   # 9 API routers (50+ endpoints)
├── core/                  # Core processing components
│   ├── thread_manager.py
│   ├── camera_single_thread.py
│   ├── model_factory.py
│   ├── event_storage_service.py
│   └── performance_monitor.py
├── models/                # 9 SQLAlchemy database models
├── database/              # Database configuration
├── face_recognition/      # Face detection and recognition
├── configs/               # Configuration modules
└── utils/                 # Utilities

scripts/
├── sync_models_to_db.py   # Load AI models into database
├── update_model_parameters.py
└── delete_all_models.py

evidence_image/            # Evidence storage (auto-created)
└── {YYYY-MM-DD}/
    └── {camera-uuid}/
        └── {HH-MM-SS.mmm.jpg}
```
