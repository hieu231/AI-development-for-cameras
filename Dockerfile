# =============================================================================
# Petrolimex AI Backend — production image
#
# Default base: official NVIDIA PyTorch container for ARM SBSA / Ubuntu 24.04.
# This matches DGX Spark / GB10 style systems (aarch64 + Ubuntu 24.04), not
# Jetson/L4T. The container already includes CUDA-enabled PyTorch.
#
# Notes:
# - `nvcr.io/nvidia/pytorch:25.10-py3` is the safer default for GB10 than the
#   old Jetson/L4T images.
# - Pulling from `nvcr.io` may require `docker login nvcr.io`.
#
# Overrides:
# - x86_64 desktop/server:
#     docker compose build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
# - If you must stay on a custom host venv instead of Docker, keep Dockerfile
#   unchanged and run the app directly from `.venv`.
# =============================================================================
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Ho_Chi_Minh

# --- System dependencies -----------------------------------------------------
# ffmpeg / libav* are required for RTSP, HTTP-FLV, MJPEG, and RTSP output.
# libgl1, libglib2.0-0 are required by opencv-python headless helpers used
# by Ultralytics. libpq5 is required by psycopg2-binary.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libavcodec-extra \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libpq5 \
    curl \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cache layer) --------------------------------------
# This NVIDIA PyTorch base image already provides CUDA-enabled torch/torchvision.
# Do NOT reinstall those. It does not reliably ship `cv2`, so install a
# headless OpenCV wheel explicitly.
# Override dustynv's broken jetson.webredirect.org pip index
ENV PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com

COPY pyproject.toml requirements.txt* ./
# `facenet-pytorch` depends on torch/torchvision. Install it with `--no-deps`
# so pip does not replace the CUDA-enabled torch provided by the base image.
# After dependency installation, fail the build immediately if torch has been
# replaced by a CPU-only wheel. We check `torch.version.cuda` instead of
# `torch.cuda.is_available()` because GPU devices may not be exposed at build time.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-deps \
    "ultralytics" && \
    pip install \
    "fastapi" \
    "uvicorn[standard]" \
    "sqlalchemy>=2.0.0" \
    "psycopg2-binary>=2.9.9" \
    "python-dotenv>=1.0.0" \
    "pyyaml>=6.0.0" \
    "requests>=2.31.0" \
    "paho-mqtt>=2.1.0" \
    "websocket-client>=1.8.0" \
    "websockets>=12.0" \
    "httpx" \
    "numpy" \
    "pandas" \
    "pillow" \
    "scipy" \
    "pgvector" \
    "lap>=0.5.12" \
    "aiortc>=1.9.0" \
    "av>=12.0.0" \
    "python-multipart>=0.0.9" \
    "openvino==2024.6.0" \
    "onnxruntime>=1.17.0" && \
    pip install --no-deps \
    "opencv-python-headless" && \
    pip install --no-deps \
    "facenet-pytorch" && \
    pip install \
    "fast-alpr[onnx-gpu]" \
    "google-generativeai" \
    "insightface>=0.7.3" && \
    # ── ORT GPU on Jetson aarch64 (CUDA 13 host) ─────────────────
    # Standard onnxruntime PyPI wheel is CPU-only on aarch64. Pull
    # the Jetson-specific GPU wheel from jetson-ai-lab. Its CUDA EP
    # dlopens libcudart.so.12 / libcublas.so.12 / libcufft.so.11 etc.
    # which ship in the nvidia-*-cu12 pip wheels — install them and
    # extend LD_LIBRARY_PATH (set in docker-compose.yml) so the EPs
    # actually load. Without this, fast-alpr + insightface burn the
    # CPU at 90%+ and bbox lag is severe.
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true && \
    pip install --force-reinstall \
      --index-url https://pypi.jetson-ai-lab.io/jp6/cu129/+simple \
      --extra-index-url https://pypi.org/simple \
      "onnxruntime-gpu==1.23.0" && \
    pip install \
      "nvidia-cuda-runtime-cu12" \
      "nvidia-cublas-cu12" \
      "nvidia-cudnn-cu12" \
      "nvidia-cufft-cu12" \
      "nvidia-curand-cu12" \
      "nvidia-cusparse-cu12" \
      "nvidia-nccl-cu12" && \
    python3 - <<'PY'
import torch
import cv2
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("torch.__file__", torch.__file__)
print("cv2", cv2.__version__)
assert torch.version.cuda is not None, "CPU-only torch was installed into the image"
PY

# --- Application source ------------------------------------------------------
# NOTE: model_weights/ is excluded via .dockerignore and mounted at runtime.
COPY src ./src
COPY scripts ./scripts
COPY conftest.py ./
COPY README.md ./

# Package metadata so `python3 -m src.server` and setuptools find the package.
RUN pip install --no-deps -e .

# Pre-create mount points with correct ownership.
RUN mkdir -p /app/evidence_image /app/src/ai_models/model_weights /app/logs

EXPOSE 8668

# Healthcheck hits the FastAPI /health endpoint on the configured port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8668}/health" || exit 1

# Entrypoint: production mode — no uvicorn reload. Hot-reload is dev-only
# and leaks CUDA contexts on file touches (see src/server.py).
CMD ["python3", "-m", "src.server"]
