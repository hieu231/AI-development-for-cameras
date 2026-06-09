#!/usr/bin/env python3
"""
Export all active YOLO .pt models to TensorRT .engine format.

Usage:
    python scripts/export_to_tensorrt.py [--half] [--imgsz 640] [--dry-run]

Requires: tensorrt, ultralytics
"""

import argparse
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Models that are actively used in the system (from DB config)
ACTIVE_MODELS = [
    # Primary detection models
    "src/ai_models/model_weights/oil_11l_100ep.pt",
    "src/ai_models/model_weights/YOLOv10-FireSmoke-X.pt",
    "src/ai_models/model_weights/yolo11m.pt",
    "src/ai_models/model_weights/petrolimex.pt",
    "src/ai_models/model_weights/atld_92.pt",
    "src/ai_models/model_weights/vehicle_detection.pt",
    "src/ai_models/model_weights/tran_dau.pt",
    # Auxiliary models (used by smoke_fire and oil_spill for false positive filtering)
    "src/ai_models/model_weights/yolo11n.pt",
]


def export_model(model_path: str, half: bool = True, imgsz: int = 640, dry_run: bool = False) -> str | None:
    """Export a single .pt model to TensorRT .engine format.

    Returns the .engine path on success, None on failure.
    """
    engine_path = model_path.replace(".pt", ".engine")

    if not os.path.exists(model_path):
        logger.warning(f"SKIP (not found): {model_path}")
        return None

    if os.path.exists(engine_path):
        logger.info(f"SKIP (engine exists): {engine_path}")
        return engine_path

    if dry_run:
        logger.info(f"DRY RUN: would export {model_path} -> {engine_path}")
        return engine_path

    logger.info(f"Exporting: {model_path} -> TensorRT (half={half}, imgsz={imgsz})")
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        result = model.export(format="engine", half=half, imgsz=imgsz)
        logger.info(f"SUCCESS: {result}")
        return str(result)
    except Exception as e:
        logger.error(f"FAILED to export {model_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Export YOLO models to TensorRT")
    parser.add_argument("--half", action="store_true", default=True, help="Use FP16 (default: True)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16, use FP32")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be exported")
    parser.add_argument("--model", type=str, help="Export only a specific model path")
    args = parser.parse_args()

    half = not args.no_half

    # Verify TensorRT is available
    if not args.dry_run:
        try:
            import tensorrt
            logger.info(f"TensorRT version: {tensorrt.__version__}")
        except ImportError:
            logger.error("TensorRT is not installed. Run: sudo apt install -y python3-libnvinfer libnvinfer10 libnvinfer-plugin10")
            sys.exit(1)

    models = [args.model] if args.model else ACTIVE_MODELS
    results = {"success": [], "failed": [], "skipped": []}

    for model_path in models:
        result = export_model(model_path, half=half, imgsz=args.imgsz, dry_run=args.dry_run)
        if result:
            results["success"].append(model_path)
        elif not os.path.exists(model_path):
            results["skipped"].append(model_path)
        else:
            results["failed"].append(model_path)

    logger.info(f"\n{'='*60}")
    logger.info(f"Export complete: {len(results['success'])} success, {len(results['failed'])} failed, {len(results['skipped'])} skipped")
    for path in results["success"]:
        engine = path.replace(".pt", ".engine")
        logger.info(f"  OK: {engine}")
    for path in results["failed"]:
        logger.error(f"  FAIL: {path}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
