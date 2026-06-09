#!/usr/bin/env python3
"""
Script to update AI model parameters in database

Usage:
    # Update specific model by ID
    python scripts/update_model_parameters.py --model-id <UUID> --conf 0.7 --iou 0.3

    # Update all models of a type
    python scripts/update_model_parameters.py --model-type helmet_detection --conf 0.7

    # Batch update all models
    python scripts/update_model_parameters.py --all --conf 0.7 --iou 0.3
"""

import sys
import os
from pathlib import Path
from uuid import UUID

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models.ai_model import AiModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_model_parameters(
    db: Session,
    model: AiModel,
    conf_threshold: float = None,
    iou_threshold: float = None,
    other_params: dict = None
):
    """
    Update model parameters

    Args:
        db: Database session
        model: AiModel instance
        conf_threshold: New confidence threshold
        iou_threshold: New IOU threshold
        other_params: Other parameters to update
    """
    parameters = model.parameters.copy()
    updated = False

    if conf_threshold is not None:
        parameters['conf_threshold'] = conf_threshold
        updated = True
        logger.info(f"  - conf_threshold: {parameters.get('conf_threshold', 'N/A')} → {conf_threshold}")

    if iou_threshold is not None:
        parameters['iou_threshold'] = iou_threshold
        updated = True
        logger.info(f"  - iou_threshold: {parameters.get('iou_threshold', 'N/A')} → {iou_threshold}")

    if other_params:
        for key, value in other_params.items():
            old_value = parameters.get(key, 'N/A')
            parameters[key] = value
            updated = True
            logger.info(f"  - {key}: {old_value} → {value}")

    if updated:
        model.parameters = parameters
        db.commit()
        logger.info(f"✓ Updated: {model.name} v{model.version}")
        return True
    else:
        logger.info(f"○ No changes: {model.name} v{model.version}")
        return False


def update_by_model_id(model_id: str, conf: float = None, iou: float = None):
    """Update parameters for a specific model by ID"""
    db = SessionLocal()
    try:
        model = db.query(AiModel).filter(AiModel.id == UUID(model_id)).first()

        if not model:
            logger.error(f"Model with ID {model_id} not found")
            return False

        logger.info(f"Updating model: {model.name} v{model.version}")
        return update_model_parameters(db, model, conf, iou)

    finally:
        db.close()


def update_by_model_type(model_type: str, conf: float = None, iou: float = None):
    """Update parameters for all models of a specific type"""
    db = SessionLocal()
    try:
        models = db.query(AiModel).all()
        matching_models = [m for m in models if m.parameters.get('model_type') == model_type]

        if not matching_models:
            logger.error(f"No models found with type: {model_type}")
            return False

        logger.info(f"Found {len(matching_models)} models with type '{model_type}'")

        updated_count = 0
        for model in matching_models:
            logger.info(f"\nUpdating: {model.name} v{model.version}")
            if update_model_parameters(db, model, conf, iou):
                updated_count += 1

        logger.info(f"\n✓ Updated {updated_count}/{len(matching_models)} models")
        return True

    finally:
        db.close()


def update_all_models(conf: float = None, iou: float = None):
    """Update parameters for all active models"""
    db = SessionLocal()
    try:
        models = db.query(AiModel).filter(AiModel.is_active == True).all()

        if not models:
            logger.warning("No active models found")
            return False

        logger.info(f"Found {len(models)} active models")

        updated_count = 0
        for model in models:
            logger.info(f"\nUpdating: {model.name} v{model.version}")
            if update_model_parameters(db, model, conf, iou):
                updated_count += 1

        logger.info(f"\n✓ Updated {updated_count}/{len(models)} models")
        return True

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update AI model parameters")

    # Selection options
    parser.add_argument("--model-id", help="Update specific model by UUID")
    parser.add_argument("--model-type", help="Update all models of this type (e.g., helmet_detection)")
    parser.add_argument("--all", action="store_true", help="Update all active models")

    # Parameter options
    parser.add_argument("--conf", type=float, help="New confidence threshold (0.0-1.0)")
    parser.add_argument("--iou", type=float, help="New IOU threshold (0.0-1.0)")

    args = parser.parse_args()

    # Validate parameters
    if args.conf is not None and not (0.0 <= args.conf <= 1.0):
        logger.error("Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if args.iou is not None and not (0.0 <= args.iou <= 1.0):
        logger.error("IOU threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if not (args.model_id or args.model_type or args.all):
        logger.error("Please specify --model-id, --model-type, or --all")
        parser.print_help()
        sys.exit(1)

    if not (args.conf or args.iou):
        logger.error("Please specify at least one parameter to update (--conf or --iou)")
        parser.print_help()
        sys.exit(1)

    # Execute update
    logger.info("=" * 60)
    logger.info("AI Model Parameters Update")
    logger.info("=" * 60)

    success = False
    if args.model_id:
        success = update_by_model_id(args.model_id, args.conf, args.iou)
    elif args.model_type:
        success = update_by_model_type(args.model_type, args.conf, args.iou)
    elif args.all:
        success = update_all_models(args.conf, args.iou)

    logger.info("=" * 60)
    if success:
        logger.info("Update completed successfully")
    else:
        logger.error("Update failed")
        sys.exit(1)
