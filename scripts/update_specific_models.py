#!/usr/bin/env python3
"""
Script to update specific AI models to database
Only syncs oil_spill and smoke_fire models
Automatically keeps all other models (--keep-stale)

Usage:
    python scripts/update_specific_models.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from sqlalchemy.orm import Session
from src.database import SessionLocal, engine
from src.models.ai_model import AiModel
from src.models.base import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Only specific models to update
MODELS_CONFIG = [
    {
        "name": "Vết loang bất thường",
        "description": "Phát hiện vết loang bất thường trên mặt đất, có thể là dầu tràn hoặc chất lỏng nguy hiểm",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/oil_11l_100ep.pt",
        "parameters": {
            "model_type": "oil_spill",
            "conf_threshold": 0.5,
            "iou_threshold": 0.3,
        },
        "is_active": True,
    },
    {
        "name": "Phát hiện khói lửa",
        "description": "Phát hiện khói và lửa trong khu vực giám sát, cảnh báo sớm nguy cơ cháy nổ",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/YOLOv10-FireSmoke-X.pt",
        "parameters": {
            "model_type": "smoke_fire",
            "conf_threshold": 0.5,
            "iou_threshold": 0.3,
        },
        "is_active": True,
    },
]

STANDARDIZED_MODEL_OVERRIDES = {
    "oil_spill": {
        "name": "Vết loang bất thường",
        "description": "Phát hiện vết loang bất thường trên mặt đất, có thể là dầu tràn hoặc chất lỏng nguy hiểm",
        "parameters": {
            "class_names": ["Vết loang bất thường"],
            "event_type": "Xuất hiện vết loang bất thường",
        },
    },
    "smoke_fire": {
        "name": "Phát hiện khói/cháy",
        "description": "Phát hiện khói bất thường và nhận diện cháy trong khu vực giám sát",
        "parameters": {
            "class_names": ["Lửa", "Khói"],
            "event_type": "Phát hiện khói/cháy",
        },
    },
}

for model_config in MODELS_CONFIG:
    parameters = model_config.setdefault("parameters", {})
    overrides = STANDARDIZED_MODEL_OVERRIDES.get(parameters.get("model_type"))
    if not overrides:
        continue

    for field in ("name", "description"):
        if field in overrides:
            model_config[field] = overrides[field]

    for key, value in overrides.get("parameters", {}).items():
        parameters[key] = value


def get_model_key(model_config: Dict) -> Tuple[str, str]:
    """Return the unique key used to identify a configured model."""
    return model_config["name"], model_config["version"]


def create_tables():
    """Create database tables if they don't exist"""
    logger.info("Creating database tables if needed...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")


def sync_model(db: Session, model_config: dict) -> AiModel:
    """
    Sync a single model to database
    Updates if exists, creates if not

    Args:
        db: Database session
        model_config: Model configuration dictionary

    Returns:
        AiModel instance
    """
    # Check if model exists by name and version
    existing_model = db.query(AiModel).filter(
        AiModel.name == model_config["name"],
        AiModel.version == model_config["version"]
    ).first()

    if existing_model:
        # Update existing model
        logger.info(f"Updating existing model: {model_config['name']} v{model_config['version']}")
        existing_model.description = model_config["description"]
        existing_model.model_path = model_config["model_path"]
        existing_model.parameters = model_config["parameters"]
        existing_model.is_active = model_config["is_active"]
        db.commit()
        db.refresh(existing_model)
        return existing_model
    else:
        # Create new model
        logger.info(f"Creating new model: {model_config['name']} v{model_config['version']}")
        new_model = AiModel(
            name=model_config["name"],
            description=model_config["description"],
            version=model_config["version"],
            model_path=model_config["model_path"],
            parameters=model_config["parameters"],
            is_active=model_config["is_active"],
            is_latest_used=False,
        )
        db.add(new_model)
        db.commit()
        db.refresh(new_model)
        return new_model


def sync_all_models():
    """Synchronize specific models to database (keeps all other models)"""
    logger.info("=" * 60)
    logger.info("Starting Specific AI Models Update")
    logger.info("=" * 60)

    # Create tables
    create_tables()

    # Open database session
    db = SessionLocal()

    try:
        synced_models = []

        for model_config in MODELS_CONFIG:
            try:
                model = sync_model(db, model_config)
                synced_models.append(model)
                logger.info(f"✓ Synced: {model.name} (ID: {model.id})")
            except Exception as e:
                logger.error(f"✗ Failed to sync {model_config['name']}: {e}", exc_info=True)
                db.rollback()

        logger.info("=" * 60)
        logger.info(
            "Update Complete: %s/%s models synced",
            len(synced_models),
            len(MODELS_CONFIG),
        )
        logger.info("=" * 60)

        # Display summary
        logger.info("\nUpdated Models:")
        logger.info("-" * 60)
        for model in synced_models:
            model_type = model.parameters.get('model_type', 'N/A')
            conf_threshold = model.parameters.get('conf_threshold', 'N/A')
            status = "ACTIVE" if model.is_active else "INACTIVE"
            logger.info(f"  {model.name}")
            logger.info(f"    - ID: {model.id}")
            logger.info(f"    - Type: {model_type}")
            logger.info(f"    - Version: {model.version}")
            logger.info(f"    - Confidence: {conf_threshold}")
            logger.info(f"    - Status: {status}")
            logger.info(f"    - Path: {model.model_path}")
            logger.info("")
        
        logger.info("NOTE: Other models in database remain unchanged!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Critical error during update: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def list_models():
    """List all models in database"""
    logger.info("=" * 60)
    logger.info("Current Models in Database")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        models = db.query(AiModel).order_by(AiModel.name, AiModel.version).all()

        if not models:
            logger.info("No models found in database")
            return

        for model in models:
            model_type = model.parameters.get('model_type', 'N/A')
            conf_threshold = model.parameters.get('conf_threshold', 'N/A')
            status = "ACTIVE" if model.is_active else "INACTIVE"

            logger.info(f"\n{model.name} v{model.version}")
            logger.info(f"  ID: {model.id}")
            logger.info(f"  Type: {model_type}")
            logger.info(f"  Confidence: {conf_threshold}")
            logger.info(f"  Status: {status}")
            logger.info(f"  Path: {model.model_path}")

        logger.info(f"\nTotal: {len(models)} models")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update specific AI models")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in database"
    )

    args = parser.parse_args()

    # Always sync (no pruning)
    sync_all_models()

    if args.list:
        list_models()
