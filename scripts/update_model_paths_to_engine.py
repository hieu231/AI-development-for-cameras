#!/usr/bin/env python3
"""
Update AI model paths in PostgreSQL from .pt to .engine (TensorRT).

Usage:
    python scripts/update_model_paths_to_engine.py [--dry-run] [--revert]

This updates the `model_path` column in the `ai_models` table,
changing .pt extensions to .engine for all models where the .engine file exists.
"""

import argparse
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def update_paths(dry_run: bool = False, revert: bool = False):
    from src.database.database import SessionLocal
    from src.models.ai_model import AiModel

    db = SessionLocal()
    try:
        models = db.query(AiModel).all()
        updated = 0

        for model in models:
            if not model.model_path:
                continue

            if revert:
                # Revert .engine back to .pt
                if model.model_path.endswith(".engine"):
                    pt_path = model.model_path.replace(".engine", ".pt")
                    if os.path.exists(pt_path):
                        logger.info(f"REVERT: {model.name}: {model.model_path} -> {pt_path}")
                        if not dry_run:
                            model.model_path = pt_path
                        updated += 1
            else:
                # Upgrade .pt to .engine
                if model.model_path.endswith(".pt"):
                    engine_path = model.model_path.replace(".pt", ".engine")
                    if os.path.exists(engine_path):
                        logger.info(f"UPDATE: {model.name}: {model.model_path} -> {engine_path}")
                        if not dry_run:
                            model.model_path = engine_path
                        updated += 1
                    else:
                        logger.warning(f"SKIP (no .engine): {model.name}: {model.model_path}")

        if not dry_run and updated > 0:
            db.commit()
            logger.info(f"Committed {updated} path updates to database")
        else:
            logger.info(f"{'DRY RUN: ' if dry_run else ''}Would update {updated} model paths")

    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Update model paths to TensorRT engines")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without applying")
    parser.add_argument("--revert", action="store_true", help="Revert .engine paths back to .pt")
    args = parser.parse_args()

    update_paths(dry_run=args.dry_run, revert=args.revert)


if __name__ == "__main__":
    main()
