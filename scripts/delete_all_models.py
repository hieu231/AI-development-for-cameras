#!/usr/bin/env python3
"""Delete all records from all tables in the database"""
import sys
from pathlib import Path
from typing import Set

from sqlalchemy import inspect

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import Base, engine  # noqa: E402
from src.database.get_db import SessionLocal  # noqa: E402
from src.models import (  # noqa: E402
    Camera,
    CameraModel,
    AiModel,
    Location,
    CameraSpec,
    Event,
    Performance,
)


def ensure_schema():
    """Create any missing tables before we try deleting from them."""
    Base.metadata.create_all(bind=engine)


def get_existing_tables() -> Set[str]:
    """Return set of current public tables to allow graceful skips."""
    inspector = inspect(engine)
    return set(inspector.get_table_names(schema="public"))

def main():
    ensure_schema()
    existing_tables = get_existing_tables()
    db = SessionLocal()
    try:
        # Delete in reverse dependency order
        db.query(Event).delete(synchronize_session=False)

        if Performance.__tablename__ in existing_tables:
            db.query(Performance).delete(synchronize_session=False)
        else:
            print(
                f"⚠️  Skipping delete for missing table '{Performance.__tablename__}' "
                "— run the server or schema initializer to create it if needed."
            )

        db.query(Camera).delete(synchronize_session=False)
        db.query(CameraModel).delete(synchronize_session=False)
        db.query(CameraSpec).delete(synchronize_session=False)
        db.query(AiModel).delete(synchronize_session=False)
        db.query(Location).delete(synchronize_session=False)
        db.commit()
        print("✓ All records deleted")
    except Exception as e:
        db.rollback()
        print(f"✗ Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()

