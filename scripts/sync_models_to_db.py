#!/usr/bin/env python3
"""
Script to synchronize AI models to database
Automatically adds or updates AI models in the database

Primary keys MUST match cloud device-registry canonical IDs so MQTT camera commands
(device-registry -> edge petrobe) resolve `ai_models.id` correctly.

Source of truth (keep in sync):
  aibox_backend_FRESH/services/device-registry/src/ai-models/canonical-ai-models.ts

Usage (example):
    source .venv/bin/activate
    DB_HOST=localhost DB_USER=dang321 DB_NAME=petrobe_db DB_PASSWORD=*** DB_PORT=5433 \\
      python -m scripts.sync_models_to_db

    python -m scripts.sync_models_to_db --list          # list DB rows
    python -m scripts.sync_models_to_db --sync        # sync (default)
    python -m scripts.sync_models_to_db --sync --keep-stale   # do not delete extra rows

After changing IDs on a DB that already had random UUIDs: delete conflicting `ai_models` rows
(or run with a fresh DB), then re-run sync so rows are inserted with canonical ids.
"""

import sys
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from sqlalchemy.orm import Session
from src.database import SessionLocal, engine
from src.models.ai_model import AiModel
from src.models.camera_model import CameraModel
from src.models.event import Event
from src.models.base import Base
from src.utils.parameter_defaults import get_default_additional_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Maps parameters.model_type -> ai_models.id (must match device-registry CANONICAL_AI_MODEL_SEEDS).
# See: aibox_backend_FRESH/services/device-registry/src/ai-models/canonical-ai-models.ts
#
# IMPORTANT: any UUID added here MUST also be added to the cloud canonical
# seed file in the aibox_backend repo at the same path (and that service
# redeployed) so that PA-side model assignments can refer to the same id.
CANONICAL_DEVICE_REGISTRY_UUID_BY_MODEL_TYPE: Dict[str, str] = {
    "helmet_detection": "60000000-0000-4000-8000-000000000001",
    "oil_spill": "60000000-0000-4000-8000-000000000002",
    "smoke_fire": "60000000-0000-4000-8000-000000000003",
    "face_recognition": "60000000-0000-4000-8000-000000000004",
    "oil_dumping": "60000000-0000-4000-8000-000000000005",
    "people_control": "60000000-0000-4000-8000-000000000006",
    "alpr": "60000000-0000-4000-8000-000000000007",
    "gemini_plate": "60000000-0000-4000-8000-000000000008",
    "petrolimex_detection_model": "60000000-0000-4000-8000-000000000009",
    "tran_dau": "60000000-0000-4000-8000-00000000000a",
    "oil_cap_detection": "60000000-0000-4000-8000-00000000000b",
    "smoking_behavior": "60000000-0000-4000-8000-00000000000c",
    "workspace_monitor": "60000000-0000-4000-8000-00000000000d",
    "evn_smartech": "60000000-0000-4000-8000-00000000000e",
}


def canonical_uuid_for_model_config(model_config: dict) -> Optional[uuid.UUID]:
    """Return fixed UUID when model_type is in the device-registry canonical map; else None."""
    mt = model_config.get("parameters", {}).get("model_type")
    if not mt or not isinstance(mt, str):
        return None
    s = CANONICAL_DEVICE_REGISTRY_UUID_BY_MODEL_TYPE.get(mt.strip())
    return uuid.UUID(s) if s else None


# Define available models with their configurations
MODELS_CONFIG = [
    
    {
        "name": "Trang phục bảo hộ EVN & Smartech",
        "description": "Phát hiện và ghi nhận công nhân EVN, Smartech và người lạ trong khu vực giám sát",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/ppe_100ep.pt",
        "parameters": {
            "model_type": "evn_smartech",
            "conf_threshold": 0.55,
            "classes": [0, 1, 2],
            "class_names": ["EVN-Worker", "Smartech-Worker", "no vest"]
        },
        "is_active": True,
    },


    {
        "name": "An toàn lao động",
        "description": "An toàn lao động - Phát hiện vi phạm PPE: không mặc áo phản quang, không đội mũ bảo hộ",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/atld_92.pt",
        "parameters": {
            "model_type": "helmet_detection",
            "conf_threshold": 0.55,
            "person_conf_threshold": 0.6,
            "classes": [0, 1, 2, 3, 4],  # Hardhat, NO-Hardhat, NO-Vest, Person, Vest
        },
        "is_active": True,
    },
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
        "name": "Dòng chảy tràn dầu đột ngột",
        "description": "Phát hiện dòng chảy tràn dầu đột ngột, cảnh báo nhanh và yêu cầu xử lý tức thời",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/tran_dau.pt",
        "parameters": {
            "model_type": "tran_dau",
            "conf_threshold": 0.7,
            "iou_threshold": 0.3,
            "detection_cooldown": 60,
            "min_consecutive_frames": 1,
            "min_flow_frames": 1,
            "flow_motion_ratio_threshold": 0.08,
            "flow_edge_motion_ratio_threshold": 0.12,
            "sudden_area_growth_ratio": 0.18,
            "high_severity_area_ratio": 0.008,
            "critical_severity_area_ratio": 0.02,
            "global_event_cooldown": 3.0,
            "spill_label": "Dòng chảy tràn dầu đột ngột",
            "metadata_type": "Dòng chảy tràn dầu đột ngột",
            "overlay_label": "DONG CHAY DAU DOT NGOT",
            "class_names": ["Dòng chảy tràn dầu đột ngột"],
            "event_type": "Dòng chảy tràn dầu đột ngột",
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

    {
        "name": "Nhận diện khuôn mặt",
        "description": "Nhận diện khuôn mặt công nhân, phân loại người quen và người lạ trong khu vực giám sát",
        "version": "1.0.0",
        "model_path": None,  # Uses face recognition engine
        "parameters": {
            "model_type": "face_recognition",
            "face_similarity_threshold": 0.45,
            "known_cooldown": 300,
            "unknown_cooldown": 60,
        },
        "is_active": True,
    },
    {
        "name": "Phát hiện hành vi xả van dầu",
        "description": "Phát hiện người đứng chờ trong khu vực được chỉ định (tiềm ẩn đổ dầu)",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11n.pt",
        "parameters": {
            "model_type": "oil_dumping",
            "conf_threshold": 0.5,
            "detection_time_threshold": 20,
            "detection_cooldown": 28800,
        },
        "is_active": True,
    },
    {
        "name": "Người vào khu vực cấm",
        "description": "Phát hiện người vào khu vực cấm thông qua hàng rào ảo được định nghĩa bởi người dùng",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11m.pt",
        "parameters": {
            "model_type": "people_control",
            "conf_threshold": 0.6,
            "detection_cooldown": 300,
        },
        "is_active": True,
    },
    {
        "name": "Nhận diện biển số xe tự động",
        "description": "Nhận diện biển số xe tự động (ALPR) sử dụng mô hình Fast ALPR",
        "version": "1.0.0",
        "model_path": None,  # Uses fast-alpr
        "parameters": {
            "model_type": "alpr",
            "buffer_size": 5,
            "similarity_threshold": 0.7,
            "inactivity_timeout": 60,
        },
        "is_active": True,
    },
    {
        "name": "Phát hiện xe và biển số",
        "description": "Phát hiện xe và biển số sử dụng",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/vehicle_detection.pt",
        "parameters": {
            "model_type": "gemini_plate",
            "detection_cooldown": 10,
            "frame_width": 1280,
            "frame_height": 720,
        },
        "is_active": True,
    },
    {
        "name": "Nhân viên dầu khí",
        "description": "Nhận diện nhân viên dầu khí, ghi nhận vi phạm an toàn lao động và người lạ trong khu vực giám sát",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/petrolimex.pt",
        "parameters": {
            "model_type": "petrolimex_detection_model",
            "conf_threshold": 0.45,
            "iou_threshold": 0.45,
            "person_conf_threshold": 0.5,
            "detection_cooldown": 300,
        },
        "is_active": True,
    },
    {
        "name": "Giám sát nắp dầu",
        "description": "Phát hiện trạng thái đóng hoặc mở nắp dầu, mặc định cảnh báo khi nắp dầu đang mở",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/oil_cap_detection.pt",
        "parameters": {
            "model_type": "oil_cap_detection",
            "conf_threshold": 0.45,
            "iou_threshold": 0.45,
            "detection_cooldown": 300,
            "violation_min_center_y_ratio": 0.35,
            "violation_labels": ["oil_cap_opened"],
        },
        "is_active": True,
    },
    {
        "name": "Giám sát khu vực làm việc",
        "description": "Giám sát khu vực làm việc cho thấy tình trạng quá tải, vi phạm an toàn lao động, và nhận diện nhân viên không được phép vào khu vực",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11m.pt",
        "parameters": {
            "model_type": "workspace_monitor",
            "conf_threshold": 0.4,
            "ppe_conf_threshold": 0.45,
            "max_workers": 5,
            "detection_cooldown": 300,
            "face_similarity_threshold": 0.45,
        },
        "is_active": True,
    },
    {
        "name": "Phát hiện hành vi hút thuốc",
        "description": "Phát hiện hành vi hút thuốc trong khu vực giám sát",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/last.onnx",
        "parameters": {
            "model_type": "smoking_behavior",
            "cgr_conf": 0.3,
            "threshold": 5,
            "skeleton": True,
            "cig_box": True,
            "annotate": True,
            "event_type": "Phát hiện hành vi hút thuốc",
            "class_names": ["Hút thuốc"],
        },
        "is_active": True,
    },
]

STANDARDIZED_MODEL_OVERRIDES = {
    "petrolimex_detection_model": {
        "name": "Bảo hộ lao động",
        "description": "Phát hiện vi phạm bảo hộ lao động: không mặc trang phục bảo hộ hoặc không đội mũ bảo hộ",
        "parameters": {
            "event_type": "Không tuân thủ bảo hộ lao động",
        },
    },
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
    "oil_dumping": {
        "name": "Phát hiện xả đáy",
        "description": "Phát hiện hành vi nghi ngờ xả đáy dầu trong khu vực giám sát",
        "parameters": {
            "event_type": "Phát hiện xả đáy",
        },
    },
    "people_control": {
        "name": "Xâm nhập trái phép khu vực",
        "description": "Phát hiện xâm nhập trái phép khu vực thông qua hàng rào ảo hoặc ROI giám sát",
        "parameters": {
            "event_type": "Xâm nhập trái phép khu vực",
        },
    },
    "oil_cap_detection": {
        "name": "Giám sát nắp dầu",
        "description": "Phát hiện trạng thái đóng hoặc mở nắp dầu, mặc định cảnh báo khi nắp dầu đang mở",
        "parameters": {
            "event_type": "Giám sát nắp dầu",
            "class_names": ["Nắp dầu mở", "Nắp dầu đóng"],
        },
    },
    "tran_dau": {
        "name": "Phát hiện tràn dầu",
        "description": "Phát hiện dòng chảy tràn dầu đột ngột, cảnh báo nhanh và yêu cầu xử lý tức thời",
        "parameters": {
            "spill_label": "Phát hiện tràn dầu",
            "metadata_type": "Phát hiện tràn dầu",
            "class_names": ["Phát hiện tràn dầu"],
            "event_type": "Phát hiện tràn dầu",
        },
    },
    "smoking_behavior": {
        "name": "Phát hiện hành vi hút thuốc",
        "description": "Phát hiện hành vi hút thuốc trong khu vực giám sát",
        "parameters": {
            "event_type": "Phát hiện hành vi hút thuốc",
            "class_names": ["Hút thuốc"],
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


# ── Fold parameter_defaults into every model's general parameters ───────────
# `parameter_defaults.py` is the single source of truth for sane per-camera
# defaults (ROI, conf thresholds, cooldowns, model-specific knobs). The seed
# merges them into each model's `parameters` so every freshly-assigned camera
# inherits the safe defaults from the start, instead of silently running with
# missing keys (the bug that caused alpr/gemini/face cameras to come up with
# `roi=None`, no max_roi_polygons, default cooldowns, etc.).
#
# Model-config values WIN over parameter_defaults — anything we set
# explicitly here (e.g. fine-tuned conf_threshold=0.55 for helmets) is
# preserved. Defaults only fill in keys that are missing.
for model_config in MODELS_CONFIG:
    parameters = model_config.setdefault("parameters", {})
    model_type = parameters.get("model_type")
    if not model_type:
        continue
    extra_defaults = get_default_additional_parameters(model_type) or {}
    for key, value in extra_defaults.items():
        parameters.setdefault(key, value)


# ── Fine-tuned model-level overrides (operator-tested) ─────────────────────
# Overrides that ship the latest tuned defaults to fresh installs:
# - face_recognition: 5s cooldowns + 0.55 dedup so the same face emits
#   one event per ~5s instead of dozens per second (cosine-sim dedup is
#   in face_recognition_model.py).
# - smoking_behavior: lower threshold so an event fires after ~1s of
#   continuous smoking instead of 10s, plus skeleton/cig_box ON so the
#   overlay shows what the model is reasoning about.
TUNED_MODEL_DEFAULTS = {
    "face_recognition": {
        "face_similarity_threshold": 0.45,
        "known_cooldown": 30,
        "unknown_cooldown": 5,
    },
    "smoking_behavior": {
        "threshold": 5,
        "cgr_conf": 0.3,
        "skeleton": True,
        "cig_box": True,
        "annotate": True,
    },
}
for model_config in MODELS_CONFIG:
    parameters = model_config.setdefault("parameters", {})
    tuned = TUNED_MODEL_DEFAULTS.get(parameters.get("model_type"))
    if not tuned:
        continue
    for key, value in tuned.items():
        parameters[key] = value

# Disabled hidden list: sync all model types.
HIDDEN_MODEL_TYPES = set()

ALL_MODELS_CONFIG = MODELS_CONFIG


def build_models_config(include_hidden: bool = False) -> List[Dict]:
    """Return model configs, optionally including hidden model types."""
    if include_hidden:
        return list(ALL_MODELS_CONFIG)

    return [
        model_config
        for model_config in ALL_MODELS_CONFIG
        if model_config.get("parameters", {}).get("model_type") not in HIDDEN_MODEL_TYPES
    ]


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
    Sync a single model to database.
    Prefer lookup by canonical id (matches cloud MQTT), then by (name, version).
    """
    canonical_id = canonical_uuid_for_model_config(model_config)

    if canonical_id is not None:
        by_id = db.query(AiModel).filter(AiModel.id == canonical_id).first()
        if by_id is not None:
            logger.info(
                "Updating model by canonical id %s: %s v%s",
                canonical_id,
                model_config["name"],
                model_config["version"],
            )
            by_id.name = model_config["name"]
            by_id.description = model_config["description"]
            by_id.version = model_config["version"]
            by_id.model_path = model_config["model_path"]
            by_id.parameters = model_config["parameters"]
            by_id.is_active = model_config["is_active"]
            db.commit()
            db.refresh(by_id)
            return by_id

    existing_model = (
        db.query(AiModel)
        .filter(
            AiModel.name == model_config["name"],
            AiModel.version == model_config["version"],
        )
        .first()
    )

    if existing_model is not None:
        if canonical_id is not None and existing_model.id != canonical_id:
            logger.warning(
                "Model %s v%s exists with id %s but canonical id is %s. "
                "Update/delete this row manually so id matches cloud, then re-run sync.",
                model_config["name"],
                model_config["version"],
                existing_model.id,
                canonical_id,
            )
        logger.info(
            "Updating existing model by name+version: %s v%s",
            model_config["name"],
            model_config["version"],
        )
        existing_model.description = model_config["description"]
        existing_model.model_path = model_config["model_path"]
        existing_model.parameters = model_config["parameters"]
        existing_model.is_active = model_config["is_active"]
        db.commit()
        db.refresh(existing_model)
        return existing_model

    logger.info(
        "Creating new model: %s v%s (canonical_id=%s)",
        model_config["name"],
        model_config["version"],
        str(canonical_id) if canonical_id else "random uuid4",
    )
    create_kwargs = dict(
        name=model_config["name"],
        description=model_config["description"],
        version=model_config["version"],
        model_path=model_config["model_path"],
        parameters=model_config["parameters"],
        is_active=model_config["is_active"],
        is_latest_used=False,
    )
    if canonical_id is not None:
        create_kwargs["id"] = canonical_id
    new_model = AiModel(**create_kwargs)
    db.add(new_model)
    db.commit()
    db.refresh(new_model)
    return new_model


def ensure_events_can_detach_from_models(db: Session):
    """
    Allow events to survive when an AI model is deleted.

    Existing databases may still have model_id as NOT NULL with ON DELETE CASCADE,
    so we normalize the column and FK before pruning stale models.
    """
    constraint_rows = db.execute(
        text(
            """
            SELECT tc.constraint_name, rc.delete_rule
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            JOIN information_schema.referential_constraints AS rc
              ON tc.constraint_name = rc.constraint_name
             AND tc.table_schema = rc.constraint_schema
            WHERE tc.table_name = 'events'
              AND tc.constraint_type = 'FOREIGN KEY'
              AND kcu.column_name = 'model_id'
            """
        )
    ).mappings().all()

    needs_fk_update = len(constraint_rows) != 1 or constraint_rows[0]["delete_rule"] != "SET NULL"

    try:
        db.execute(text("ALTER TABLE events ALTER COLUMN model_id DROP NOT NULL"))

        if needs_fk_update:
            for row in constraint_rows:
                db.execute(text(f'ALTER TABLE events DROP CONSTRAINT "{row["constraint_name"]}"'))

            db.execute(
                text(
                    """
                    ALTER TABLE events
                    ADD CONSTRAINT events_model_id_fkey
                    FOREIGN KEY (model_id) REFERENCES ai_models(id) ON DELETE SET NULL
                    """
                )
            )

        db.commit()
    except Exception:
        db.rollback()
        raise


def snapshot_and_detach_events(db: Session, model: AiModel) -> int:
    """Persist model metadata into event payloads, then detach the FK."""
    events = db.query(Event).filter(Event.model_id == model.id).all()
    model_snapshot = {
        "id": str(model.id),
        "name": model.name,
        "version": model.version,
        "model_type": (model.parameters or {}).get("model_type"),
    }

    for event in events:
        event.detection_data = {
            **(event.detection_data or {}),
            "archived_model": model_snapshot,
        }
        event.model_id = None

    return len(events)


def delete_stale_models(
    db: Session,
    configured_model_keys: Set[Tuple[str, str]],
) -> List[Dict]:
    """
    Delete models that are no longer present in MODELS_CONFIG.

    Camera assignments are removed automatically. Historical events are preserved
    by detaching them from the model after storing a lightweight snapshot.
    """
    ensure_events_can_detach_from_models(db)

    existing_models = db.query(AiModel).all()
    deleted_models: List[Dict] = []

    for existing_model in existing_models:
        model_key = (existing_model.name, existing_model.version)
        if model_key in configured_model_keys:
            continue

        model_snapshot = {
            "id": existing_model.id,
            "name": existing_model.name,
            "version": existing_model.version,
        }

        camera_ref_count = db.query(CameraModel).filter(
            CameraModel.model_id == existing_model.id
        ).count()

        logger.info(
            "Deleting stale model: %s v%s",
            model_snapshot["name"],
            model_snapshot["version"],
        )

        try:
            event_ref_count = snapshot_and_detach_events(db, existing_model)
            db.query(CameraModel).filter(
                CameraModel.model_id == existing_model.id
            ).delete(synchronize_session=False)
            db.flush()
            db.query(AiModel).filter(AiModel.id == existing_model.id).delete(
                synchronize_session=False
            )
            db.commit()
            deleted_models.append(
                {
                    "id": model_snapshot["id"],
                    "name": model_snapshot["name"],
                    "version": model_snapshot["version"],
                    "camera_refs": camera_ref_count,
                    "event_refs": event_ref_count,
                }
            )
        except Exception as e:
            db.rollback()
            logger.error(
                "Failed to delete stale model %s v%s: %s",
                model_snapshot["name"],
                model_snapshot["version"],
                e,
                exc_info=True,
            )

    return deleted_models


def sync_all_models(prune_stale: bool = True, include_hidden: bool = False):
    """Synchronize all models to database"""
    logger.info("=" * 60)
    logger.info("Starting AI Models Database Synchronization")
    logger.info("=" * 60)
    logger.info("Include hidden models: %s", include_hidden)

    models_config = build_models_config(include_hidden=include_hidden)

    # Create tables
    create_tables()

    # Open database session
    db = SessionLocal()

    try:
        synced_models = []
        configured_model_keys = {get_model_key(model_config) for model_config in models_config}
        deleted_models = []

        for model_config in models_config:
            try:
                model = sync_model(db, model_config)
                synced_models.append(model)
                logger.info(f"✓ Synced: {model.name} (ID: {model.id})")
            except Exception as e:
                logger.error(f"✗ Failed to sync {model_config['name']}: {e}", exc_info=True)
                db.rollback()

        if prune_stale:
            deleted_models = delete_stale_models(db, configured_model_keys)
            logger.info("Deleted %s stale model(s)", len(deleted_models))
        else:
            logger.info("Skipping stale model deletion (--keep-stale enabled)")

        logger.info("=" * 60)
        logger.info(
            "Synchronization Complete: %s/%s models synced, %s stale model(s) deleted",
            len(synced_models),
            len(models_config),
            len(deleted_models),
        )
        logger.info("=" * 60)

        # Display summary
        logger.info("\nModel Summary:")
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

        if deleted_models:
            logger.info("Deleted stale models:")
            logger.info("-" * 60)
            for model in deleted_models:
                logger.info(
                    "  %s v%s (camera_refs=%s, event_refs=%s)",
                    model["name"],
                    model["version"],
                    model["camera_refs"],
                    model["event_refs"],
                )

    except Exception as e:
        logger.error(f"Critical error during synchronization: {e}", exc_info=True)
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
            logger.info(f"  Description: {model.description}")

        logger.info(f"\nTotal: {len(models)} models")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync AI models to database")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all models in database"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync models to database"
    )
    parser.add_argument(
        "--keep-stale",
        action="store_true",
        help="Keep models that are not present in MODELS_CONFIG",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include model types normally hidden from sync",
    )

    args = parser.parse_args()

    # Default action is sync if no args provided
    if not args.list and not args.sync:
        args.sync = True

    if args.sync:
        sync_all_models(
            prune_stale=not args.keep_stale,
            include_hidden=args.include_hidden,
        )

    if args.list:
        list_models()
