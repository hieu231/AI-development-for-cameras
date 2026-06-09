#!/usr/bin/env python3
"""
Script to update AI models in database with Vietnamese configurations
Updates model parameters to use Vietnamese class names and descriptions

Usage:
    python scripts/update_sync_models_to_db.py
    python scripts/update_sync_models_to_db.py --list  # List current models
    python scripts/update_sync_models_to_db.py --sync  # Sync/update models
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from src.database import SessionLocal, engine
from src.models.ai_model import AiModel
from src.models.base import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Updated models config with Vietnamese values
MODELS_CONFIG_VIETNAMESE = [
    {
        "name": "Trang phục bảo hộ EVN & Smartech",
        "description": "Phát hiện và ghi nhận nhân viên EVN, Smartech và người không mặc áo bảo hộ trong khu vực giám sát",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/ppe_100ep.pt",
        "parameters": {
            "model_type": "evn_smartech",
            "conf_threshold": 0.55,
            "classes": [0, 1, 2],
            "class_names": ["Nhân viên EVN", "Nhân viên Smartech", "Không áo bảo hộ"],
            "event_type": "An toàn lao động"
        },
        "is_active": True,
    },
    {
        "name": "An toàn lao động",
        "description": "Phát hiện vi phạm không mặc áo bảo hộ (NO-Vest)",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/atld_92.pt",
        "parameters": {
            "model_type": "helmet_detection",
            "conf_threshold": 0.55,
            "person_conf_threshold": 0.6,
            "classes": [0, 1, 2, 3, 4],
            "class_names": ["Đội mũ bảo hộ", "Không đội mũ bảo hộ", "Không mặc áo bảo hộ", "Người", "Mặc áo bảo hộ"],
            "event_type": "An toàn lao động"
        },
        "is_active": True,
    },
    {
        "name": "Tràn dầu kho",
        "description": "Phát hiện vết tràn dầu trên mặt đất",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/oil_11l_100ep.pt",
        "parameters": {
            "model_type": "oil_spill",
            "conf_threshold": 0.7,
            "iou_threshold": 0.3,
            "class_names": ["Tràn dầu"],
            "event_type": "Tràn dầu kho"
        },
        "is_active": True,
    },
    {
        "name": "Báo cháy",
        "description": "Phát hiện khói và lửa",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/YOLOv10-FireSmoke-X.pt",
        "parameters": {
            "model_type": "smoke_fire",
            "conf_threshold": 0.7,
            "iou_threshold": 0.3,
            "class_names": ["Lửa", "Khói"],
            "event_type": "Báo cháy"
        },
        "is_active": True,
    },
    {
        "name": "Phát hiện đối tượng (YOLOv8)",
        "description": "Phát hiện đối tượng tổng quát sử dụng YOLOv8/YOLOv11",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolov8m.pt",
        "parameters": {
            "model_type": "yolo",
            "conf_threshold": 0.7,
            "iou_threshold": 0.3,
            "min_detections": 1,
            "track_by_class": True,
            "event_type": "Phát hiện đối tượng"
        },
        "is_active": True,
    },
    {
        "name": "Nhận diện khuôn mặt",
        "description": "Phát hiện và nhận diện khuôn mặt trong luồng video",
        "version": "1.0.0",
        "model_path": None,
        "parameters": {
            "model_type": "face_recognition",
            "face_similarity_threshold": 0.45,
            "known_cooldown": 300,
            "unknown_cooldown": 60,
            "event_type": "Nhận diện khuôn mặt"
        },
        "is_active": True,
    },
    {
        "name": "Xả đáy dầu",
        "description": "Phát hiện người đứng lâu trong khu vực cấm (nghi ngờ xả đáy dầu)",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11n.pt",
        "parameters": {
            "model_type": "oil_dumping",
            "conf_threshold": 0.5,
            "detection_time_threshold": 20,
            "detection_cooldown": 28800,
            "class_names": ["Người"],
            "event_type": "Xả đáy dầu"
        },
        "is_active": True,
    },
    {
        "name": "Người vào khu vực cấm",
        "description": "Phát hiện người vào khu vực kho/khu vực hạn chế",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11m.pt",
        "parameters": {
            "model_type": "people_control",
            "conf_threshold": 0.6,
            "detection_cooldown": 300,
            "class_names": ["Người"],
            "event_type": "Người vào khu vực cấm"
        },
        "is_active": True,
    },
    {
        "name": "Nhận diện biển số",
        "description": "Nhận diện biển số xe tự động sử dụng fast-alpr",
        "version": "1.0.0",
        "model_path": None,
        "parameters": {
            "model_type": "alpr",
            "buffer_size": 5,
            "similarity_threshold": 0.7,
            "inactivity_timeout": 60,
            "vehicle_types": ["Xe Bồn", "Xe Con"],
            "event_type": "Nhận diện biển số"
        },
        "is_active": True,
    },
    {
        "name": "Nhận diện biển số (Gemini)",
        "description": "Phát hiện xe và nhận diện biển số sử dụng Gemini AI",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/vehicle_detection.pt",
        "parameters": {
            "model_type": "gemini_plate",
            "detection_cooldown": 10,
            "frame_width": 1280,
            "frame_height": 720,
            "event_type": "Nhận diện biển số (Gemini)"
        },
        "is_active": True,
    },
    {
        "name": "Xe qua cổng",
        "description": "Phát hiện xe qua cổng, bãi đỗ, khu vực hạn chế",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolov8m.pt",
        "parameters": {
            "model_type": "vehicle_gate",
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "min_vehicles_to_alert": 1,
            "class_names": ["Xe ô tô", "Xe tải", "Xe buýt", "Xe máy", "Xe đạp", "Tàu hỏa"],
            "event_type": "Xe qua cổng"
        },
        "is_active": True,
    },
    {
        "name": "Vi phạm mũ bảo hộ",
        "description": "Phát hiện vi phạm PPE: không đội mũ bảo hộ",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/atld_92.pt",
        "parameters": {
            "model_type": "ppe_helmet",
            "conf_threshold": 0.45,
            "no_hardhat_threshold": 0.45,
            "class_names": ["Không đội mũ bảo hộ", "Đội mũ bảo hộ"],
            "event_type": "Vi phạm mũ bảo hộ"
        },
        "is_active": True,
    },
    {
        "name": "Giám sát khu vực làm việc",
        "description": "Giám sát khu vực làm việc: đếm người, phát hiện vi phạm PPE, nhận diện người lạ",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/yolo11m.pt",
        "parameters": {
            "model_type": "workspace_monitor",
            "conf_threshold": 0.4,
            "ppe_conf_threshold": 0.45,
            "face_similarity_threshold": 0.45,
            "max_workers": 5,
            "detection_cooldown": 300,
            "event_type": "Giám sát khu vực làm việc"
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
            "event_type": "Dòng chảy tràn dầu đột ngột"
        },
        "is_active": True,
    },
    {
        "name": "Giam sat nap dau",
        "description": "Phat hien trang thai dong hoac mo nap dau, mac dinh canh bao khi nap dau dang mo",
        "version": "1.0.0",
        "model_path": "src/ai_models/model_weights/oil_cap_detection.pt",
        "parameters": {
            "model_type": "oil_cap_detection",
            "conf_threshold": 0.45,
            "iou_threshold": 0.45,
            "detection_cooldown": 300,
            "violation_min_center_y_ratio": 0.35,
            "violation_labels": ["oil_cap_opened"],
            "event_type": "Giam sat nap dau"
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
            "class_names": ["Hút thuốc"],
            "event_type": "Phát hiện hành vi hút thuốc"
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
    "vehicle_gate": {
        "name": "Phương tiện ra/vào khu vực",
        "description": "Phát hiện và ghi nhận phương tiện ra/vào khu vực giám sát",
        "parameters": {
            "event_type": "Phương tiện ra/vào khu vực",
        },
    },
    "oil_cap_detection": {
        "name": "Giam sat nap dau",
        "description": "Phat hien trang thai dong hoac mo nap dau, mac dinh canh bao khi nap dau dang mo",
        "parameters": {
            "event_type": "Giam sat nap dau",
            "class_names": ["Nap dau mo", "Nap dau dong"],
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
            "class_names": ["Hút thuốc"],
            "event_type": "Phát hiện hành vi hút thuốc",
        },
    },
}

for model_config in MODELS_CONFIG_VIETNAMESE:
    parameters = model_config.setdefault("parameters", {})
    overrides = STANDARDIZED_MODEL_OVERRIDES.get(parameters.get("model_type"))
    if not overrides:
        continue

    for field in ("name", "description"):
        if field in overrides:
            model_config[field] = overrides[field]

    for key, value in overrides.get("parameters", {}).items():
        parameters[key] = value

# Mapping from old names to new names for update
NAME_MAPPING = {
    "Trang phục bảo hộ": "An toàn lao động",
    "Oil Spill Detection": "Tràn dầu kho",
    "Smoke and Fire Detection": "Báo cháy",
    "General Object Detection (YOLOv8)": "Phát hiện đối tượng (YOLOv8)",
    "Face Recognition": "Nhận diện khuôn mặt",
    "Oil Dumping Detection": "Xả đáy dầu",
    "People Control": "Người vào khu vực cấm",
    "Hàng rào ảo": "Người vào khu vực cấm",  # Also map old Vietnamese name
    "ALPR (License Plate Recognition)": "Nhận diện biển số",
    "Gemini Plate Detection": "Nhận diện biển số (Gemini)",
}


NAME_MAPPING.update(
    {
        "An toàn lao động": "An toàn lao động",
        "Bảo hộ lao động": "Bảo hộ lao động",
        "Nhân viên dầu khí": "Bảo hộ lao động",
        "Oil Spill Detection": "Vết loang bất thường",
        "Tràn dầu kho": "Vết loang bất thường",
        "Smoke and Fire Detection": "Phát hiện khói/cháy",
        "Báo cháy": "Phát hiện khói/cháy",
        "Oil Dumping Detection": "Phát hiện xả đáy",
        "Xả đáy dầu": "Phát hiện xả đáy",
        "People Control": "Xâm nhập trái phép khu vực",
        "Hàng rào ảo": "Xâm nhập trái phép khu vực",
        "Người vào khu vực cấm": "Xâm nhập trái phép khu vực",
        "Xe qua cổng": "Phương tiện ra/vào khu vực",
        "Dòng chảy tràn dầu đột ngột": "Phát hiện tràn dầu",
        "Phát hiện hành vi hút thuốc": "Phát hiện hành vi hút thuốc",
    }
)


def create_tables():
    """Create database tables if they don't exist"""
    logger.info("Creating database tables if needed...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready")


def update_model(db: Session, model_config: dict) -> AiModel:
    """
    Update or create a model in database with Vietnamese config
    """
    # Try to find by new name first
    existing_model = db.query(AiModel).filter(
        AiModel.name == model_config["name"],
        AiModel.version == model_config["version"]
    ).first()

    # If not found, try to find by old name
    if not existing_model:
        for old_name, new_name in NAME_MAPPING.items():
            if new_name == model_config["name"]:
                existing_model = db.query(AiModel).filter(
                    AiModel.name == old_name,
                    AiModel.version == model_config["version"]
                ).first()
                if existing_model:
                    break

    if existing_model:
        # Update existing model
        logger.info(f"Cập nhật model: {existing_model.name} → {model_config['name']}")
        existing_model.name = model_config["name"]
        existing_model.description = model_config["description"]
        existing_model.model_path = model_config["model_path"]
        existing_model.parameters = model_config["parameters"]
        existing_model.is_active = model_config["is_active"]
        db.commit()
        db.refresh(existing_model)
        return existing_model
    else:
        # Create new model
        logger.info(f"Tạo mới model: {model_config['name']} v{model_config['version']}")
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
    """Synchronize all models with Vietnamese configurations"""
    logger.info("=" * 60)
    logger.info("CẬP NHẬT AI MODELS VỚI CẤU HÌNH TIẾNG VIỆT")
    logger.info("=" * 60)

    create_tables()
    db = SessionLocal()

    try:
        synced_models = []

        for model_config in MODELS_CONFIG_VIETNAMESE:
            try:
                model = update_model(db, model_config)
                synced_models.append(model)
                logger.info(f"✓ Đã cập nhật: {model.name} (ID: {model.id})")
            except Exception as e:
                logger.error(f"✗ Lỗi khi cập nhật {model_config['name']}: {e}", exc_info=True)
                db.rollback()

        logger.info("=" * 60)
        logger.info(f"Hoàn thành: {len(synced_models)}/{len(MODELS_CONFIG_VIETNAMESE)} models đã cập nhật")
        logger.info("=" * 60)

        # Display summary
        logger.info("\nTóm tắt Models:")
        logger.info("-" * 60)
        for model in synced_models:
            model_type = model.parameters.get('model_type', 'N/A')
            event_type = model.parameters.get('event_type', 'N/A')
            class_names = model.parameters.get('class_names', [])
            status = "HOẠT ĐỘNG" if model.is_active else "TẮT"
            
            logger.info(f"  📌 {model.name}")
            logger.info(f"      - ID: {model.id}")
            logger.info(f"      - Loại: {model_type}")
            logger.info(f"      - Event Type: {event_type}")
            if class_names:
                logger.info(f"      - Class Names: {', '.join(class_names)}")
            logger.info(f"      - Trạng thái: {status}")
            logger.info("")

    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng: {e}", exc_info=True)
        db.rollback()
    finally:
        db.close()


def list_models():
    """List all models in database"""
    logger.info("=" * 60)
    logger.info("DANH SÁCH MODELS TRONG DATABASE")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        models = db.query(AiModel).order_by(AiModel.name, AiModel.version).all()

        if not models:
            logger.info("Không tìm thấy model nào trong database")
            return

        for model in models:
            model_type = model.parameters.get('model_type', 'N/A')
            event_type = model.parameters.get('event_type', 'N/A')
            class_names = model.parameters.get('class_names', [])
            status = "HOẠT ĐỘNG" if model.is_active else "TẮT"

            logger.info(f"\n📌 {model.name} v{model.version}")
            logger.info(f"   ID: {model.id}")
            logger.info(f"   Loại: {model_type}")
            logger.info(f"   Event Type: {event_type}")
            if class_names:
                logger.info(f"   Class Names: {', '.join(class_names)}")
            logger.info(f"   Trạng thái: {status}")
            logger.info(f"   Mô tả: {model.description}")

        logger.info(f"\nTổng cộng: {len(models)} models")

    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cập nhật AI models với cấu hình tiếng Việt")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Liệt kê tất cả models trong database"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Đồng bộ/cập nhật models vào database"
    )

    args = parser.parse_args()

    # Default action is sync if no args provided
    if not args.list and not args.sync:
        args.sync = True

    if args.sync:
        sync_all_models()

    if args.list:
        list_models()
