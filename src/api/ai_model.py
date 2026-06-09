# src/api/ai_model.py
from fastapi import APIRouter, Query, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import logging

from fastapi.responses import JSONResponse
import cv2
import numpy as np

from src.models.ai_model import AiModel
from src.models.camera_model import CameraModel
from src.database import get_db
from src.core.thread_manager import thread_manager
from src.core.model_factory import ModelFactory
from src.core.event_storage import event_storage_service
from src.core.websocket_manager import websocket_manager
from src.utils.model_validator import ModelValidator
from src.utils.parameter_defaults import get_default_additional_parameters
from src.ai_models.base_model import DetectionResult, AlertInfo

logger = logging.getLogger(__name__)

# Pydantic schemas
class AiModelCreate(BaseModel):
    name: str
    description: str
    version: str
    model_path: Optional[str] = None
    parameters: dict  # model_type stored in here
    is_active: bool
    is_latest_used: bool
    previous_used_version_id: Optional[UUID] = None
    changelog: Optional[str] = None


class AiModelAddVersion(BaseModel):
    """Schema for adding a new version to an existing model"""
    version: str
    model_path: Optional[str] = None
    description: Optional[str] = None
    changelog: Optional[str] = None
    copy_to_weights_dir: bool = True  # If True, copy model to model_weights directory

class AiModelResponse(BaseModel):
    id: UUID
    name: str
    description: str
    version: str
    model_path: Optional[str] = None
    parameters: dict
    is_active: bool
    is_latest_used: bool
    previous_used_version_id: Optional[UUID]
    changelog: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}

router = APIRouter(prefix="/ai_models", tags=["ai_models"])

@router.get("/help")
def get_help():
    """Get API documentation for AI models management"""
    return {
        "message": "API for AI models management with versioning support",
        "endpoints": {
            "basic_operations": {
                "get_all": "GET /ai_models/get_all_models - Get all ai models",
                "get_by_id": "GET /ai_models/get_model_by_id/{ai_model_id} - Get a ai model by ID",
                "create": "POST /ai_models/create_new_model - Create a new AI model (first version)",
            },
            "camera_assignment": {
                "assign": "PUT /ai_models/assing_ai_model_to_camera/{camera_id}/{ai_model_id} - Enable AI model for a camera",
                "get_by_camera": "GET /ai_models/get_model_by_camera_id/{camera_id} - Get enabled ai models by camera ID",
                "remove": "PUT /ai_models/remove_ai_model_from_camera/{camera_id}/{ai_model_id} - Disable AI model from a camera",
            },
            "activation_control": {
                "activate": "PUT /ai_models/activate_model/{ai_model_id} - Activate AI model (set is_active to True)",
                "deactivate": "PUT /ai_models/deactivate_model/{ai_model_id} - Deactivate AI model (disable from all cameras and reload)",
            },
            "version_management": {
                "add_version": "POST /ai_models/add_version/{base_model_id} - Add a new version to an existing model (validates, copies to weights dir, updates cameras)",
                "list_versions": "GET /ai_models/versions/{model_name} - List all versions of a model by name",
                "switch_version": "PUT /ai_models/switch_version/{target_version_id} - Switch to a specific version (updates all cameras using this model)",
                "delete_version": "DELETE /ai_models/delete_version/{version_id} - Delete a version (cannot delete active or only version)",
            }
        },
        "version_workflow": {
            "1_create_first_version": "POST /ai_models/create_new_model with model data",
            "2_add_new_version": "POST /ai_models/add_version/{model_id} with new model_path and version",
            "3_list_versions": "GET /ai_models/versions/{model_name} to see all versions",
            "4_switch_if_needed": "PUT /ai_models/switch_version/{version_id} to rollback or test",
            "5_delete_old": "DELETE /ai_models/delete_version/{version_id} to cleanup"
        }
    }

@router.get("/get_all_models", response_model=List[AiModelResponse])
def get_ai_models(skip: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1), db: Session = Depends(get_db)):
    """Lấy danh sách ai models"""
    ai_models = db.query(AiModel).offset(skip).limit(limit).all()
    return ai_models

@router.get("/get_model_by_id/{ai_model_id}", response_model=AiModelResponse)
def get_model_by_id(ai_model_id: UUID, db: Session = Depends(get_db)):
    """Lấy ai model theo ID"""
    ai_model = db.query(AiModel).filter(AiModel.id == ai_model_id).first()
    if not ai_model:
        raise HTTPException(status_code=404, detail="AI Model not found")
    return ai_model

@router.post("/create_new_model", response_model=AiModelResponse, status_code=status.HTTP_201_CREATED)
def create_new_model(ai_model: AiModelCreate, db: Session = Depends(get_db)):
    """Tạo ai model mới"""
    db_ai_model = AiModel(**ai_model.model_dump())
    db.add(db_ai_model)
    db.commit()
    db.refresh(db_ai_model)
    return db_ai_model

@router.get("/get_model_by_camera_id/{camera_id}", response_model=List[AiModelResponse])
def get_model_by_camera_id(camera_id: UUID, db: Session = Depends(get_db)):
    """Lấy ai model theo ID của camera (chỉ các model đang enabled)"""
    camera_models = db.query(CameraModel).filter(
        CameraModel.camera_id == camera_id,
        CameraModel.is_enabled == True
    ).options(joinedload(CameraModel.ai_model)).all()
    
    ai_models = [
        cm.ai_model
        for cm in camera_models
        if cm.ai_model is not None and cm.ai_model.is_active
    ]
    return ai_models

@router.put("/assing_ai_model_to_camera/{camera_id}/{ai_model_id}", response_model=AiModelResponse)
def assing_ai_model_to_camera(camera_id: UUID, ai_model_id: UUID, db: Session = Depends(get_db)):
    """Gán ai model cho camera"""
    db_ai_model = db.query(AiModel).filter(AiModel.id == ai_model_id).first()
    if not db_ai_model:
        raise HTTPException(status_code=404, detail="AI Model not found")

    db_camera_model = db.query(CameraModel).filter(
        CameraModel.camera_id == camera_id,
        CameraModel.model_id == ai_model_id
    ).first()

    if not db_camera_model:
        default_additional_params = get_default_additional_parameters(db_ai_model.parameters.get("model_type"))

        db_camera_model = CameraModel(
            camera_id=camera_id,
            model_id=ai_model_id,
            is_enabled=True,
            additional_parameters=default_additional_params
        )
        db.add(db_camera_model)
        flag_modified(db_camera_model, "additional_parameters")
        logger.info(f"Created camera-model relationship with default additional_parameters: {default_additional_params}")
    else:
        db_camera_model.is_enabled = True

        current_params = db_camera_model.additional_parameters or {}
        default_params = get_default_additional_parameters(db_ai_model.parameters.get("model_type"))

        updated = False
        for key, default_value in default_params.items():
            if key not in current_params or current_params[key] is None:
                current_params[key] = default_value
                updated = True

        if updated:
            db_camera_model.additional_parameters = current_params
            flag_modified(db_camera_model, "additional_parameters")
            logger.info(f"Updated additional_parameters: {current_params}")

    db.commit()

    thread_manager.reload_models(camera_id)

    return db_ai_model

@router.put("/remove_ai_model_from_camera/{camera_id}/{ai_model_id}", response_model=AiModelResponse)
def stop_ai_model(camera_id: UUID, ai_model_id: UUID, db: Session = Depends(get_db)):
    """Xóa ai model khỏi camera"""
    db_camera_model = db.query(CameraModel).filter(
        CameraModel.camera_id == camera_id,
        CameraModel.model_id == ai_model_id
    ).first()

    if not db_camera_model:
        raise HTTPException(status_code=404, detail="Camera-Model relationship not found")

    db_camera_model.is_enabled = False
    db.commit()

    thread_manager.reload_models(camera_id)

    db_ai_model = db.query(AiModel).filter(AiModel.id == ai_model_id).first()
    return db_ai_model

@router.put("/activate_model/{ai_model_id}", response_model=AiModelResponse)
def activate_model(ai_model_id: UUID, db: Session = Depends(get_db)):
    """Kích hoạt AI model theo ID (set is_active = True)"""
    db_ai_model = db.query(AiModel).filter(AiModel.id == ai_model_id).first()
    
    if not db_ai_model:
        raise HTTPException(status_code=404, detail="AI Model not found")
    
    db_ai_model.is_active = True
    db.commit()
    db.refresh(db_ai_model)
    
    return db_ai_model

@router.put("/deactivate_model/{ai_model_id}", response_model=AiModelResponse)
def deactivate_model(ai_model_id: UUID, db: Session = Depends(get_db)):
    """Vô hiệu hóa AI model theo ID (set is_active = False) và remove khỏi tất cả cameras"""
    db_ai_model = db.query(AiModel).filter(AiModel.id == ai_model_id).first()
    
    if not db_ai_model:
        raise HTTPException(status_code=404, detail="AI Model not found")
    
    db_ai_model.is_active = False
    
    camera_models = db.query(CameraModel).filter(
        CameraModel.model_id == ai_model_id,
        CameraModel.is_enabled == True
    ).all()
    
    affected_camera_ids = []
    for camera_model in camera_models:
        camera_model.is_enabled = False
        affected_camera_ids.append(camera_model.camera_id)
    
    db.commit()
    db.refresh(db_ai_model)
    
    for camera_id in affected_camera_ids:
        thread_manager.reload_models(camera_id)
    
    return db_ai_model

@router.post("/detect")
async def detect_frame(file: UploadFile = File(...), model_name: str = "all"):
    """
    Phát hiện và trả về cảnh báo
    
    Response:
        {
            "status": "success" | "no_alert",
            "alert": {
                "level": "high" | "low" | null,
                "color": "#FF0000" | "#FFA500" | "#FFFF00" | null,
                "message": "...",
                "confidence": 0.95,
                "detected_objects": [...]
            },
            "timestamp": "2025-12-01T12:00:00"
        }
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        alert: AlertInfo = ModelFactory.detect(frame, model_name)
        
        if alert is None:
            return {
                "status": "no_alert",
                "alert": None,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "alert": {
                "level": alert.level.value,
                "color": alert.color.value,
                "message": alert.message,
                "confidence": alert.confidence,
                "detected_objects": alert.detected_objects
            },
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/camera/{camera_id}")
async def detect_from_camera(camera_id: str, model_name: str = "all"):
    """
    Phát hiện từ camera live stream
    Trả về cảnh báo nếu có
    """
    raise HTTPException(
        status_code=501,
        detail="detect_from_camera endpoint is not implemented in this refactor; use camera threads instead.",
    )

@router.post("/add_version/{base_model_id}", response_model=AiModelResponse, status_code=status.HTTP_201_CREATED)
def add_new_version(
    base_model_id: UUID,
    version_data: AiModelAddVersion,
    db: Session = Depends(get_db)
):
    """
    Add a new version to an existing AI model

    This creates a new model entry with the same name but different version.
    - Validates model path exists and can be loaded
    - Optionally copies model to model_weights directory
    - Sets new version as active and latest_used
    - Deactivates previous versions
    - Updates all cameras using the old version to use the new version
    - Restarts affected camera threads

    Args:
        base_model_id: ID of any existing version of the model
        version_data: New version data (version, model_path, description, changelog)

    Returns:
        The newly created model version
    """
    base_model = db.query(AiModel).filter(AiModel.id == base_model_id).first()
    if not base_model:
        raise HTTPException(status_code=404, detail="Base model not found")

    model_type = base_model.parameters.get('model_type', 'object_detection')

    is_valid, error_msg = ModelValidator.validate_model_path(version_data.model_path, model_type)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Model validation failed: {error_msg}"
        )

    final_model_path = version_data.model_path
    if version_data.copy_to_weights_dir:
        success, new_path, error_msg = ModelValidator.copy_model_to_weights_dir(
            version_data.model_path,
            base_model.name,
            version_data.version
        )
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to copy model to weights directory: {error_msg}"
            )
        final_model_path = new_path
        logger.info(f"Model copied to: {final_model_path}")

    existing_version = db.query(AiModel).filter(
        AiModel.name == base_model.name,
        AiModel.version == version_data.version
    ).first()
    if existing_version:
        raise HTTPException(
            status_code=400,
            detail=f"Version '{version_data.version}' already exists for model '{base_model.name}'"
        )

    current_active = db.query(AiModel).filter(
        AiModel.name == base_model.name,
        AiModel.is_active == True
    ).first()

    new_version = AiModel(
        name=base_model.name,
        description=version_data.description or base_model.description,
        version=version_data.version,
        model_path=final_model_path,
        parameters=base_model.parameters,
        is_active=True,
        is_latest_used=True,
        previous_used_version_id=current_active.id if current_active else None,
        changelog=version_data.changelog or f"Added version {version_data.version}"
    )
    db.add(new_version)
    db.flush()

    db.query(AiModel).filter(
        AiModel.name == base_model.name,
        AiModel.id != new_version.id
    ).update({
        "is_active": False,
        "is_latest_used": False
    }, synchronize_session=False)

    old_camera_models = db.query(CameraModel).join(AiModel).filter(
        AiModel.name == base_model.name,
        CameraModel.is_enabled == True
    ).all()

    affected_camera_ids = set()
    for camera_model in old_camera_models:
        camera_model.model_id = new_version.id
        affected_camera_ids.add(camera_model.camera_id)

    db.commit()
    db.refresh(new_version)

    for camera_id in affected_camera_ids:
        try:
            thread_manager.reload_models(camera_id)
            logger.info(f"Reloaded models for camera {camera_id}")
        except Exception as e:
            logger.warning(f"Failed to reload models for camera {camera_id}: {e}")

    logger.info(
        f"Added new version '{version_data.version}' for model '{base_model.name}', "
        f"updated {len(affected_camera_ids)} cameras"
    )

    return new_version


@router.get("/versions/{model_name}", response_model=List[AiModelResponse])
def list_model_versions(
    model_name: str,
    include_inactive: bool = Query(default=True, description="Include inactive versions"),
    db: Session = Depends(get_db)
):
    """
    List all versions of a model by model name

    Args:
        model_name: Name of the model
        include_inactive: Whether to include inactive versions

    Returns:
        List of all versions, ordered by creation date (newest first)
    """
    query = db.query(AiModel).filter(AiModel.name == model_name)

    if not include_inactive:
        query = query.filter(AiModel.is_active == True)

    versions = query.order_by(AiModel.created_at.desc()).all()

    if not versions:
        raise HTTPException(status_code=404, detail=f"No versions found for model '{model_name}'")

    return versions


@router.put("/switch_version/{target_version_id}", response_model=AiModelResponse)
def switch_to_version(
    target_version_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Switch to a specific version of a model

    This will:
    - Set target version as active and latest_used
    - Deactivate all other versions of the same model
    - Update all cameras using this model to use the target version
    - Restart affected camera threads

    Args:
        target_version_id: ID of the version to switch to

    Returns:
        The activated model version
    """
    target_version = db.query(AiModel).filter(AiModel.id == target_version_id).first()
    if not target_version:
        raise HTTPException(status_code=404, detail="Target version not found")

    model_type = target_version.parameters.get('model_type', 'object_detection')
    is_valid, error_msg = ModelValidator.validate_model_path(target_version.model_path, model_type)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Target version model file is invalid or missing: {error_msg}"
        )

    current_active = db.query(AiModel).filter(
        AiModel.name == target_version.name,
        AiModel.is_active == True
    ).first()

    if current_active and current_active.id != target_version_id:
        target_version.previous_used_version_id = current_active.id

    target_version.is_active = True
    target_version.is_latest_used = True

    db.query(AiModel).filter(
        AiModel.name == target_version.name,
        AiModel.id != target_version_id
    ).update({
        "is_active": False,
        "is_latest_used": False
    }, synchronize_session=False)

    old_camera_models = db.query(CameraModel).join(AiModel).filter(
        AiModel.name == target_version.name,
        CameraModel.is_enabled == True
    ).all()

    affected_camera_ids = set()
    for camera_model in old_camera_models:
        camera_model.model_id = target_version.id
        affected_camera_ids.add(camera_model.camera_id)

    db.commit()
    db.refresh(target_version)

    for camera_id in affected_camera_ids:
        try:
            thread_manager.reload_models(camera_id)
            logger.info(f"Reloaded models for camera {camera_id}")
        except Exception as e:
            logger.warning(f"Failed to reload models for camera {camera_id}: {e}")

    logger.info(
        f"Switched to version '{target_version.version}' for model '{target_version.name}', "
        f"updated {len(affected_camera_ids)} cameras"
    )

    return target_version


@router.delete("/delete_version/{version_id}", response_model=dict)
def delete_model_version(
    version_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a specific version of a model

    Rules:
    - Cannot delete the currently active version
    - If deleted version is the latest_used, the most recent remaining version becomes latest_used
    - If no other versions exist, deletion is not allowed

    Args:
        version_id: ID of the version to delete

    Returns:
        Status message with replacement version info
    """
    version_to_delete = db.query(AiModel).filter(AiModel.id == version_id).first()
    if not version_to_delete:
        raise HTTPException(status_code=404, detail="Version not found")

    if version_to_delete.is_active:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the currently active version. Switch to another version first."
        )

    other_versions = db.query(AiModel).filter(
        AiModel.name == version_to_delete.name,
        AiModel.id != version_id
    ).order_by(AiModel.created_at.desc()).all()

    if not other_versions:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the only version of this model. At least one version must remain."
        )

    model_name = version_to_delete.name
    deleted_version = version_to_delete.version

    db.delete(version_to_delete)

    if version_to_delete.is_latest_used and other_versions:
        most_recent = other_versions[0]
        most_recent.is_latest_used = True

    db.commit()

    logger.info(f"Deleted version '{deleted_version}' of model '{model_name}'")

    return {
        "message": f"Successfully deleted version '{deleted_version}' of model '{model_name}'",
        "deleted_version": deleted_version,
        "model_name": model_name,
        "remaining_versions": len(other_versions)
    }
