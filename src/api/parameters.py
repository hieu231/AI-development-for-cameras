# src/api/parameters.py
"""
Parameter Management API
Endpoints for managing AI model parameters (general and camera-specific)
"""

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from typing import Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, validator
import logging

from src.database import get_db
from src.models.ai_model import AiModel
from src.models.camera_model import CameraModel
from src.models.camera import Camera
from src.utils.parameter_validator import (
    validate_general_parameters,
    validate_additional_parameters,
    get_parameter_documentation,
)
from src.utils.parameter_defaults import get_default_additional_parameters

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/parameters", tags=["Parameters"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


async def restart_cameras_background(camera_ids: List[UUID]):
    """
    Restart cameras in background after parameter update

    Args:
        camera_ids: List of camera UUIDs to restart
    """
    try:
        from src.core.thread_manager import thread_manager
        import asyncio
        import time

        logger.info(f"Restarting {len(camera_ids)} cameras after parameter update...")

        for camera_id in camera_ids:
            try:
                logger.info(f"Restarting camera {camera_id}...")

                # Stop camera
                thread_manager.stop_camera(camera_id)

                # Small delay to ensure cleanup
                await asyncio.sleep(0.5)

                # Start camera (show_display=False for server mode)
                thread_manager.start_camera(camera_id, show_display=False)

                logger.info(f"Camera {camera_id} restarted successfully")

            except Exception as e:
                logger.error(
                    f"Failed to restart camera {camera_id}: {e}", exc_info=True
                )

        logger.info(f"Completed restarting {len(camera_ids)} cameras")

    except Exception as e:
        logger.error(f"Error in restart_cameras_background: {e}", exc_info=True)


# Pydantic models for request/response
class GeneralParametersUpdate(BaseModel):
    """Request model for updating general model parameters"""

    parameters: Dict[str, Any] = Field(
        ...,
        description="General parameters to update (stored in ai_models.parameters)",
        example={"conf_threshold": 0.6, "stranger_conf_threshold": 0.4},
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "parameters": {
                        "conf_threshold": 0.5,
                        "iou_threshold": 0.45,
                        "detection_cooldown": 15,
                    }
                },
                {"parameters": {"conf_threshold": 0.6, "stranger_conf_threshold": 0.4}},
                {
                    "parameters": {
                        "conf_threshold": 0.4,
                        "ppe_conf_threshold": 0.45,
                        "face_similarity_threshold": 0.45,
                        "max_workers": 5,
                        "detection_cooldown": 300,
                    }
                },
            ]
        }


class AdditionalParametersUpdate(BaseModel):
    """Request model for updating camera-specific additional parameters"""

    additional_parameters: Dict[str, Any] = Field(
        ...,
        description="Additional parameters for this camera (stored in camera_models.additional_parameters)",
        example={"roi": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]]},
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "additional_parameters": {
                        "roi": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
                        "conf_threshold": 0.45,
                    }
                },
                {
                    "additional_parameters": {
                        "roi": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]]
                    }
                },
                {
                    "additional_parameters": {
                        "roi": [
                            [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                            [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]],
                        ],
                        "max_roi_polygons": 2,
                    }
                },
                {
                    "additional_parameters": {
                        "roi": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                        "max_roi_polygons": 1,
                        "roi_edit_mode": True,
                        "registered_employee_ids": [
                            "EMP001",
                            "EMP002",
                            "EMP003",
                            "EMP004",
                            "EMP005",
                        ],
                        "registered_worker_count": 5,
                    }
                },
            ]
        }


class ParametersResponse(BaseModel):
    """Response model for parameter queries"""

    model_id: str
    model_name: str
    model_type: str
    general_parameters: Dict[str, Any]
    additional_parameters: Dict[str, Any] = {}
    camera_id: str = None

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_name": "Helmet Detection ATLD",
                "model_type": "helmet_detection",
                "general_parameters": {
                    "model_type": "helmet_detection",
                    "iou_threshold": 0.45,
                    "max_det": 50,
                },
                "additional_parameters": {
                    "conf_threshold": 0.45,
                    "roi": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
                    "max_roi_polygons": 1,
                },
                "camera_id": "456e4567-e89b-12d3-a456-426614174001",
            }
        }


# ============================================================================
# GENERAL PARAMETERS ENDPOINTS (ai_models.parameters)
# ============================================================================


@router.get(
    "/general/{model_id}",
    response_model=Dict[str, Any],
    summary="Get general parameters for a model",
    description="""
    Get general parameters for an AI model.

    General parameters are stored in `ai_models.parameters` and apply to all cameras using this model.

    **Returns:**
    - `model_id`: Model UUID
    - `model_name`: Name of the model
    - `model_type`: Type of model (e.g., helmet_detection, people_control, evn_smartech, tran_dau, oil_cap_detection, smoking_behavior)
    - `parameters`: General parameters object

    **Example Response (helmet_detection):**
    ```json
    {
        "model_id": "123e4567-e89b-12d3-a456-426614174000",
        "model_name": "Helmet Detection ATLD",
        "model_type": "helmet_detection",
        "parameters": {
            "model_type": "helmet_detection",
            "iou_threshold": 0.45,
            "max_det": 50,
            "min_detections": 1
        }
    }
    ```

    **Example Response (evn_smartech):**
    ```json
    {
        "model_id": "789e4567-e89b-12d3-a456-426614174000",
        "model_name": "EVN Smartech Face",
        "model_type": "evn_smartech",
        "parameters": {
            "model_type": "evn_smartech",
            "conf_threshold": 0.6,
            "stranger_conf_threshold": 0.4
        }
    }
    ```
    """,
)
async def get_general_parameters(model_id: UUID, db: Session = Depends(get_db)):
    """Get general parameters for a model"""
    ai_model = db.query(AiModel).filter(AiModel.id == model_id).first()
    if not ai_model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    return {
        "model_id": str(ai_model.id),
        "model_name": ai_model.name,
        "model_type": ai_model.model_type,
        "parameters": ai_model.parameters or {},
    }


@router.put(
    "/general/{model_id}",
    response_model=Dict[str, Any],
    summary="Update general parameters for a model",
    description="""
    Update general parameters for an AI model.

    **Important:**
    - General parameters apply to ALL cameras using this model
    - Cameras using this model will be automatically restarted to apply changes
    - Parameters are validated based on model_type

    **Request Body:**
    ```json
    {
        "parameters": {
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "detection_cooldown": 15
        }
    }
    ```

    **Parameter Validation:**
    - `conf_threshold`: float, range [0.0, 1.0]
    - `iou_threshold`: float, range [0.0, 1.0]
    - `detection_cooldown`: int, min 0 (seconds)
    - `min_detections`: int, min 1
    - `max_det`: int, min 1
    - `stranger_conf_threshold`: float, range [0.0, 1.0] (evn_smartech only)

    See `/api/parameters/schema/{model_type}` for model-specific allowed parameters.

    **Response:**
    Returns updated model parameters and list of affected cameras that will be restarted.
    """,
)
async def update_general_parameters(
    model_id: UUID,
    update_data: GeneralParametersUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Update general parameters for a model"""
    ai_model = db.query(AiModel).filter(AiModel.id == model_id).first()
    if not ai_model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    # Validate parameters
    is_valid, errors = validate_general_parameters(
        ai_model.model_type,
        update_data.parameters,
        strict=False,  # Allow extra parameters for flexibility
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Parameter validation failed", "errors": errors},
        )

    # Merge with existing parameters (create new dict to ensure change detection)
    current_params = dict(ai_model.parameters or {})
    current_params.update(update_data.parameters)
    ai_model.parameters = current_params

    # Mark as modified to ensure SQLAlchemy tracks the JSONB change
    flag_modified(ai_model, "parameters")

    db.commit()
    db.refresh(ai_model)

    # Get list of cameras using this model. Skip soft-deleted cameras —
    # restarting them brings their AI thread back from the dead and
    # makes them emit events as if they were live (the orphan-stream
    # bug we hit when face_recognition params were tuned and a deleted
    # camera came back to life).
    from src.models.camera import Camera

    camera_models = (
        db.query(CameraModel)
        .join(Camera, Camera.id == CameraModel.camera_id)
        .filter(
            CameraModel.model_id == model_id,
            CameraModel.is_enabled == True,  # noqa: E712
            Camera.is_deleted == False,  # noqa: E712
        )
        .all()
    )

    affected_camera_ids = [cm.camera_id for cm in camera_models]

    # Schedule camera restart in background
    if affected_camera_ids:
        background_tasks.add_task(restart_cameras_background, affected_camera_ids)
        logger.info(f"Scheduled restart for {len(affected_camera_ids)} cameras")

    return {
        "model_id": str(ai_model.id),
        "model_name": ai_model.name,
        "model_type": ai_model.model_type,
        "parameters": ai_model.parameters,
        "affected_cameras": [str(cid) for cid in affected_camera_ids],
        "restart_scheduled": len(affected_camera_ids) > 0,
        "message": f"Parameters updated. {len(affected_camera_ids)} camera(s) will be restarted automatically.",
    }


# ============================================================================
# ADDITIONAL PARAMETERS ENDPOINTS (camera_models.additional_parameters)
# ============================================================================


@router.get(
    "/additional/{camera_id}/{model_id}",
    response_model=Dict[str, Any],
    summary="Get camera-specific additional parameters",
    description="""
    Get additional parameters for a specific camera-model assignment.

    Additional parameters are stored in `camera_models.additional_parameters` and only apply to this specific camera.

    **Common Additional Parameters:**
    - `roi`: Normalized ROI polygon `[[x1,y1], [x2,y2], ...]` with values in [0,1]
    - `conf_threshold`: Override general confidence threshold for this camera
    - `detection_cooldown`: Override general cooldown for this camera

    **Example Response:**
    ```json
    {
        "camera_id": "456e4567-e89b-12d3-a456-426614174001",
        "model_id": "123e4567-e89b-12d3-a456-426614174000",
        "model_name": "Helmet Detection ATLD",
        "model_type": "helmet_detection",
        "additional_parameters": {
            "roi": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
            "conf_threshold": 0.5,
            "detection_cooldown": 300
        }
    }
    ```
    """,
)
async def get_additional_parameters(
    camera_id: UUID, model_id: UUID, db: Session = Depends(get_db)
):
    """Get additional parameters for a camera-model assignment"""
    camera_model = (
        db.query(CameraModel)
        .filter(CameraModel.camera_id == camera_id, CameraModel.model_id == model_id)
        .first()
    )

    if not camera_model:
        raise HTTPException(
            status_code=404,
            detail=f"Camera-Model assignment not found (camera={camera_id}, model={model_id})",
        )

    ai_model = camera_model.ai_model

    default_additional = get_default_additional_parameters(ai_model.model_type)
    stored_additional = camera_model.additional_parameters or {}
    merged_additional = {**default_additional, **stored_additional}

    return {
        "camera_id": str(camera_model.camera_id),
        "model_id": str(ai_model.id),
        "model_name": ai_model.name,
        "model_type": ai_model.model_type,
        "additional_parameters": merged_additional,
    }


@router.put(
    "/additional/{camera_id}/{model_id}",
    response_model=Dict[str, Any],
    summary="Update camera-specific additional parameters",
    description="""
    Update additional parameters for a specific camera-model assignment.

    **Important:**
    - Additional parameters only affect THIS camera
    - The camera will be automatically restarted to apply changes
    - ROI coordinates must be normalized (values in [0, 1] range)

    **Request Body:**
    ```json
    {
        "additional_parameters": {
            "roi": [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]],
            "conf_threshold": 0.5,
            "detection_cooldown": 300
        }
    }
    ```

    **ROI Format:**
    - ROI must be a list of at least 3 points
    - Each point is `[x, y]` where x and y are in range [0.0, 1.0]
    - Points define a polygon (can be rectangle, triangle, or any polygon)
    - Coordinates are normalized: `x = pixel_x / frame_width`, `y = pixel_y / frame_height`

    **Example ROI for center rectangle:**
    ```json
    {
        "roi": [
            [0.2, 0.3],   // top-left: 20% from left, 30% from top
            [0.8, 0.3],   // top-right: 80% from left, 30% from top
            [0.8, 0.7],   // bottom-right: 80% from left, 70% from top
            [0.2, 0.7]    // bottom-left: 20% from left, 70% from top
        ]
    }
    ```

    **Response:**
    Returns updated parameters and restart status.
    """,
)
async def update_additional_parameters(
    camera_id: UUID,
    model_id: UUID,
    update_data: AdditionalParametersUpdate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Update additional parameters for a camera-model assignment"""
    camera_model = (
        db.query(CameraModel)
        .filter(CameraModel.camera_id == camera_id, CameraModel.model_id == model_id)
        .first()
    )

    if not camera_model:
        raise HTTPException(
            status_code=404,
            detail=f"Camera-Model assignment not found (camera={camera_id}, model={model_id})",
        )

    ai_model = camera_model.ai_model

    # Validate parameters
    is_valid, errors = validate_additional_parameters(
        ai_model.model_type,
        update_data.additional_parameters,
        strict=False,  # Allow extra parameters for flexibility
    )

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Parameter validation failed", "errors": errors},
        )

    # Merge with existing parameters (create new dict to ensure change detection)
    current_params = dict(camera_model.additional_parameters or {})
    current_params.update(update_data.additional_parameters)
    camera_model.additional_parameters = current_params

    # Mark as modified to ensure SQLAlchemy tracks the JSONB change
    flag_modified(camera_model, "additional_parameters")

    db.commit()
    db.refresh(camera_model)

    # Schedule camera restart in background
    background_tasks.add_task(restart_cameras_background, [camera_id])
    logger.info(f"Scheduled restart for camera {camera_id}")

    return {
        "camera_id": str(camera_model.camera_id),
        "model_id": str(ai_model.id),
        "model_name": ai_model.name,
        "model_type": ai_model.model_type,
        "additional_parameters": camera_model.additional_parameters,
        "restart_scheduled": True,
        "message": "Additional parameters updated. Camera will be restarted automatically.",
    }
