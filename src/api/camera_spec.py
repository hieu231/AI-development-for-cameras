# src/api/camera_spec.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

from src.models.camera_spec import CameraSpec
from src.database import get_db

# Pydantic schemas
class CameraSpecBase(BaseModel):
    name: str
    manufacturer: str
    model_series: Optional[str] = None
    category: Optional[str] = None
    resolution: Optional[str] = None
    max_fps: Optional[int] = None
    ptz_support: bool = False
    audio_support: bool = False
    ir_support: bool = False
    ai_support: bool = False
    onvif_profile: Optional[str] = None
    brand_website: Optional[str] = None
    brand_description: Optional[str] = None
    rtsp_format: Optional[str] = None
    spec_metadata: dict = {}
    is_active: bool = True

class CameraSpecCreate(CameraSpecBase):
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Hikvision DS-2CD2185G0-I",
                    "manufacturer": "Hikvision",
                    "model_series": "DS-2CD2185G0-I",
                    "category": "IP Camera",
                    "resolution": "3840x2160",
                    "max_fps": 30,
                    "ptz_support": False,
                    "audio_support": True,
                    "ir_support": True,
                    "ai_support": False,
                    "onvif_profile": "S",
                    "brand_website": "https://www.hikvision.com",
                    "brand_description": "Leading video surveillance manufacturer",
                    "rtsp_format": "rtsp://admin:password@IP:554/Streaming/Channels/101",
                    "spec_metadata": {"lens": "2.8mm", "night_vision": "30m"},
                    "is_active": True
                },
                {
                    "name": "Dahua SD59225U-HNI",
                    "manufacturer": "Dahua",
                    "model_series": "SD59225U-HNI",
                    "category": "PTZ Camera",
                    "resolution": "1920x1080",
                    "max_fps": 60,
                    "ptz_support": True,
                    "audio_support": True,
                    "ir_support": True,
                    "ai_support": True,
                    "onvif_profile": "T",
                    "rtsp_format": "rtsp://admin:password@IP:554/cam/realmonitor?channel=1&subtype=0",
                    "spec_metadata": {"zoom": "25x optical"}
                }
            ]
        }
    }

class CameraSpecUpdate(BaseModel):
    name: Optional[str] = None
    manufacturer: Optional[str] = None
    model_series: Optional[str] = None
    category: Optional[str] = None
    resolution: Optional[str] = None
    max_fps: Optional[int] = None
    ptz_support: Optional[bool] = None
    audio_support: Optional[bool] = None
    ir_support: Optional[bool] = None
    ai_support: Optional[bool] = None
    onvif_profile: Optional[str] = None
    brand_website: Optional[str] = None
    brand_description: Optional[str] = None
    rtsp_format: Optional[str] = None
    spec_metadata: Optional[dict] = None
    is_active: Optional[bool] = None

class CameraSpecResponse(CameraSpecBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

router = APIRouter(prefix="/camera-specs", tags=["camera-specs"])

@router.get("/help")
def get_help():
    """Lấy hướng dẫn sử dụng Camera Spec API"""
    return {
        "endpoints": {
            "GET /camera-specs/": "Lấy danh sách camera specs (hỗ trợ filter)",
            "GET /camera-specs/help": "Hiển thị hướng dẫn này",
            "GET /camera-specs/{camera_spec_id}": "Lấy thông tin chi tiết của 1 camera spec theo ID",
            "POST /camera-specs/": "Tạo camera spec mới",
            "PUT /camera-specs/{camera_spec_id}": "Cập nhật thông tin camera spec",
            "DELETE /camera-specs/{camera_spec_id}": "Xóa camera spec"
        },
        "camera_spec_schema": {
            "name": "string - Tên model camera",
            "manufacturer": "string - Nhà sản xuất (Hikvision, Dahua, etc.)",
            "model_series": "string - Dòng sản phẩm (optional)",
            "category": "string - Loại camera (IP Camera, PTZ Camera, etc.) (optional)",
            "resolution": "string - Độ phân giải (vd: 1920x1080, 3840x2160) (optional)",
            "max_fps": "integer - FPS tối đa (optional)",
            "ptz_support": "boolean - Hỗ trợ PTZ (Pan-Tilt-Zoom) (default: false)",
            "audio_support": "boolean - Hỗ trợ audio (default: false)",
            "ir_support": "boolean - Hỗ trợ hồng ngoại (default: false)",
            "ai_support": "boolean - Hỗ trợ AI (default: false)",
            "onvif_profile": "string - ONVIF profile (S, T, G, etc.) (optional)",
            "brand_website": "string - Website nhà sản xuất (optional)",
            "brand_description": "string - Mô tả thương hiệu (optional)",
            "rtsp_format": "string - RTSP URL format template (e.g., rtsp://username:password@IP:554/stream/main) (optional)",
            "spec_metadata": "object - Thông tin bổ sung dạng JSON (optional)",
            "is_active": "boolean - Trạng thái active (default: true)"
        },
        "filter_parameters": {
            "name": "string - Tìm kiếm theo tên (partial match, không phân biệt hoa thường)",
            "manufacturer": "string - Tìm kiếm theo nhà sản xuất (partial match)",
            "category": "string - Tìm kiếm theo loại camera (partial match)",
            "is_active": "boolean - Lọc theo trạng thái active (true/false)",
            "ptz_support": "boolean - Lọc theo hỗ trợ PTZ (true/false)",
            "audio_support": "boolean - Lọc theo hỗ trợ audio (true/false)",
            "ir_support": "boolean - Lọc theo hỗ trợ hồng ngoại (true/false)",
            "ai_support": "boolean - Lọc theo hỗ trợ AI (true/false)",
            "skip": "int - Số bản ghi bỏ qua (default: 0)",
            "limit": "int - Số bản ghi tối đa trả về (default: 100)"
        },
        "filter_examples": [
            "GET /camera-specs/?manufacturer=hikvision - Lấy specs của Hikvision",
            "GET /camera-specs/?category=ptz - Lấy specs loại PTZ",
            "GET /camera-specs/?ptz_support=true - Lấy specs có hỗ trợ PTZ",
            "GET /camera-specs/?ai_support=true&is_active=true - Lấy specs có AI và đang active",
            "GET /camera-specs/?name=DS-2CD - Tìm specs có tên chứa 'DS-2CD'"
        ],
        "example": {
            "name": "Hikvision DS-2CD2185G0-I",
            "manufacturer": "Hikvision",
            "model_series": "DS-2CD2185G0-I",
            "category": "IP Camera",
            "resolution": "3840x2160",
            "max_fps": 30,
            "ptz_support": False,
            "audio_support": True,
            "ir_support": True,
            "ai_support": False,
            "onvif_profile": "S",
            "brand_website": "https://www.hikvision.com",
            "brand_description": "Leading video surveillance manufacturer",
            "rtsp_format": "rtsp://admin:password@IP:554/Streaming/Channels/101",
            "spec_metadata": {"lens": "2.8mm", "night_vision": "30m"},
            "is_active": True
        }
    }


@router.get("/", response_model=List[CameraSpecResponse])
def get_camera_specs(
    skip: int = 0, 
    limit: int = 100,
    name: Optional[str] = None,
    manufacturer: Optional[str] = None,
    category: Optional[str] = None,
    is_active: Optional[bool] = None,
    ptz_support: Optional[bool] = None,
    audio_support: Optional[bool] = None,
    ir_support: Optional[bool] = None,
    ai_support: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """Lấy danh sách camera specs với các filter tùy chọn
    
    - **name**: Tìm kiếm theo tên (partial match, không phân biệt hoa thường)
    - **manufacturer**: Tìm kiếm theo nhà sản xuất (partial match, không phân biệt hoa thường)
    - **category**: Tìm kiếm theo loại camera (partial match, không phân biệt hoa thường)
    - **is_active**: Lọc theo trạng thái active (true/false)
    - **ptz_support**: Lọc theo hỗ trợ PTZ (true/false)
    - **audio_support**: Lọc theo hỗ trợ audio (true/false)
    - **ir_support**: Lọc theo hỗ trợ hồng ngoại (true/false)
    - **ai_support**: Lọc theo hỗ trợ AI (true/false)
    """
    query = db.query(CameraSpec)
    
    # Apply filters
    if name is not None:
        query = query.filter(CameraSpec.name.ilike(f"%{name}%"))
    if manufacturer is not None:
        query = query.filter(CameraSpec.manufacturer.ilike(f"%{manufacturer}%"))
    if category is not None:
        query = query.filter(CameraSpec.category.ilike(f"%{category}%"))
    if is_active is not None:
        query = query.filter(CameraSpec.is_active == is_active)
    if ptz_support is not None:
        query = query.filter(CameraSpec.ptz_support == ptz_support)
    if audio_support is not None:
        query = query.filter(CameraSpec.audio_support == audio_support)
    if ir_support is not None:
        query = query.filter(CameraSpec.ir_support == ir_support)
    if ai_support is not None:
        query = query.filter(CameraSpec.ai_support == ai_support)
    
    camera_specs = query.offset(skip).limit(limit).all()
    return camera_specs

@router.get("/{camera_spec_id}", response_model=CameraSpecResponse)
def get_camera_spec(camera_spec_id: UUID, db: Session = Depends(get_db)):
    """Lấy thông tin camera spec theo ID"""
    camera_spec = db.query(CameraSpec).filter(CameraSpec.id == camera_spec_id).first()
    if not camera_spec:
        raise HTTPException(status_code=404, detail="Camera spec not found")
    return camera_spec

@router.post("/", response_model=CameraSpecResponse, status_code=status.HTTP_201_CREATED)
def create_camera_spec(camera_spec: CameraSpecCreate, db: Session = Depends(get_db)):
    """Tạo camera spec mới"""
    db_camera_spec = CameraSpec(**camera_spec.model_dump())
    db.add(db_camera_spec)
    db.commit()
    db.refresh(db_camera_spec)
    return db_camera_spec

@router.put("/{camera_spec_id}", response_model=CameraSpecResponse)
def update_camera_spec(camera_spec_id: UUID, camera_spec: CameraSpecUpdate, db: Session = Depends(get_db)):
    """Cập nhật camera spec"""
    db_camera_spec = db.query(CameraSpec).filter(CameraSpec.id == camera_spec_id).first()
    if not db_camera_spec:
        raise HTTPException(status_code=404, detail="Camera spec not found")
    update_data = camera_spec.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_camera_spec, key, value)
    db.commit()
    db.refresh(db_camera_spec)
    return db_camera_spec

@router.delete("/{camera_spec_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera_spec(camera_spec_id: UUID, db: Session = Depends(get_db)):
    """Xóa camera spec"""
    db_camera_spec = db.query(CameraSpec).filter(CameraSpec.id == camera_spec_id).first()
    if not db_camera_spec:
        raise HTTPException(status_code=404, detail="Camera spec not found")
    db.delete(db_camera_spec)
    db.commit()
    return None
