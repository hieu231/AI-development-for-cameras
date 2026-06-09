# src/api/location.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

from src.models.location import Location
from src.database import get_db

# Pydantic schemas
class LocationBase(BaseModel):
    name: str
    description: str

class LocationCreate(LocationBase):
    pass

class LocationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class LocationResponse(LocationBase):
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

router = APIRouter(prefix="/locations", tags=["locations"])

@router.get("/help")
def get_help():
    """Lấy hướng dẫn sử dụng Location API"""
    return {
        "endpoints": {
            "GET /locations/": "Lấy danh sách locations (hỗ trợ filter)",
            "GET /locations/help": "Hiển thị hướng dẫn này",
            "GET /locations/{location_id}": "Lấy thông tin chi tiết của 1 location theo ID",
            "POST /locations/": "Tạo location mới",
            "PUT /locations/{location_id}": "Cập nhật thông tin location",
            "DELETE /locations/{location_id}": "Xóa location"
        },
        "location_schema": {
            "name": "string - Tên location",
            "description": "string - Mô tả location"
        },
        "filter_parameters": {
            "name": "string - Tìm kiếm theo tên (partial match, không phân biệt hoa thường)",
            "description": "string - Tìm kiếm theo mô tả (partial match, không phân biệt hoa thường)",
            "skip": "int - Số bản ghi bỏ qua (default: 0)",
            "limit": "int - Số bản ghi tối đa trả về (default: 100)"
        },
        "filter_examples": [
            "GET /locations/?name=cổng - Tìm locations có tên chứa 'cổng'",
            "GET /locations/?description=chính - Tìm locations có mô tả chứa 'chính'",
            "GET /locations/?name=kho&description=hàng - Kết hợp nhiều filter"
        ],
        "example": {
            "name": "Location Cổng chính",
            "description": "Mô tả location cổng chính"
        }
    }


@router.get("/", response_model=List[LocationResponse])
def get_locations(
    skip: int = 0, 
    limit: int = 100,
    name: Optional[str] = None,
    description: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Lấy danh sách locations với các filter tùy chọn
    
    - **name**: Tìm kiếm theo tên (partial match, không phân biệt hoa thường)
    - **description**: Tìm kiếm theo mô tả (partial match, không phân biệt hoa thường)
    """
    query = db.query(Location)
    
    # Apply filters
    if name is not None:
        query = query.filter(Location.name.ilike(f"%{name}%"))
    if description is not None:
        query = query.filter(Location.description.ilike(f"%{description}%"))
    
    locations = query.offset(skip).limit(limit).all()
    return locations

@router.get("/{location_id}", response_model=LocationResponse)
def get_location(location_id: UUID, db: Session = Depends(get_db)):
    """Lấy thông tin location theo ID"""
    location = db.query(Location).filter(Location.id == location_id).first()
    if not location:
        raise HTTPException(status_code=404, detail="Location not found")
    return location

@router.post("/", response_model=LocationResponse, status_code=status.HTTP_201_CREATED)
def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    """Tạo location mới"""
    db_location = Location(**location.model_dump())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

@router.put("/{location_id}", response_model=LocationResponse)
def update_location(location_id: UUID, location: LocationUpdate, db: Session = Depends(get_db)):
    """Cập nhật location"""
    db_location = db.query(Location).filter(Location.id == location_id).first()
    if not db_location:
        raise HTTPException(status_code=404, detail="Location not found")
    update_data = location.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_location, key, value)
    db.commit()
    db.refresh(db_location)
    return db_location

@router.delete("/{location_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_location(location_id: UUID, db: Session = Depends(get_db)):
    """Xóa location"""
    db_location = db.query(Location).filter(Location.id == location_id).first()
    if not db_location:
        raise HTTPException(status_code=404, detail="Location not found")
    db.delete(db_location)
    db.commit()
    return {"message": "Location deleted successfully"}
