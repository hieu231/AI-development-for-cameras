# API endpoint cho performance: CPU, GPU, RAM, VRAM, TEMP, FPS, MAX FPS, MIN FPS

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from src.database import get_db
from src.models.performance import Performance
from src.core.performance_monitor import performance_monitor
import psutil

router = APIRouter(prefix="/performance", tags=["performance"])

# Pydantic schemas
class PerformanceResponse(BaseModel):
    id: int
    timestamp: datetime
    cpu_usage: float
    gpu_usage: float
    ram_usage: float
    vram_usage: float
    temperature: float
    fps: float
    max_fps: float
    min_fps: float
    
    model_config = {"from_attributes": True}

class LatestPerformanceResponse(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    temperature: Optional[float] = None
    total_cameras: int
    running_cameras: int
    total_models: int
    total_fps: float
    avg_fps_per_camera: float
    total_events: int
    events_last_hour: int
    camera_details: dict
    
    model_config = {"from_attributes": True}

@router.get("/help")
def get_help():
    """Get help for performance API"""
    return {
        "message": "API for performance monitoring",
        "endpoints": {
            "GET /performance/": "Get all performance records",
            "GET /performance/latest": "Get latest performance record",
            "GET /performance/{performance_id}": "Get performance record by ID"
        }
    }

@router.get("/", response_model=List[PerformanceResponse])
def get_performance(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get all performance records"""
    performance_records = db.query(Performance).offset(skip).limit(limit).all()
    return performance_records

@router.get("/latest", response_model=LatestPerformanceResponse)
def get_latest_performance(db: Session = Depends(get_db)):
    """Get latest performance record - reads real-time system metrics, FPS from DB"""
    # Get real-time system metrics
    metrics = performance_monitor.collect_metrics()
    
    # Get FPS data from database (latest record)
    latest_db = db.query(Performance).order_by(Performance.timestamp.desc()).first()
    
    # Use DB FPS if available, otherwise use real-time
    if latest_db:
        metrics['total_fps'] = latest_db.total_fps
        metrics['avg_fps_per_camera'] = latest_db.avg_fps_per_camera
        if latest_db.camera_details:
            metrics['camera_details'] = latest_db.camera_details
    
    return metrics
