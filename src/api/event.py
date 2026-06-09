# src/routers/events.py
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, asc, or_, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
import logging

from src.database import get_db
from src.models.event import Event
from src.models.camera import Camera
from src.models.ai_model import AiModel
from src.utils.alert_levels import AlertLevel
from src.utils.datetime_utils import serialize_utc_datetime, to_local_naive_datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["Events"])


# ======================= HELPER FUNCTIONS =======================

def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip().replace(" ", "T").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(value)
        return to_local_naive_datetime(dt)
    except ValueError:
        return None


def apply_event_filters(query, filters: Dict[str, Any]):
    # Hard guard: never return events for inactive or soft-deleted cameras.
    # This enforces "camera off/deleted -> no event returned" across list/count/recent/chart.
    query = query.filter(
        Event.camera.has(
            and_(
                Camera.status == True,  # noqa: E712
                Camera.is_deleted == False,  # noqa: E712
            )
        )
    )

    if camera_id := filters.get("camera_id"):
        query = query.filter(Event.camera_id == camera_id)
    if model_id := filters.get("model_id"):
        query = query.filter(
            or_(
                Event.model_id == model_id,
                Event.detection_data["archived_model"]["id"].astext == str(model_id),
            )
        )
    if location_id := filters.get("location_id"):
        query = query.join(Event.camera).filter(Camera.location_id == location_id)
    if camera_spec_id := filters.get("camera_spec_id"):
        query = query.join(Event.camera).filter(Camera.camera_spec_id == camera_spec_id)
    if alert_level := filters.get("alert_level"):
        level_val = AlertLevel.from_value(alert_level, AlertLevel.LOW)
        query = query.filter(Event.alert_level == level_val.value)
    if start_time := filters.get("start_time"):
        query = query.filter(Event.time >= start_time)
    if end_time := filters.get("end_time"):
        query = query.filter(Event.time <= end_time)
    return query


def serialize_event(event: Event) -> Dict[str, Any]:
    archived_model = (event.detection_data or {}).get("archived_model") or {}
    model_type = "Không xác định"
    if event.ai_model:
        model_type = getattr(event.ai_model, "model_type", "Không xác định")
        if model_type == "Không xác định" and event.ai_model.parameters:
            model_type = event.ai_model.parameters.get("model_type", "Không xác định")
    elif archived_model:
        model_type = archived_model.get("model_type") or model_type

    model_id = event.model_id or archived_model.get("id")
    model_name = archived_model.get("name") or "Model không xác định"
    if event.ai_model:
        model_name = event.ai_model.name

    return {
        "id": str(event.id),
        "time": serialize_utc_datetime(event.time),
        "camera_id": str(event.camera_id),
        "model_id": str(model_id) if model_id else None,
        "detection_data": event.detection_data or {},
        "image_path": event.image_path,
        "alert_level": AlertLevel.from_value(event.alert_level, AlertLevel.LOW).value,
        "camera": {
            "id": str(event.camera.id) if event.camera else str(event.camera_id),
            "name": event.camera.name if event.camera else "Camera không xác định",
        },
        "ai_model": {
            "id": str(event.ai_model.id) if event.ai_model else (str(model_id) if model_id else None),
            "name": model_name,
            "model_type": model_type,
        },
    }


# ======================= MAIN EVENT ENDPOINTS =======================

@router.get("/")
def get_events_paginated(
    page: int = Query(1, ge=1),
    pageSize: int = Query(20, ge=1, le=100),
    camera_id: Optional[UUID] = None,
    model_id: Optional[UUID] = None,
    location_id: Optional[UUID] = None,
    camera_spec_id: Optional[UUID] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    alert_level: Optional[AlertLevel] = None,
    sort: str = Query("-time", pattern="^(time|-time)$"),
    db: Session = Depends(get_db),
):
    start_dt = parse_iso_datetime(start_time)
    end_dt = parse_iso_datetime(end_time)
    if start_time and not start_dt:
        raise HTTPException(400, "Invalid start_time format")
    if end_time and not end_dt:
        raise HTTPException(400, "Invalid end_time format")

    query = db.query(Event).options(joinedload(Event.camera), joinedload(Event.ai_model))
    query = apply_event_filters(query, {
        "camera_id": camera_id, "model_id": model_id, "location_id": location_id,
        "camera_spec_id": camera_spec_id, "alert_level": alert_level,
        "start_time": start_dt, "end_time": end_dt,
    })
    query = query.order_by(desc(Event.time)) if sort == "-time" else query.order_by(asc(Event.time))

    total_items = query.count()
    total_pages = (total_items + pageSize - 1) // pageSize if total_items else 0
    events = query.offset((page - 1) * pageSize).limit(pageSize).all()

    return {
        "items": [serialize_event(e) for e in events],
        "page": page, "pageSize": pageSize, "totalItems": total_items,
        "totalPages": total_pages, "sort": sort,
    }


@router.get("/count")
def get_events_count(
    camera_id: Optional[UUID] = None,
    model_id: Optional[UUID] = None,
    location_id: Optional[UUID] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    alert_level: Optional[AlertLevel] = None,
    db: Session = Depends(get_db),
):
    start_dt = parse_iso_datetime(start_time)
    end_dt = parse_iso_datetime(end_time)

    query = db.query(func.count(Event.id))
    query = apply_event_filters(query, {
        "camera_id": camera_id, "model_id": model_id, "location_id": location_id,
        "alert_level": alert_level, "start_time": start_dt, "end_time": end_dt,
    })
    return {"total": query.scalar() or 0}


@router.get("/recent")
def get_recent_events(
    limit: int = Query(20, ge=1, le=100),
    camera_id: Optional[UUID] = None,
    model_id: Optional[UUID] = None,
    location_id: Optional[UUID] = None,
    camera_spec_id: Optional[UUID] = None,
    alert_level: Optional[AlertLevel] = None,
    db: Session = Depends(get_db)
):
    """Lấy danh sách events gần nhất với các filter tùy chọn
    
    - **limit**: Số lượng events trả về (default: 20, max: 100)
    - **camera_id**: Lọc theo camera
    - **model_id**: Lọc theo AI model
    - **location_id**: Lọc theo location
    - **camera_spec_id**: Lọc theo camera spec
    - **alert_level**: Lọc theo mức cảnh báo (high/low)
    """
    query = db.query(Event).options(joinedload(Event.camera), joinedload(Event.ai_model))
    query = apply_event_filters(query, {
        "camera_id": camera_id, "model_id": model_id,
        "location_id": location_id, "camera_spec_id": camera_spec_id,
        "alert_level": alert_level,
    })
    events = query.order_by(desc(Event.time)).limit(limit).all()
    return [serialize_event(e) for e in events]


@router.get("/chart_data")
def get_chart_data(
    start_time: str = Query(...),
    end_time: str = Query(...),
    group_by: str = Query("day", pattern="^(day|week|month)$"),
    camera_id: Optional[UUID] = None,
    model_id: Optional[UUID] = None,
    location_id: Optional[UUID] = None,
    camera_spec_id: Optional[UUID] = None,
    db: Session = Depends(get_db),
):
    start_dt = parse_iso_datetime(start_time)
    end_dt = parse_iso_datetime(end_time)
    if not start_dt or not end_dt:
        return {"items": [], "group_by": group_by, "total_events": 0}
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    query = db.query(
        func.date_trunc(group_by, Event.time).label("bucket"),
        func.count(Event.id).label("count")
    )
    query = apply_event_filters(query, {
        "camera_id": camera_id, "model_id": model_id,
        "location_id": location_id, "camera_spec_id": camera_spec_id,
        "start_time": start_dt, "end_time": end_dt,
    }).group_by("bucket").order_by("bucket")

    rows = query.all()
    items = []
    total = 0
    for row in rows:
        bucket_start = row.bucket
        if group_by == "day":
            bucket_end = bucket_start + timedelta(days=1) - timedelta(microseconds=1)
        elif group_by == "week":
            bucket_end = bucket_start + timedelta(weeks=1) - timedelta(microseconds=1)
        else:
            from calendar import monthrange
            year, month = bucket_start.year, bucket_start.month
            days_in_month = monthrange(year, month)[1]
            bucket_end = bucket_start.replace(day=days_in_month) + timedelta(days=1) - timedelta(microseconds=1)

        items.append({
            "start_time": serialize_utc_datetime(bucket_start),
            "end_time": serialize_utc_datetime(bucket_end),
            "total_events": row.count,
        })
        total += row.count

    return {"items": items, "group_by": group_by, "total_events": total}


@router.get("/{event_id}")
def get_event_by_id(event_id: UUID = Path(...), db: Session = Depends(get_db)):
    event = db.get(Event, event_id, options=[joinedload(Event.camera), joinedload(Event.ai_model)])
    if (
        not event
        or not event.camera
        or bool(getattr(event.camera, "is_deleted", False))
        or not bool(getattr(event.camera, "status", False))
    ):
        raise HTTPException(404, "Event not found")
    return serialize_event(event)


# ======================= ALERT ENDPOINTS  =======================

@router.get("/alerts")
def get_all_alerts(
    level: Optional[AlertLevel] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    days: int = Query(7, ge=1, le=90),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    cutoff = datetime.now() - timedelta(days=days)
    query = db.query(Event).options(joinedload(Event.camera), joinedload(Event.ai_model)) \
        .filter(Event.time >= cutoff, Event.alert_level.in_(["high", "low"]))

    # Keep alerts consistent with event list behavior: ignore inactive/deleted cameras.
    query = query.filter(
        Event.camera.has(
            and_(
                Camera.status == True,  # noqa: E712
                Camera.is_deleted == False,  # noqa: E712
            )
        )
    )

    if level:
        query = query.filter(Event.alert_level == level.value)
    if camera_id:
        query = query.filter(Event.camera_id == camera_id)

    total = query.count()
    alerts = query.order_by(desc(Event.time)).offset(skip).limit(limit).all()

    return {
        "status": "success",
        "total": total,
        "returned": len(alerts),
        "skip": skip,
        "limit": limit,
        "alerts": [serialize_event(a) for a in alerts],
    }


@router.get("/alerts/high")
def get_high_alerts(days: int = Query(7, ge=1, le=90), skip: int = Query(0), limit: int = Query(100), db: Session = Depends(get_db)):
    resp = get_all_alerts(level=AlertLevel.HIGH, days=days, skip=skip, limit=limit, db=db)
    resp.update({"level": "high", "color": "#FFA500", "message": "Cảnh báo mức cao - Nguy hiểm tiềm ẩn"})
    return resp


@router.get("/alerts/low")
def get_low_alerts(days: int = Query(7, ge=1, le=90), skip: int = Query(0), limit: int = Query(100), db: Session = Depends(get_db)):
    resp = get_all_alerts(level=AlertLevel.LOW, days=days, skip=skip, limit=limit, db=db)
    resp.update({"level": "low", "color": "#FFFF00", "message": "Cảnh báo mức thấp - Thông thường"})
    return resp


@router.get("/alerts/camera/{camera_id}")
def get_alerts_by_camera(camera_id: UUID = Path(...), days: int = Query(7), skip: int = Query(0), limit: int = Query(100), db: Session = Depends(get_db)):
    resp = get_all_alerts(camera_id=camera_id, days=days, skip=skip, limit=limit, db=db)
    resp["camera_id"] = str(camera_id)
    return resp


@router.get("/alerts/camera/{camera_id}/level/{level}")
def get_alerts_by_camera_level(
    camera_id: UUID = Path(...), level: AlertLevel = Path(...),
    days: int = Query(7), skip: int = Query(0), limit: int = Query(100), db: Session = Depends(get_db)
):
    resp = get_all_alerts(level=level, camera_id=camera_id, days=days, skip=skip, limit=limit, db=db)
    resp.update({"camera_id": str(camera_id), "alert_level": level.value})
    return resp


@router.get("/alerts/stats")
def get_alert_statistics(days: int = Query(7, ge=1, le=90), db: Session = Depends(get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    stats = db.query(Event.alert_level, func.count(Event.id)) \
        .filter(Event.time >= cutoff, Event.alert_level.in_(["high", "low"])) \
        .group_by(Event.alert_level).all()

    result = {"high": 0, "low": 0}
    for level, count in stats:
        normalized_level = AlertLevel.from_value(level, AlertLevel.LOW).value
        result[normalized_level] = result.get(normalized_level, 0) + count
    total = sum(result.values())

    return {
        "status": "success",
        "period_days": days,
        "statistics": {
            "high": {"count": result["high"], "color": "#FFA500", "message": "Cảnh báo mức cao - Nguy hiểm tiềm ẩn"},
            "low": {"count": result["low"], "color": "#FFFF00", "message": "Cảnh báo mức thấp - Thông thường"},
            "total": total
        },
        "by_camera": {},  # Nếu cần thống kê theo camera thì thêm sau
        "total_cameras": 0
    }


@router.delete("/alerts/{alert_id}")
def delete_alert(alert_id: UUID = Path(...), db: Session = Depends(get_db)):
    event = db.get(Event, alert_id)
    if not event:
        raise HTTPException(404, "Alert not found")
    db.delete(event)
    db.commit()
    return {"status": "success", "message": "Alert deleted successfully", "alert_id": str(alert_id)}


@router.delete("/alerts/clear/older-than/{days}")
def clear_old_alerts(days: int = Path(..., ge=1, le=365), db: Session = Depends(get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    deleted = db.query(Event).filter(Event.time < cutoff).delete()
    db.commit()
    return {
        "status": "success",
        "message": f"Deleted {deleted} alerts older than {days} days",
        "deleted_count": deleted
    }


# ======================= HELP ENDPOINT - CHI TIẾT NHƯ BẠN GỬI =======================

@router.get("/help")
def get_help():
    """Get Event API help - Full documentation like old version"""
    return {
        "message": "API for events management",
        "endpoints": {
            "get_events": "GET /events/ - Get events list with pagination and filtering",
            "get_help": "GET /events/help - Show this help",
            "get_events_count": "GET /events/count - Get total events count with filters",
            "get_recent_events": "GET /events/recent - Get recent events with optional filters",
            "get_event": "GET /events/{event_id} - Get event details by ID",
            "get_chart_data": "GET /events/chart_data - Get aggregated event counts grouped by time period",
            "get_all_alerts": "GET /events/alerts - Get alerts with level, camera_id, days filter",
            "get_high_alerts": "GET /events/alerts/high",
            "get_low_alerts": "GET /events/alerts/low",
            "get_alerts_by_camera": "GET /events/alerts/camera/{camera_id}",
            "get_alert_stats": "GET /events/alerts/stats",
            "delete_alert": "DELETE /events/alerts/{id}",
        },
        "query_parameters": {
            "events_list": {
                "page": "int - Page number, starts from 1 (default: 1)",
                "pageSize": "int - Number of items per page (default: 20, max: 100)",
                "camera_id": "UUID - Filter by camera ID (optional)",
                "model_id": "UUID - Filter by AI model ID (optional)",
                "location_id": "UUID - Filter by location ID (optional)",
                "camera_spec_id": "UUID - Filter by camera spec ID (optional)",
                "start_time": "datetime - Filter events after this time, ISO 8601 format (optional)",
                "end_time": "datetime - Filter events before this time, ISO 8601 format (optional)",
                "alert_level": "high/low - Filter by alert level",
                "sort": "string - Sort order: 'time' (ascending) or '-time' (descending, default)"
            },
            "events_count": {
                "camera_id": "UUID - Filter by camera ID (optional)",
                "model_id": "UUID - Filter by AI model ID (optional)",
                "location_id": "UUID - Filter by location ID (optional)",
                "start_time": "datetime - Filter events after this time, ISO 8601 format (optional)",
                "end_time": "datetime - Filter events before this time, ISO 8601 format (optional)"
            },
            "recent_events": {
                "limit": "int - Number of events to return (default: 20, max: 100)",
                "camera_id": "UUID - Filter by camera ID (optional)",
                "model_id": "UUID - Filter by AI model ID (optional)",
                "location_id": "UUID - Filter by location ID (optional)",
                "camera_spec_id": "UUID - Filter by camera spec ID (optional)",
                "alert_level": "high/low - Filter by alert level (optional)"
            },
            "chart_data": {
                "camera_id": "UUID - Filter by camera ID (optional)",
                "model_id": "UUID - Filter by AI model ID (optional)",
                "location_id": "UUID - Filter by location ID (optional)",
                "camera_spec_id": "UUID - Filter by camera spec ID (optional)",
                "start_time": "datetime - Filter events after this time, ISO 8601 format (required)",
                "end_time": "datetime - Filter events before this time, ISO 8601 format (required)",
                "group_by": "string - Group by: 'day', 'week', or 'month' (required)"
            },
            "alerts": {
                "level": "high/low - Filter by alert level",
                "camera_id": "UUID - Filter by camera",
                "days": "int - Last N days (default 7)",
                "skip": "int - Pagination offset",
                "limit": "int - Page size (max 500)"
            }
        },
        "response_schemas": {
            "events_count": {"total": "int - Total number of events matching the filters"},
            "paginated_events": {
                "items": "array - List of events",
                "page": "int - Current page number",
                "pageSize": "int - Number of items per page",
                "totalItems": "int - Total number of events",
                "totalPages": "int - Total number of pages",
                "sort": "string - Current sort order"
            },
            "event_item": {
                "id": "UUID - Event ID",
                "time": "datetime - Detection time",
                "camera_id": "UUID - Camera ID",
                "model_id": "UUID - AI model ID",
                "detection_data": "dict - Detection metadata",
                "image_path": "string - Saved image path (optional)",
                "alert_level": "string - high/low",
                "camera": "object - Camera info (id, name)",
                "ai_model": "object - AI model info (id, name, model_type)"
            }
        },
    }
