"""
src/routers/websocket.py
WebSocket API Router – Real-time Event & Alert Broadcasting
Hoàn toàn tương thích frontend Petrofe (đã test thực tế)
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import UUID
from datetime import datetime

from src.core.websocket_manager import websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


@router.websocket("/events")
async def websocket_all_events(websocket: WebSocket):
    """
    Nhận TẤT CẢ sự kiện từ mọi camera
    Dùng cho: Dashboard tổng quan
    """
    await websocket_manager.connect(websocket, camera_id=None)

    try:
        while True:
            data = await websocket.receive_text()
            if data.strip() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("Client disconnected from /ws/events (all events)")
    except Exception as e:
        logger.error(f"WebSocket error (/ws/events): {e}")
        websocket_manager.disconnect(websocket)


@router.websocket("/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    ENDPOINT QUAN TRỌNG NHẤT – Chỉ nhận ALERT (high/low)
    Đây là endpoint chính mà frontend Petrofe dùng để hiện cảnh báo cháy realtime!
    """
    await websocket_manager.connect(websocket, alert_only=True)  # ĐÃ SỬA

    try:
        while True:
            data = await websocket.receive_text()
            if data.strip() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
        logger.info("Client disconnected from /ws/alerts (alert-only)")
    except Exception as e:
        logger.error(f"WebSocket error (/ws/alerts): {e}")
        websocket_manager.disconnect(websocket)


@router.websocket("/events/{camera_id}")
async def websocket_camera_events(websocket: WebSocket, camera_id: UUID):
    """
    Nhận sự kiện chỉ từ 1 camera cụ thể
    Dùng cho: Xem chi tiết camera
    """
    camera_id_str = str(camera_id)
    await websocket_manager.connect(websocket, camera_id=camera_id_str)

    try:
        while True:
            data = await websocket.receive_text()
            if data.strip() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, camera_id=camera_id_str)
        logger.info(f"Client disconnected from camera {camera_id_str}")
    except Exception as e:
        logger.error(f"WebSocket error (camera {camera_id_str}): {e}")
        websocket_manager.disconnect(websocket, camera_id=camera_id_str)


@router.get("/stats")
async def get_websocket_stats():
    """
    API xem thống kê kết nối realtime
    Dùng để monitoring hệ thống
    """
    stats = websocket_manager.get_connection_stats()
    return {
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "connections": stats
    }


@router.get("/ping")
async def ws_ping():
    """Health check nhanh cho WebSocket system"""
    return {"status": "ok", "message": "WebSocket system is alive"}
