"""
src/core/websocket_manager.py
WebSocket Manager – Real-time Event & Alert Broadcasting
Hoàn toàn tương thích frontend Petrofe + PostgreSQL backend
"""

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Set, List, Any, Optional

from fastapi import WebSocket, WebSocketDisconnect
from src.utils.datetime_utils import serialize_utc_datetime

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Quản lý WebSocket – Real-time Alert & Event Broadcasting"""

    def __init__(self):
        # 1. Nhận TẤT CẢ sự kiện (global)
        self.global_connections: List[WebSocket] = []

        # 2. Chỉ nhận sự kiện từ 1 camera cụ thể
        self.camera_subscriptions: Dict[str, Set[WebSocket]] = {}

        # 3. CHỈ NHẬN ALERT (high/low) – endpoint /ws/alerts
        self.alert_connections: List[WebSocket] = []

        # Dùng để broadcast từ thread khác (nếu cần)
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

    def register_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Gọi trong FastAPI startup"""
        self.event_loop = loop
        logger.info("WebSocketManager: Event loop registered – ready for realtime")

    # ======================= CONNECT / DISCONNECT =======================

    async def connect(
        self,
        websocket: WebSocket,
        camera_id: Optional[str] = None,
        alert_only: bool = False  # THAM SỐ QUAN TRỌNG CHO /ws/alerts
    ):
        await websocket.accept()

        if alert_only:
            self.alert_connections.append(websocket)
            logger.info("WebSocket connected → /ws/alerts (alert-only)")
        elif camera_id:
            if camera_id not in self.camera_subscriptions:
                self.camera_subscriptions[camera_id] = set()
            self.camera_subscriptions[camera_id].add(websocket)
            logger.info(f"WebSocket subscribed → camera {camera_id}")
        else:
            self.global_connections.append(websocket)
            logger.info("WebSocket connected → all events")

    def disconnect(self, websocket: WebSocket, camera_id: Optional[str] = None):
        if websocket in self.alert_connections:
            self.alert_connections.remove(websocket)
            logger.info("WebSocket disconnected from /ws/alerts")
            return

        if camera_id and camera_id in self.camera_subscriptions:
            self.camera_subscriptions[camera_id].discard(websocket)
            if not self.camera_subscriptions[camera_id]:
                del self.camera_subscriptions[camera_id]
            logger.info(f"WebSocket unsubscribed from camera {camera_id}")
        else:
            self.global_connections = [ws for ws in self.global_connections if ws != websocket]
            logger.info("WebSocket disconnected from global")

    # ======================= BROADCAST METHODS =======================

    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Gửi event cho /ws/events và /ws/events/{id}"""
        message = {
            "type": "event",
            "timestamp": serialize_utc_datetime(datetime.now()),
            **event_data
        }
        camera_id = event_data.get("camera_id")
        await self._broadcast(message, camera_id_str=camera_id)

    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """
        Gửi alert cho:
        - /ws/alerts (alert-only clients)
        - global clients
        - camera-specific clients (nếu có)
        """
        message = {
            "type": "alert",
            "timestamp": serialize_utc_datetime(datetime.now()),
            **alert_data
        }
        camera_id = alert_data.get("camera_id")

        # Gửi riêng cho /ws/alerts subscribers
        await self._broadcast_to_alerts(message)

        # Gửi cho global + camera-specific
        await self._broadcast(message, camera_id_str=camera_id)
    
    async def broadcast_sync(self, sync_data: Dict[str, Any]):
        """
        Broadcast system synchronization events to all connected clients.
        
        Args:
            sync_data: Synchronization result data
        """
        message = {
            "type": "sync",
            "timestamp": serialize_utc_datetime(datetime.now()),
            **sync_data
        }
        await self._broadcast(message)

    async def _broadcast_to_alerts(self, message: dict):
        """Gửi riêng cho các client đang kết nối /ws/alerts"""
        json_msg = json.dumps(message, ensure_ascii=False)
        disconnected = []
        for ws in self.alert_connections[:]:
            try:
                await ws.send_text(json_msg)
            except:
                disconnected.append(ws)
        for ws in disconnected:
            self.alert_connections.remove(ws)

    async def _broadcast(self, message: dict, camera_id_str: Optional[str] = None):
        json_msg = json.dumps(message, ensure_ascii=False)
        disconnected = []

        # Global clients
        for ws in self.global_connections[:]:
            try:
                await ws.send_text(json_msg)
            except:
                disconnected.append((ws, None))

        # Camera-specific clients
        if camera_id_str and camera_id_str in self.camera_subscriptions:
            for ws in self.camera_subscriptions[camera_id_str].copy():
                try:
                    await ws.send_text(json_msg)
                except:
                    disconnected.append((ws, camera_id_str))

        # Cleanup
        for ws, cam_id in disconnected:
            self.disconnect(ws, cam_id)

    # ======================= STATS =======================

    def get_connection_stats(self) -> dict:
        total_cam_subs = sum(len(s) for s in self.camera_subscriptions.values())
        return {
            "global_connections": len(self.global_connections),
            "alert_only_connections": len(self.alert_connections),  # MỚI
            "camera_subscriptions": len(self.camera_subscriptions),
            "total_camera_subscribers": total_cam_subs,
            "active_cameras": list(self.camera_subscriptions.keys()),
            "total_active_connections": len(self.global_connections) + len(self.alert_connections) + total_cam_subs
        }

    # Backward compatible
    def get_stats(self):
        return self.get_connection_stats()


# Global instance
websocket_manager = WebSocketManager()
