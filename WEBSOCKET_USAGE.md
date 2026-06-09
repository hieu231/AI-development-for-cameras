# WebSocket Usage Guide

## Overview
The WebSocket system provides **real-time event broadcasting** for AI detection events from cameras. When an AI model detects something (person, vehicle, etc.), the event is automatically broadcast to all connected WebSocket clients.

## WebSocket Endpoints

### 1. All Events (All Cameras)
**Endpoint:** `ws://localhost:8668/ws/events`

Subscribe to events from **all cameras** in the system.

**Usage:**
```javascript
const ws = new WebSocket('ws://localhost:8668/ws/events');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Event received:', data);
};
```

### 2. Specific Camera Events
**Endpoint:** `ws://localhost:8668/ws/events/{camera_id}`

Subscribe to events from a **specific camera** only.

**Usage:**
```javascript
const cameraId = 'your-camera-uuid-here';
const ws = new WebSocket(`ws://localhost:8668/ws/events/${cameraId}`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Camera event:', data);
};
```

### 3. Connection Statistics
**Endpoint:** `GET http://localhost:8668/ws/stats`

Get current WebSocket connection statistics.

**Response:**
```json
{
    "status": "success",
    "data": {
        "total_connections": 2,
        "camera_subscriptions": 1,
        "total_camera_subscribers": 1,
        "subscribed_cameras": ["camera-uuid-1"]
    }
}
```

## Event Message Format

When an AI detection event occurs, all subscribed WebSocket clients receive a JSON message:

```json
{
    "id": "event-uuid",
    "event_id": "event-uuid",
    "time": "2025-11-19T22:40:00.123456",
    "camera_id": "camera-uuid",
    "model_id": "model-uuid",
    "camera_name": "Camera 1",
    "model_name": "YOLOv8 Person Detection",
    "model_type": "person_detection",
    "image_path": "/evidence_image/2025/11/19/event-123.jpg",
    "metadata": {
        "type": "person",
        "confidence": 0.95,
        "bbox": [100, 200, 300, 400],
        "count": 1
    },
    "detection_data": {
        "type": "person",
        "confidence": 0.95,
        "bbox": [100, 200, 300, 400]
    },
    "camera": {
        "id": "camera-uuid",
        "name": "Camera 1"
    },
    "ai_model": {
        "id": "model-uuid",
        "name": "YOLOv8 Person Detection",
        "model_type": "person_detection"
    }
}
```

## How It Works

### 1. Event Detection Flow
```
Camera Stream → AI Model Processing → Event Detected → Save to Database → Broadcast via WebSocket
```

### 2. Broadcasting Process
- When an event is detected in `camera_single_thread.py`, it calls `_broadcast_event()`
- The event data is prepared and sent to `websocket_manager.broadcast_event()`
- All connected WebSocket clients receive the event in real-time

### 3. Connection Management
- **All Events**: Clients connected to `/ws/events` receive events from all cameras
- **Camera-Specific**: Clients connected to `/ws/events/{camera_id}` only receive events from that camera
- Connections are automatically cleaned up when clients disconnect

## Code Locations

### WebSocket API Routes
**File:** `src/api/websocket.py`
- `/ws/events` - All events endpoint
- `/ws/events/{camera_id}` - Camera-specific endpoint
- `/ws/stats` - Connection statistics

### WebSocket Manager
**File:** `src/core/websocket_manager.py`
- Manages all WebSocket connections
- Handles broadcasting to all or specific camera subscribers
- Tracks connection statistics

### Event Broadcasting
**File:** `src/core/camera_single_thread.py`
- Line 210: `_broadcast_event()` is called after saving event
- Line 218-268: Event data preparation and WebSocket broadcast

### Server Setup
**File:** `src/server.py`
- Line 17-18: WebSocket imports
- Line 60-61: Event loop registration on startup
- Line 72: WebSocket router included

## Example Client Code

### JavaScript/TypeScript
```javascript
// Connect to all events
const ws = new WebSocket('ws://localhost:8668/ws/events');

ws.onopen = () => {
    console.log('WebSocket connected');
    // Send ping to keep connection alive
    setInterval(() => ws.send('ping'), 30000);
};

ws.onmessage = (event) => {
    if (event.data === 'pong') {
        console.log('Received pong');
        return;
    }
    
    const eventData = JSON.parse(event.data);
    console.log('AI Detection Event:', {
        camera: eventData.camera_name,
        model: eventData.model_name,
        time: eventData.time,
        detection: eventData.metadata
    });
    
    // Display event in UI
    displayEvent(eventData);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('WebSocket disconnected');
    // Reconnect after 5 seconds
    setTimeout(() => {
        ws = new WebSocket('ws://localhost:8668/ws/events');
    }, 5000);
};
```

### Python
```python
import asyncio
import websockets
import json

async def listen_events():
    uri = "ws://localhost:8668/ws/events"
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        # Send ping to keep connection alive
        asyncio.create_task(ping_loop(websocket))
        
        while True:
            message = await websocket.recv()
            
            if message == "pong":
                print("Received pong")
                continue
            
            event = json.loads(message)
            print(f"Event from {event['camera_name']}: {event['metadata']}")

async def ping_loop(websocket):
    while True:
        await asyncio.sleep(30)
        await websocket.send("ping")

# Run
asyncio.run(listen_events())
```

### cURL (Testing)
```bash
# Test WebSocket connection (requires websocat or similar tool)
websocat ws://localhost:8668/ws/events

# Or use wscat
wscat -c ws://localhost:8668/ws/events
```

## Demo Page

There's a demo HTML page available: `websocket_demo.html`

To use it:
1. Open `websocket_demo.html` in a browser
2. Enter WebSocket URL: `ws://localhost:8668/ws/events`
3. Click "Connect"
4. Events will appear in real-time as they're detected

## Current Status

Check current WebSocket connections:
```bash
curl http://localhost:8668/ws/stats
```

**Current Status:** 1 active connection (from the demo/test)

## Ping/Pong Keepalive

Clients can send `"ping"` messages to keep the connection alive. The server responds with `"pong"`.

**Recommended:** Send ping every 30 seconds to maintain connection.

## Use Cases

1. **Real-time Dashboard**: Display events as they happen
2. **Alert System**: Trigger notifications when specific events occur
3. **Live Monitoring**: Monitor camera activity in real-time
4. **Event Logging**: Log all events to external systems
5. **Analytics**: Process events for real-time analytics

## Troubleshooting

### Connection Issues
- Check server is running: `curl http://localhost:8668/health`
- Verify WebSocket endpoint: `curl http://localhost:8668/ws/stats`
- Check firewall allows WebSocket connections (port 8668)

### No Events Received
- Verify cameras are running and processing
- Check AI models are detecting events
- Verify events are being saved to database
- Check server logs for broadcast errors

### Connection Drops
- Implement reconnection logic in client
- Use ping/pong to keep connection alive
- Check network stability
- Review server logs for errors

