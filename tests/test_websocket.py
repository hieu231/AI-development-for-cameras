"""
WebSocket Test Client
Test the real-time event broadcasting via WebSocket
"""
import asyncio
import websockets
import json
import argparse
from datetime import datetime


async def test_all_events(url: str):
    """Test WebSocket connection for all events"""
    print(f"Connecting to {url}...")

    try:
        async with websockets.connect(url) as websocket:
            print("✓ Connected to WebSocket endpoint")
            print("Listening for events... (Press Ctrl+C to stop)\n")

            # Send ping to keep connection alive
            await websocket.send("ping")
            pong = await websocket.recv()
            print(f"Ping test: {pong}\n")

            # Listen for events
            while True:
                message = await websocket.recv()

                # Skip pong responses
                if message == "pong":
                    continue

                # Parse and display event
                try:
                    event_data = json.loads(message)
                    print("=" * 80)
                    print(f"🔔 NEW EVENT RECEIVED at {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 80)
                    print(f"Event ID:     {event_data.get('event_id')}")
                    print(f"Camera:       {event_data.get('camera_name')} ({event_data.get('camera_id')})")
                    print(f"Model:        {event_data.get('model_name')} ({event_data.get('model_type')})")
                    print(f"Time:         {event_data.get('time')}")
                    print(f"Image Path:   {event_data.get('image_path')}")
                    print(f"Metadata:     {json.dumps(event_data.get('metadata', {}), indent=2)}")
                    print("=" * 80)
                    print()

                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
    except KeyboardInterrupt:
        print("\n\nDisconnected by user")
    except Exception as e:
        print(f"✗ Error: {e}")


async def test_camera_events(url: str, camera_id: str):
    """Test WebSocket connection for specific camera events"""
    endpoint = f"{url}/{camera_id}"
    print(f"Connecting to {endpoint}...")

    try:
        async with websockets.connect(endpoint) as websocket:
            print(f"✓ Connected to WebSocket endpoint for camera {camera_id}")
            print("Listening for events... (Press Ctrl+C to stop)\n")

            # Send ping to keep connection alive
            await websocket.send("ping")
            pong = await websocket.recv()
            print(f"Ping test: {pong}\n")

            # Listen for events
            while True:
                message = await websocket.recv()

                # Skip pong responses
                if message == "pong":
                    continue

                # Parse and display event
                try:
                    event_data = json.loads(message)
                    print("=" * 80)
                    print(f"🔔 NEW EVENT RECEIVED at {datetime.now().strftime('%H:%M:%S')}")
                    print("=" * 80)
                    print(f"Event ID:     {event_data.get('event_id')}")
                    print(f"Camera:       {event_data.get('camera_name')} ({event_data.get('camera_id')})")
                    print(f"Model:        {event_data.get('model_name')} ({event_data.get('model_type')})")
                    print(f"Time:         {event_data.get('time')}")
                    print(f"Image Path:   {event_data.get('image_path')}")
                    print(f"Metadata:     {json.dumps(event_data.get('metadata', {}), indent=2)}")
                    print("=" * 80)
                    print()

                except json.JSONDecodeError:
                    print(f"Received non-JSON message: {message}")

    except websockets.exceptions.WebSocketException as e:
        print(f"✗ WebSocket error: {e}")
    except KeyboardInterrupt:
        print("\n\nDisconnected by user")
    except Exception as e:
        print(f"✗ Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Test WebSocket event broadcasting")
    parser.add_argument(
        "--url",
        default="ws://localhost:8668/ws/events",
        help="WebSocket URL (default: ws://localhost:8668/ws/events)"
    )
    parser.add_argument(
        "--camera-id",
        help="Camera ID to subscribe to specific camera events"
    )

    args = parser.parse_args()

    if args.camera_id:
        await test_camera_events(args.url, args.camera_id)
    else:
        await test_all_events(args.url)


if __name__ == "__main__":
    print("=" * 80)
    print("WebSocket Event Test Client")
    print("=" * 80)
    print()

    asyncio.run(main())
