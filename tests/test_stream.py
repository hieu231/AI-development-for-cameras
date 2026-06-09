#!/usr/bin/env python3
"""
Interactive script to test video streams.

Modes:
  1. MJPEG from API - Test stream from backend (camera must be running)
  2. Direct URL - Test RTSP/HTTP stream connectivity
  3. List cameras - Show cameras and their stream status

Usage:
  python test_stream.py                          # Interactive mode
  python test_stream.py --list                   # List cameras and exit
  python test_stream.py --camera-id <uuid>       # MJPEG stream from API
  python test_stream.py --url "rtsp://..."      # Direct stream URL
  python test_stream.py --api-url http://localhost:8668
"""
import argparse
import sys
import time
import threading
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import tkinter as tk
from PIL import Image, ImageTk

# FFmpeg options for RTSP
import os
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|max_delay;5000000|timeout;10000000",
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def fetch_cameras(api_url: str):
    """Fetch camera list from API."""
    try:
        import requests
        r = requests.get(f"{api_url.rstrip('/')}/api/cameras/", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Cannot connect to API at {api_url}. Is the server running?")
        return []
    except Exception as e:
        print(f"[ERROR] API: {e}")
        return []


def fetch_stream_status(api_url: str, camera_id: str):
    """Fetch stream status for a camera."""
    try:
        import requests
        r = requests.get(
            f"{api_url.rstrip('/')}/api/stream/camera/{camera_id}/status",
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"online": False, "message": str(e)}


def list_cameras(api_url: str) -> int:
    """List cameras and their stream status. Returns 0 on success."""
    cameras = fetch_cameras(api_url)
    if not cameras:
        print("No cameras found or API unreachable.")
        return 1

    print(f"\nCameras (API: {api_url})")
    print("-" * 70)
    for i, cam in enumerate(cameras):
        cid = cam.get("id", "?")
        name = cam.get("name", "?")
        status = "[Running]" if cam.get("status") else "[Stopped]"
        url_preview = (cam.get("rtsp_url") or "?")[:50] + "..."
        print(f"  [{i}] {name}")
        print(f"      ID: {cid}")
        print(f"      Status: {status}")
        print(f"      URL: {url_preview}")
        # Fetch stream status if running
        if cam.get("status"):
            st = fetch_stream_status(api_url, cid)
            online = st.get("online", False)
            fresh = st.get("frames_fresh", False)
            msg = st.get("message", "")
            print(f"      Stream: {'online' if online else 'offline'} | frames_fresh={fresh} | {msg}")
        print()
    return 0


# ---------------------------------------------------------------------------
# Stream display (tkinter)
# ---------------------------------------------------------------------------
def run_display(
    stream_url: str,
    title: str = "Stream Test",
    is_mjpeg_api: bool = False,
):
    """Open stream and display in tkinter window."""
    backend = cv2.CAP_FFMPEG if not is_mjpeg_api else cv2.CAP_FFMPEG
    cap = cv2.VideoCapture(stream_url, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Connecting to: {stream_url[:80]}...")
    time.sleep(1)

    if not cap.isOpened():
        print("[ERROR] Failed to open stream")
        if is_mjpeg_api:
            print("   Hint: Ensure camera is started (PUT /api/cameras/{id}/start)")
        return 1

    print("[OK] Stream connected")

    frame_count = 0
    start_time = time.time()
    running = True
    current_frame = None

    def process_frames():
        nonlocal frame_count, current_frame, running
        while running:
            ret, frame = cap.read()
            if not ret:
                if not running:
                    break
                time.sleep(0.05)
                continue
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame, f"FPS: {fps:.1f} | Frame: {frame_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            current_frame = frame
        cap.release()

    def update_display():
        nonlocal current_frame
        if current_frame is not None and running:
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame_rgb = cv2.resize(
                    frame_rgb, (1280, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
        if running:
            label.after(10, update_display)

    def on_closing():
        nonlocal running
        running = False
        elapsed = time.time() - start_time
        print(f"\n[Stopped] Frames: {frame_count} | Avg FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "\n[Stopped]")
        root.quit()
        root.destroy()

    root = tk.Tk()
    root.title(title)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    label = tk.Label(root)
    label.pack()

    t = threading.Thread(target=process_frames, daemon=True)
    t.start()
    update_display()
    root.mainloop()
    running = False
    t.join(timeout=2)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test video streams (MJPEG from API or direct RTSP/HTTP)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8668",
        help="API base URL (default: http://localhost:8668)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cameras and stream status, then exit",
    )
    parser.add_argument(
        "--camera-id",
        metavar="UUID",
        help="Test MJPEG stream for this camera (must be running)",
    )
    parser.add_argument(
        "--url",
        metavar="URL",
        help="Test direct stream URL (rtsp://, http://, etc.)",
    )
    args = parser.parse_args()

    if args.list:
        return list_cameras(args.api_url)

    if args.url:
        return run_display(
            args.url,
            title="Direct Stream Test",
            is_mjpeg_api=False,
        )

    if args.camera_id:
        mjpeg_url = f"{args.api_url.rstrip('/')}/api/stream/camera/{args.camera_id}/mjpeg"
        return run_display(
            mjpeg_url,
            title=f"MJPEG Camera {args.camera_id[:8]}",
            is_mjpeg_api=True,
        )

    # Interactive mode
    print("Stream Test - Interactive")
    print("=" * 50)
    cameras = fetch_cameras(args.api_url)
    if not cameras:
        print("No cameras. Enter a direct stream URL to test.")
        url = input("URL (rtsp:// or http://): ").strip()
        if url:
            return run_display(url, title="Direct Stream", is_mjpeg_api=False)
        return 1

    print("\nCameras:")
    for i, cam in enumerate(cameras):
        status = ">" if cam.get("status") else "x"
        print(f"  [{i}] {status} {cam.get('name', '?')} ({cam.get('id', '?')[:8]}...)")
    print("  [u] Enter custom URL")
    print("  [q] Quit")

    choice = input("\nChoice: ").strip().lower()
    if choice == "q":
        return 0
    if choice == "u":
        url = input("URL: ").strip()
        if url:
            return run_display(url, title="Direct Stream", is_mjpeg_api=False)
        return 1

    try:
        idx = int(choice)
        cam = cameras[idx]
    except (ValueError, IndexError):
        print("Invalid choice")
        return 1

    cid = cam.get("id")
    if not cam.get("status"):
        print(f"Camera {cam.get('name')} is not running.")
        start = input("Start it? (y/n): ").strip().lower()
        if start == "y":
            try:
                import requests
                r = requests.put(
                    f"{args.api_url.rstrip('/')}/api/cameras/{cid}/start",
                    timeout=10,
                )
                if r.status_code == 200:
                    print("Started. Waiting 3s...")
                    time.sleep(3)
                else:
                    print(f"Start failed: {r.status_code}")
                    return 1
            except Exception as e:
                print(f"Error: {e}")
                return 1
        else:
            return 1

    mjpeg_url = f"{args.api_url.rstrip('/')}/api/stream/camera/{cid}/mjpeg"
    return run_display(
        mjpeg_url,
        title=f"MJPEG - {cam.get('name', 'Camera')}",
        is_mjpeg_api=True,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
