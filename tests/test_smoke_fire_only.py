"""
Test SmokeFireModel với RTSP stream - chỉ SmokeFireModel, in kết quả ra console
"""
import cv2
import numpy as np
import time
import logging

# Bật logging để thấy detection logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

from src.ai_models.smoke_fire_model import SmokeFireModel

STREAM_URL = "rtsp://go2rtc.pathtech.net:8554/fire?mp4"

print(f"[INFO] Khởi tạo SmokeFireModel...")
model = SmokeFireModel(detection_time_threshold=0)  # Tắt temporal filtering để test nhanh
print(f"[INFO] Model loaded thành công! Device: {model.device}")

print(f"[INFO] Mở stream: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"[ERROR] Không mở được stream: {STREAM_URL}")
    exit(1)

print(f"[INFO] Stream opened. Bắt đầu đọc frames...")

frame_count = 0
detection_count = 0
start_time = time.time()
MAX_FRAMES = 100  # Chỉ xử lý 100 frames rồi dừng

try:
    while frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Không đọc được frame, retry...")
            time.sleep(0.5)
            continue

        frame_count += 1
        result = model.process_frame(frame, annotate=False)

        n_detections = result.metadata.get('count', 0)
        if n_detections > 0:
            detection_count += 1
            detections = result.metadata.get('detections', [])
            for d in detections:
                print(f"  [DETECT] Frame {frame_count}: {d['class_name']} "
                      f"conf={d['confidence']:.2f} bbox={d['bbox']} track_id={d.get('track_id')}")

        if result.event:
            violations = result.metadata.get('violations', [])
            for v in violations:
                print(f"  [EVENT!] Frame {frame_count}: {v['violation_type']} "
                      f"conf={v['confidence']:.2f} track_id={v.get('track_id')}")

        # In progress mỗi 10 frames
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"[PROGRESS] Frame {frame_count}/{MAX_FRAMES} | "
                  f"FPS: {fps:.1f} | Detections so far: {detection_count}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
finally:
    cap.release()
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"[DONE] Processed {frame_count} frames in {elapsed:.1f}s")
    print(f"[DONE] Average FPS: {fps:.1f}")
    print(f"[DONE] Frames with detections: {detection_count}/{frame_count}")
    print(f"{'='*60}")
