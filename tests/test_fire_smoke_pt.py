"""Test fire_smoke.pt model with RTSP stream"""
import cv2, time, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
from src.ai_models.smoke_fire_model import SmokeFireModel

model = SmokeFireModel(model_path='src/ai_models/model_weights/fire_smoke.pt', detection_time_threshold=0)
print(f"Device: {model.device}")

cap = cv2.VideoCapture("rtsp://go2rtc.pathtech.net:8554/fire?mp4")
fc = dc = 0
st = time.time()

while fc < 100:
    ret, frame = cap.read()
    if not ret:
        continue
    fc += 1
    r = model.process_frame(frame, annotate=False)
    c = r.metadata.get("count", 0)
    if c > 0:
        dc += 1
        for d in r.metadata.get("detections", []):
            print(f"  [DETECT] Frame {fc}: {d['class_name']} conf={d['confidence']:.2f} bbox={d['bbox']}")
    if fc % 10 == 0:
        print(f"[PROGRESS] {fc}/100 | FPS: {fc/(time.time()-st):.1f} | Det: {dc}")

cap.release()
print(f"\nDONE: {fc} frames | {fc/(time.time()-st):.1f} FPS | {dc} frames with detections")
