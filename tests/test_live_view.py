"""Live view - SmokeFireModel with cv2.imshow"""
import cv2
import time
from src.ai_models.smoke_fire_model import SmokeFireModel

STREAM_URL = "rtsp://go2rtc.pathtech.net:8554/fire?mp4"

model = SmokeFireModel(model_path='src/ai_models/model_weights/fire_smoke.pt', detection_time_threshold=0)
print(f"Device: {model.device}")

cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"[ERROR] Cannot open stream: {STREAM_URL}")
    exit(1)

print("Stream opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    result = model.process_frame(frame, annotate=True)
    display = result.frame if result.frame is not None else frame

    if result.metadata.get('count', 0) > 0:
        for d in result.metadata.get('detections', []):
            print(f"[DETECT] {d['class_name']} conf={d['confidence']:.2f}")

    cv2.imshow("SmokeFireModel - Live", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
