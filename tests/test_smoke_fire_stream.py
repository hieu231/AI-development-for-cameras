
import cv2
import numpy as np
import time
from src.ai_models.smoke_fire_model import SmokeFireModel
from src.ai_models.people_control_model import PeopleControlModel


# Link RTSP stream (hỗ trợ tốt với OpenCV)
STREAM_URL = "rtsp://go2rtc.pathtech.net:8554/fire?mp4"


# Khởi tạo cả 2 model
smoke_fire_model = SmokeFireModel(device='cuda')
people_control_model = PeopleControlModel(device='cuda')


cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"[ERROR] Không mở được stream: {STREAM_URL}")
    exit(1)


while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Không đọc được frame từ stream!")
        time.sleep(0.5)
        continue

    # Xử lý với cả 2 model, overlay annotation lên cùng 1 frame
    smoke_result = smoke_fire_model.process_frame(frame, annotate=True)
    # Dùng annotated frame của smoke làm nền
    combined = smoke_result.frame.copy() if smoke_result.frame is not None else frame.copy()

    # Vẽ annotation của people_control_model lên combined
    people_result = people_control_model.process_frame(combined, annotate=True)
    if people_result.frame is not None:
        combined = people_result.frame

    cv2.imshow("Smoke/Fire + People Control", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
