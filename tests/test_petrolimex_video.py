"""Test PetrolimexDetectionModel - Full-res video + bbox overlay from GPU inference."""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
import logging
import torch

logging.basicConfig(level=logging.WARNING)

from src.ai_models.petrolimex_detection_model import PetrolimexDetectionModel

VIDEO_PATH = r"D:\PA\IMG_4792.mp4"
WINDOW_NAME = "Petrolimex Detection - Q to quit"
INFER_EVERY = 3   # Run inference every N frames
MAX_DIM = 640     # Inference resolution

os.environ["AIBE_DEVICE"] = "cuda"
model = PetrolimexDetectionModel()
model.model.to("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)} | Classes: {model.class_names}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO_PATH}")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {total_frames} frames | {fps_video:.0f} FPS")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 540, 960)

# GPU warmup
ret, warmup_frame = cap.read()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
if ret:
    h, w = warmup_frame.shape[:2]
    s = MAX_DIM / max(h, w)
    small = cv2.resize(warmup_frame, (int(w * s), int(h * s)))
    for _ in range(3):
        model.model.track(small, conf=0.45, persist=True, verbose=False, device="cuda")
    torch.cuda.synchronize()
print("Ready!")

# Stored bboxes from last inference (in original frame coords)
last_boxes = []  # [(x1,y1,x2,y2, cls_name, conf, tid), ...]

fc = 0
start = time.time()
delay = int(1000 / fps_video)  # ms per frame to match original FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fc += 1
    h, w = frame.shape[:2]

    # Run inference every N frames
    if fc % INFER_EVERY == 0:
        scale = MAX_DIM / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        results = model.model.track(
            source=small, conf=model.conf_threshold, iou=model.iou_threshold,
            persist=True, verbose=False, device="cuda",
        )

        # Extract bboxes and scale back to original resolution
        inv_scale = 1.0 / scale
        last_boxes = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for i, box in enumerate(boxes):
                sx1, sy1, sx2, sy2 = box.xyxy[0].tolist()
                # Scale bbox back to original frame size
                ox1 = int(sx1 * inv_scale)
                oy1 = int(sy1 * inv_scale)
                ox2 = int(sx2 * inv_scale)
                oy2 = int(sy2 * inv_scale)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.class_names.get(cls_id, "?")
                tid = track_ids[i] if i < len(track_ids) else None
                last_boxes.append((ox1, oy1, ox2, oy2, cls_name, conf, tid))

    # Draw cached bboxes on original frame
    display = frame
    for (x1, y1, x2, y2, cls_name, conf, tid) in last_boxes:
        color = (0, 255, 0) if cls_name == "vest" else (0, 0, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name} {conf:.2f}"
        if tid is not None:
            label += f" #{tid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    # Info overlay
    elapsed = time.time() - start
    fps_now = fc / elapsed if elapsed > 0 else 0
    cv2.putText(display, f"FPS:{fps_now:.0f} F:{fc}/{total_frames} D:{len(last_boxes)}",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow(WINDOW_NAME, display)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
elapsed = time.time() - start
print(f"DONE: {fc} frames in {elapsed:.1f}s | {fc/elapsed:.1f} FPS")
