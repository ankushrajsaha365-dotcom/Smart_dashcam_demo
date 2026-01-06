import cv2
import torch
from ultralytics import YOLO

# ================== PERFORMANCE TUNING ==================
torch.set_num_threads(2)          # Prevent CPU meltdown
torch.set_grad_enabled(False)     # Inference only
FRAME_SKIP = 3                    # Run YOLO every N frames
CONF_THRESHOLD = 0.4              # Ignore weak detections
# =======================================================

# Load YOLOv8 nano (CPU-friendly)
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0

def alert_driver(message):
    print("ALARM:", message)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Skip frames to reduce load
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Smart Dashcam AI", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # YOLO inference (CPU optimized)
    results = model(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # -------- Hazard Detection --------
            if label == "person" and y2 > 350:
                alert_driver("Immediate Hazard Ahead!")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "HAZARD",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("Smart Dashcam AI", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
