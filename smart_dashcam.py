import cv2
from ultralytics import YOLO
import cvzone

# Load a pre-trained YOLOv8 model (n is for 'nano' - fastest for real-time)
model = YOLO('yolov8n.pt') 

# Initialize Dashcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

def alert_driver(message):
    print(f"ALARM: {message}")
    # Integration point for audio feedback (e.g., using pyttsx3)

def report_to_police(vehicle_id, violation_type):
    print(f"REPORTING: Vehicle {vehicle_id} committed {violation_type} to nearest station.")
    # Integration point for Twilio API or Email to send data

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Detect objects in the frame
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates and class info
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # 2. Hazard Detection Logic (Simplified)
            # If a person or obstacle is too close to the center-bottom of the frame
            if label in ['person', 'dog', 'obstacle'] and y2 > 400:
                alert_driver("Immediate Hazard Ahead!")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

            # 3. Violation Logic (Example: Wrong Way/Illegal Lane)
            # In a real scenario, you'd track the 'centroid' movement over frames
            if label == 'car':
                # Hypothetical: if car is in your lane and moving toward you
                is_violating = False # Replace with trajectory logic
                if is_violating:
                    # In a production app, use OCR (EasyOCR/Tesseract) to read plate here
                    report_to_police("XYZ-123", "Wrong-way driving")

    # Display the feed
    cv2.imshow("Smart Dashcam AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()