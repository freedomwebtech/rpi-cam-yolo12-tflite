# Import required libraries
import cv2
from ultralytics import YOLO
import cvzone
from picamera2 import Picamera2
import time

# Load YOLOv8 model
model = YOLO('yolo12n_float32.tflite')
names = model.names



# Setup PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800, 500)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
time.sleep(2)

# Define the mouse callback function
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

# Create a named OpenCV window and set the mouse callback
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

frame_count = 0

while True:
    # Capture frame from PiCamera2
    frame = picam2.capture_array()
    frame=cv2.flip(frame,-1)
    frame_count += 1
    if frame_count % 2 != 0:
        continue  # Skip odd frames to reduce processing

    # Detect and track persons (class 0)
    results = model.track(frame, persist=True, classes=[0], imgsz=240)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        for box, track_id, class_id in zip(boxes, ids, class_ids):
            x1, y1, x2, y2 = box
            name = names[class_id]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cvzone.putTextRect(frame, f'{name}', (x1, y1), 1, 1)

    

    # Show the frame
    cv2.imshow("RGB", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
picam2.close()
