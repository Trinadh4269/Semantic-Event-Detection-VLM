import cv2
import time
from ultralytics import YOLO

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open video file
cap = cv2.VideoCapture("input_video.mp4")

person_threshold = 10  # crowd condition
fps_list = []

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Count persons
    person_count = 0
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # person class
            person_count += 1

    # Crowd detection
    if person_count >= person_threshold:
        cv2.putText(frame, "Crowded Scene Detected",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    fps_list.append(1 / (time.time() - start))

cap.release()

print("Average FPS (Before Optimization):", sum(fps_list) / len(fps_list))
