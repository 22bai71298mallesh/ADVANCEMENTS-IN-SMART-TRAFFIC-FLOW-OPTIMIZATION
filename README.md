# ADVANCEMENTS-IN-SMART-TRAFFIC-FLOW-OPTIMIZATION
Program code:
pip install ultralytics
!pip uninstall torch torchvision torchaudio -y
!pip cache purge
!pip install torch torchvision torchaudio ultralytics --index-url https://download.pytorch.org/whl/cu118
import cv2
import torch
import numpy as np
import math
from ultralytics import YOLO

# ✅ Load YOLOv8 Model
model = YOLO("yolov8n.pt")  # Pre-trained small model

# ✅ Define emergency vehicle classes
emergency_classes = {"ambulance", "fire_truck", "police_car"}

# ✅ Kalman Filter for tracking vehicles
class KalmanFilterTracker:
    def _init_(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        return self.kf.predict()

    def correct(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        return self.kf.predict()

# ✅ Function to Estimate Speed
def estimate_speed(prev, curr, fps, scale_factor=0.1):
    distance = math.hypot(curr[0] - prev[0], curr[1] - prev[1])  # Pixel distance
    real_distance = distance * scale_factor  # Convert pixels to meters
    speed_mps = real_distance * fps  # Speed in meters per second
    speed_kph = speed_mps * 3.6  # Convert to km/h
    return speed_kph

# ✅ Load Video
video_path = "traffic (1).mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}.")
    exit()

# ✅ Video Properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_video = cv2.VideoWriter("processed_video.avi", fourcc, fps, (frame_width, frame_height))

# ✅ Tracking Dictionary
trackers = {}  # {object_id: KalmanFilterTracker}
previous_positions = {}  # {object_id: (x, y)}

frame_count = 0  # To limit processing

# ✅ Process Video Frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)

    emergency_detected = False
    vms_message = "Traffic Normal"
    current_positions = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)

            color = (0, 255, 0)  # Green for normal vehicles

            # ✅ Emergency Vehicle Detection
            if class_name in emergency_classes:
                emergency_detected = True
                color = (0, 0, 255)  # Red for emergency vehicles
                vms_message = "🚨 Emergency Vehicle Detected! Give Way!"

            # ✅ Track Vehicles with Kalman Filter
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            matched = False

            for object_id, tracker in trackers.items():
                prediction = tracker.predict()
                pred_x, pred_y = int(prediction[0]), int(prediction[1])

                if math.hypot(center_x - pred_x, center_y - pred_y) < 50:
                    new_position = tracker.correct(center_x, center_y)
                    current_positions[object_id] = (int(new_position[0]), int(new_position[1]))

                    if object_id in previous_positions:
                        speed = estimate_speed(previous_positions[object_id], current_positions[object_id], fps)
                        cv2.putText(frame, f"ID {object_id} Speed: {speed:.1f} km/h", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    matched = True
                    break

            if not matched:
                new_id = len(trackers) + 1
                trackers[new_id] = KalmanFilterTracker()
                trackers[new_id].correct(center_x, center_y)
                current_positions[new_id] = (center_x, center_y)

            # ✅ Draw Bounding Box and Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ✅ Update Previous Positions
    previous_positions = current_positions.copy()

    # ✅ Display VMS Message
    cv2.putText(frame, vms_message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # ✅ Write Frame to Output Video
    output_video.write(frame)

    # ✅ Stop After 500 Frames for Testing (Remove This in Production)
    if frame_count > 500:
        break

cap.release()
output_video.release()
cv2.destroyAllWindows()
