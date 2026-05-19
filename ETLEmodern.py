from ultralytics import YOLO
import cv2
import time
import numpy as np
import math

model = YOLO('yolov8m.pt')
video_path = 'Background tetap Objek Gerak.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

LINE_Y1 = 200
LINE_Y2 = 400

real_world_distance = 10  # meters

vehicle_times = {}
vehicle_speeds = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)  # Filter for vehicles (car, motorcycle, bus, truck)

    cv2.line(frame, (0, LINE_Y1), (frame.shape[1], LINE_Y1), (0, 255, 0), 2)
    cv2.line(frame, (0, LINE_Y2), (frame.shape[1], LINE_Y2), (0, 255, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

            current_time = time.time()

            if LINE_Y1 - 5 <= cy <= LINE_Y1 + 5:
                if track_id not in vehicle_times:
                    vehicle_times[track_id] = current_time

            if LINE_Y2 - 5 <= cy <= LINE_Y2 + 5:
                if track_id in vehicle_times and track_id not in vehicle_speeds:
                    time_taken = current_time - vehicle_times[track_id]
                    if time_taken > 0:
                        speed_mps = real_world_distance / time_taken
                        speed_kmph = speed_mps * 3.6
                        vehicle_speeds[track_id] = speed_kmph


            if track_id in vehicle_speeds:
                speed_text = f'ID: {int(track_id)} Speed: {vehicle_speeds[track_id]:.2f} km/h'
            else : 
                speed_text = f'ID: {int(track_id)} Speed: Calculating...'
            
            cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)        

    cv2.imshow("Vehicle Speed Estimation", frame)    

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()