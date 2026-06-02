from ultralytics import YOLO
import cv2
import time
import numpy as np
import math

model = YOLO('best.pt')
video_path = "based_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

LINE_Y1 = 345
LINE_Y2 = 461

real_world_distance = 18  # meters

vehicle_times_in = {}
vehicle_times_out = {}  
vehicle_speeds = {}
previous_positions = {}
all_tracked_ids = set()

while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.4, iou=0.5, verbose=False)  # Filter for vehicles (car, motorcycle, bus, truck)

    cv2.line(frame, (0, LINE_Y1), (frame.shape[1], LINE_Y1), (0, 255, 0), 2)
    cv2.line(frame, (0, LINE_Y2), (frame.shape[1], LINE_Y2), (0, 255, 0), 2)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):
            track_id = int(track_id)
            all_tracked_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = y2

            if track_id in previous_positions:
                prev_cy = previous_positions[track_id]
#Kendaraan lewat garis pertama
                if prev_cy < LINE_Y1 and cy >= LINE_Y1:

                    if track_id not in vehicle_times_in:
                        vehicle_times_in[track_id] = frame_count
         #Kendaraan Lewat garis kedua       
                if prev_cy < LINE_Y2 and cy >= LINE_Y2:
                    if track_id in vehicle_times_in and track_id not in vehicle_speeds:
                        frame_difference = frame_count - vehicle_times_in[track_id]
                        time_taken = frame_difference / fps

                        if time_taken > 0:
                            speed_mps = real_world_distance / time_taken
                            speed_kmph = speed_mps * 3.6

                            vehicle_speeds[track_id] = speed_kmph
           #Outgoing Vehicle                 
                if prev_cy > LINE_Y2 and cy <= LINE_Y2:
                    if track_id not in vehicle_times_out:
                        vehicle_times_out[track_id] = frame_count

                if prev_cy > LINE_Y1 and cy <= LINE_Y1:
                    if track_id in vehicle_times_out and track_id not in vehicle_speeds:
                        frame_difference = frame_count - vehicle_times_out[track_id]
                        time_taken = frame_difference / fps

                        if time_taken > 0:
                            speed_mps = real_world_distance / time_taken
                            speed_kmph = speed_mps * 3.6

                            vehicle_speeds[track_id] = speed_kmph

            previous_positions[track_id] = cy

            box_color = (0 ,255, 0)

            if track_id in vehicle_speeds:
                if vehicle_speeds[track_id] > 80: 
                    box_color = (0, 0, 255)

                speed_text = f'ID: {int(track_id)} Speed: {vehicle_speeds[track_id]:.2f} km/h'
            else : 
                speed_text = f'ID: {int(track_id)} Speed: Calculating...'
            cv2.rectangle(frame, (x1,y1), (x2,y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 5, box_color, - 1)
            cv2.putText(frame, speed_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2) 

            print(f"ID : {track_id}, CY: {cy}") 

    cv2.putText(frame, "Garis Atas", (20, LINE_Y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Garis Bawah", (20, LINE_Y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    total_vehicles = len(all_tracked_ids)
    calculated_vehicles = len(vehicle_speeds)
    uncalculated_vehicles = total_vehicles - calculated_vehicles

    cv2.putText(frame, f"Total ID : {total_vehicles}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0),2)
    cv2.putText(frame, f"Calculated : {calculated_vehicles}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (165,255,0),2)
    cv2.putText(frame, f"Uncalculated : {uncalculated_vehicles}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255),2)


    cv2.imshow("Vehicle Speed Estimation", frame)    

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


#Ringkasan Akhir
print()
print("Laporan Akurasi Deteksi Kecepatan")
final_total = len(all_tracked_ids)
final_calculated = len(vehicle_speeds)
final_uncalculated = final_total - final_calculated

print(f"Total Kendaraan Terdeteksi (ID Unik)    : {final_total}")
print(f"Berhasil Dihitung Kecepatannya          : {final_calculated}")
print(f"Gagal / Tidak Selesai Dihitung          : {final_uncalculated}")

if final_total > 0 :
    akurasi = (final_calculated / final_total) * 100
    print(f"Tingkat keberhasilan deteksi (success rate)    : {akurasi:.2f}%")
else : 
    print("Tidak ada kendaraan yang terdeteksi")