#ETLE
import numpy as np
import pandas as pd
import cv2

cap = cv2.VideoCapture("Background tetap Objek Gerak.mp4")

#bagian background subtraction
foreground_background_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while(True):
    ret, frame = cap.read()
    if not ret:
        break

    #terapkan background subtraction untuk mendapatkan foreground mask
    foreground_mask = foreground_background_subtractor.apply(frame)
    
    #Threshold
    _, thresh = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('original', frame)
    cv2.imshow('foregroundMask', thresh)

    key = cv2.waitKey(30)
    if key == 27: #ESC key to exit
        break

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 500: #filter berdasarkan area kontur
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cap.release()
cv2.destroyAllWindows()




