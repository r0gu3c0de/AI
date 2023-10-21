import cv2
import time
from imutils.video import VideoStream
import numpy as np

cap = VideoStream(src=0).start() #Start the webcam
time.sleep(1.0) #wait 1 second

cap = cv2.VideoCapture(0)
ok, frame = cap.read()
bbox = cv2.selectROI(frame)
x, y, w, h = bbox
track_window = (x, y, w, h)
roi = frame[y:y+h, x:x+y]  #RGB -> BGR

# HSV
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0,180] ) #180 because we are using hsv

# 0 - 255
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
parameters = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ok, frame = cap.read()
    if ok == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1) #comparing the current frame with the histogram

        ok, track_window = cv2.CamShift(dst, (x, y, w, h), parameters)

        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        pts = cv2.boxPoints(ok)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow('Camshift', img2)

        if cv2.waitKey(1) == 13: #enter
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()