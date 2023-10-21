import cv2
import numpy as np

cap = cv2.VideoCapture('../../Videos/walking.avi')

ok, first_frame = cap.read()
frame_gray_init = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)
hsv[...,1] = 255

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        frame_gray_init, frame_gray, None, 0.5, 3, 15, 3, 5, 1.1, 0
    )
    magnitudes, angle = cv2.cartToPolar(flow[...,0], flow[...,1])  # X, Y
    hsv[...,0] = angle * (180/(np.pi / 2))
    hsv[...,2] = cv2.normalize(magnitudes, None, 0, 255, cv2.NORM_MINMAX)

    frame_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Dense opticalflow', frame_rgb)

    if cv2.waitKey(1) == 13: #enter
        break

    frame_gray_init = frame_gray

cv2.destroyAllWindows()
cap.release()
