import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ok, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

parameters_klt = dict(winSize = (15,15), maxLevel = 4, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def selectPoint(event, x, y, flags, params):
    global point, selected_point, old_point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        selected_point = True
        old_point = np.array([[x, y]], dtype=np.float32)

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', selectPoint)

selected_point = False
point = ()
old_point = np.array([[]])
mask = np.zeros_like(frame)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is True:
        cv2.circle(frame, point, 5, (0,0,255), 2)

        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, old_point, None, **parameters_klt)


        frame_gray_init = frame_gray.copy()
        old_point = new_points

        x, y = new_points.ravel()
        j, k = old_point.ravel()

        mask = cv2.line(mask, (int(x),int(y)), (int(j),int(k)), (0, 255, 255), 2)
        frame = cv2.circle(frame, (int(x),int(y)), 5, (0, 255, 0), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Opticalflow webcam', img)

    if cv2.waitKey(1) == 27: #esc
        break

cv2.destroyAllWindows()
cap.release()