import cv2
import numpy as np

cap = cv2.VideoCapture('../../Videos/walking.avi')

parameters_shiTomasi = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7)
parameters_klt = dict(winSize = (15,15), maxLevel = 2, 
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
colors = np.random.randint(0,255, (100, 3))

ok, frame = cap.read()

# Convert to grayscale
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.goodFeaturesToTrack(frame_gray_init, mask=None, **parameters_shiTomasi)

mask = np.zeros_like(frame)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray, edges, None, **parameters_klt)

    news = new_edges[status == 1]
    olds = edges[status == 1]

    for i, (new, old) in enumerate(zip(news, olds)):
        a, b = new.ravel()
        c, d = old.ravel()
        
        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a),int(b)), 5, colors[i].tolist(), -1)

    img = cv2.add(frame, mask)
    cv2.imshow('Opticalflow', img)
    if cv2.waitKey(1) == 13: #enter
        break

    frame_gray_init = frame_gray.copy()
    edges = news.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()