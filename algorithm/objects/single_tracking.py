import cv2
import sys
from random import randint

#Types of algorithms
tracker_types = ['BOOSTING','MIL','KCF','TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[6]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting_create()
elif tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == 'TLD':
    tracker = cv2.legacy.TrackerTLD_create()
elif tracker_type == 'MEDIANFLOW':
    tracker = cv2.legacy.TrackerMedianFlow_create()
elif tracker_type == 'MOSSE':
    tracker = cv2.legacy.TrackerMOSSE_create()
elif tracker_type == 'CSRT':
    tracker = cv2.legacy.TrackerCSRT_create()


video = cv2.VideoCapture('../../Videos/race.mp4')
if not video.isOpened():
    print('Video not loaded')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Error while loading first frame')
    sys.exit()

bbox = cv2.selectROI(frame) #ROI Region of interest

ok = tracker.init(frame, bbox)
colors = (randint(0,255), randint(0,255), randint(0,255)) # It works like RGB -> BGR

while True:
    ok, frame = video.read()
    if not ok:
        # Stop execution when video ends
        break

    ok, bbox = tracker.update(frame)

    # Draw the bbox
    if ok == True:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x + w, y + h), colors, 2)
    else:
        # Put text in the frame in case it does not load the bbox rectangle in the frame
        cv2.putText(frame, 'Tracking failure', (20,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
    
    # Show the name of the algorithm in the video
    cv2.putText(frame, tracker_type, (20,20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)

    # Show the tracking rectangle in the frame
    cv2.imshow('Tracking', frame)

    # If to close the window with 'esc'
    if cv2.waitKey(1) & 0XFF == 27: # esc
        break
