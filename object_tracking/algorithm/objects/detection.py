import cv2

image = cv2.imread('../../Images/people.jpg')

detector = cv2.CascadeClassifier('../../cascade/fullbody.xml')
#Recommended to transform to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# See gray image
#cv2.imshow('People', image_gray)
#cv2.waitKey(0)

detections = detector.detectMultiScale(image_gray)

for (x,y,w,h) in detections:
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows