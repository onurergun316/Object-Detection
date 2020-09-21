import cv2
import numpy

body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
capture = cv2.VideoCapture('pedestrian2.mp4')

while capture.isOpened():
    # Reading the captured video 
    ret, frame = capture.read()

    # Passing the captured frame to Classifier
    bodies = body_classifier.detectMultiScale(frame,1.2,3)

    if ret == True:
        # Bounding marking rectangles to detected bodies
        for(x,y,w,h) in bodies:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (25,125,225), 5) 
            cv2.imshow('Pedesterian Detector',frame)
        
        # exiting the programme with esc button, otherwise it will automatically re-open the video from the time it's been shut down 
        if cv2.waitKey(1) == 27:
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()

