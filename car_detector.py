import cv2
import numpy as np

car_classifier = cv2.CascadeClassifier('haarcascade_cars.xml')
capture = cv2.VideoCapture('video2.avi')
while capture.isOpened():
    # Reading the captured video 
    ret, frame = capture.read()

    # Passing the captured frame to Classifier
    cars = car_classifier.detectMultiScale(frame,1.4,2)

    if ret == True:
        # Bounding marking rectangles to detected bodies
        for(x,y,w,h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,225), 2) 
            cv2.imshow('Car Detector',frame)
        
        # exiting the programme with esc button, otherwise it will automatically re-open the video from the time it's been shut down 
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()



