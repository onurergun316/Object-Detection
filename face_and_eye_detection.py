import cv2
import matplotlib.pyplot as plt
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('oscar_selfie.jpg')
fixed_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

faces = face_classifier.detectMultiScale(fixed_image,1.3,5)
if faces is():
    print('No faces found')

def detect_faces_and_eyes(fixed_image):
    face_rects = face_classifier.detectMultiScale(fixed_image)
    for(x,y,w,h) in face_rects:
        cv2.rectangle(fixed_image, (x,y), (x+w,y+h), (255,0,0), 7) 
    
    eye_rects = eye_classifier.detectMultiScale(fixed_image)
    for(ix,iy,iw,ih) in eye_rects:
        cv2.rectangle(fixed_image, (ix,iy), (ix+iw,iy+ih), (0,0,255), 5)
    return fixed_image
plt.imshow(detect_faces_and_eyes(fixed_image))
plt.show()