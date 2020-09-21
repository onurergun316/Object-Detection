import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('oscar_selfie.jpg')
fixed_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect_eyes(fixed_image):
    eyes_rects = eye_classifier.detectMultiScale(fixed_image)
    for(x,y,w,h) in eyes_rects:
        cv2.rectangle(fixed_image, (x,y), (x+w,y+h), (255,255,255), 10)
    return fixed_image

plt.imshow(detect_eyes(fixed_image))
plt.show()
