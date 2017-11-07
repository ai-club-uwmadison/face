import matplotlib.pyplot as plt
import numpy as np
import cv2

face_xml_path = '/Users/nat/anaconda/envs/py3_5/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
#eye_xml_path = '/Users/nat/anaconda/envs/py3_5/share/OpenCV/haarcascades/haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_xml_path)
#eye_cascade = cv2.CascadeClassifier(eye_xml_path)

target_size = (128, 128)

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def faceRetriver(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    result = []
    for (x,y,w,h) in faces:
        crop_img = convertToRGB(cv2.resize(img[y:y+h, x:x+w], target_size))
        crop_img.append((result, (x,y,w,h)))

    return result