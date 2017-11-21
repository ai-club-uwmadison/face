import matplotlib.pyplot as plt
import numpy as np
import cv2

frontFace_xml_path = 'haarcascade_frontalface_default.xml'
sideFace_xml_path = 'haarcascade_profileface.xml'

frontFace_cascade = cv2.CascadeClassifier(frontFace_xml_path)
sideFace_cascade = cv2.CascadeClassifier(sideFace_xml_path)

target_size = (128, 128)

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def faceRetriever(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')

    result = []

    faces = frontFace_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(50,50))
    for (x,y,w,h) in faces:
        crop_img = cv2.resize(img[y:y+h, x:x+w], target_size)
        result.append((crop_img, (x,y,w,h)))
        
    faces = sideFace_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(50,50))
    for (x,y,w,h) in faces:
        crop_img = cv2.resize(img[y:y+h, x:x+w], target_size)
        result.append((crop_img, (x,y,w,h)))

    return result