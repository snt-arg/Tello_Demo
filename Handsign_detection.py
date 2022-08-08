#Handsign_recognition
#Not good design--> based on hand_photographer= too much unnecessary stuff
#Impure--> uses Cvzone(I don't like it-->should be able to work only on opencv--> less packages) 
#only works with right keras model

#latest problem--> training Keras model to be somehow sufficient

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from tensorflow.keras.models import load_model
from djitellopy import tello
import cv2
import numpy as np
import math
import time 

#Drone = tello.Tello()
#Drone.connect()
#Drone.streamon()
cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
classifier = Classifier("keras_model.h5","labels.txt")
#Classifier = load_model("mp_handgesture")
labels = ["Go","Stop","Up"]

while True:
    #img = Drone.get_frame_read().frame
    success,img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3),np.uint8) *255
        imgCrop = img[y - offset:y + h + offset , x - offset:x + w + offset]

        imgCropShape = imgCrop.shape
 
        aspectRatio = h/w

        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            cv2.imshow("tello",img)
            key = cv2.waitKey(1)
            
            continue

        else:
            if aspectRatio >1:
                k = imgSize/h
                wCal = math.ceil((k*w))

                imgResize = cv2.resize(imgCrop,(wCal,imgSize))

                imgResizeshape = imgResize.shape
                wGap = math.ceil((imgSize-wCal))
                imgWhite[:,wGap:wCal+wGap] = imgResize
                prediction, index = classifier.getPrediction(img)
                print(prediction, index)

            else:
                k = imgSize/w
                hCal = math.ceil((k*h) )

                imgResize = cv2.resize(imgCrop,(imgSize,hCal))

                imgResizeshape = imgResize.shape
                hGap = math.ceil((imgSize - hCal))
                imgWhite[hGap:hCal +hGap,:] = imgResize
                prediction, index = classifier.getPrediction(img)
                print(prediction, index)

        cv2.imshow("imgageWhite",imgWhite)
        
    cv2.imshow("tello",img)
    key = cv2.waitKey(1)
    if key ==ord("q"):
        break
