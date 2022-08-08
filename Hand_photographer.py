#Tello_hand_position_photographer

#there is a problem with the export of the images of this script: they are simply put in the same ordner you are working
#This script is compatible with the Tello drone
from cvzone.HandTrackingModule import HandDetector
from djitellopy import tello
import cv2
import numpy as np
import math
import time 

'''Drone = tello.Tello()
Drone.connect()
Drone.streamon()'''
#comment/uncomment those lines if you want to implement/exclude Tello
#for Tello

cap = cv2.VideoCapture(0)
#comment/uncomment this line(16) if you want to implement/exclude Tello
#for Webcam

detector = HandDetector(maxHands=1)
counter = 0
offset = 20
imgSize = 300
folder = "Tello_Stop"
while True:   
    '''img = Drone.get_frame_read().frame'''
    #for Tello
    success,img = cap.read()
    #for Webcam
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
            else:
                k = imgSize/w
                hCal = math.ceil((k*h) )

                imgResize = cv2.resize(imgCrop,(imgSize,hCal))

                imgResizeshape = imgResize.shape
                hGap = math.ceil((imgSize - hCal))
                imgWhite[hGap:hCal +hGap,:] = imgResize

        cv2.imshow("imgageWhite",imgWhite)
        
    cv2.imshow("tello",img)
    key = cv2.waitKey(1)

    if key == ord("s"):
        cv2.imwrite(f'{folder}Img_{time.time()}.jpg', imgWhite)
        counter += 1
        print(counter)
    if key ==ord("q"):
        #Drone.streamoff()
        break
