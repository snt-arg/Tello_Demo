#simple way to track hands. Can switch between computer camera and Tello(If you know what to change;).
import cv2
from djitellopy import tello
import mediapipe as mp
import time

#Drone = tello.Tello()
#Drone.connect()
#Drone.streamon()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)

while True:
    #img = Drone.get_frame_read().frame
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Output",img)
    key =cv2.waitKey(1)
    if key == ord("q"):
        #Drone.streamoff()
        break
