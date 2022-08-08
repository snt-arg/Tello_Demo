#using self-made keras model
import cv2
from djitellopy import tello
from keras.models import load_model
import mediapipe as mp
import numpy as np
import time

#Drone = tello.Tello()
#Drone.connect()
#Drone.streamon()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

model = load_model('signlanguage.h5')
 
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
_, img = cap.read()
h,w,c =img.shape
offset = 40
img_counter = 0
analysisframe = ''
Possible_commands = ['Go','Stop','Up']


#Tello Phases
class Tello_Phase:
    State = None
     
    def SwitchState(self, switch):
        if switch == 1:
            self.State = 1
        elif switch == 2:
            self.State = 2
        elif switch == 3:
            self.State = 3
        else:
            self.State = None
while True:

#getting stream+ begin of "hand"
    '''img = Drone.get_frame_read().frame'''
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    hand_landmarks = results.multi_hand_landmarks
    analysisframe = img
    showframe = analysisframe

#hand

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h,w,c =img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 4:
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#box for hand
            
            continue
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (0, 255, 0), 2)
            mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS)

#frames per second on main image

    
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
            cv2.imshow("Image", img)
#resizing shape for keras

            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            cv2.imshow("Frame", showframe)
            analysisframe = cv2.resize(analysisframe,(128,128))
            analysisframe = np.reshape(analysisframe,[1,128,128,3])
            analysisframe = analysisframe/255.0

            #RESIZING PROBLEM, again....(screenshot saved)

            #implement condition for prediction

            prediction = model.predict(analysisframe)
            predarray = np.array(prediction[0])
            letter_prediction_dict = {Possible_commands[i]: predarray[i] for i in range(len(Possible_commands))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key,value in letter_prediction_dict.items():
                if value==high1:
                    print("Order: ", key)
                    print('Confidence 1: ', 100*value)
                elif value==high2:
                    print("Order: ", key)
                    print('Confidence 2: ', 100*value)
                elif value==high3:
                    print("Order ", key)
                    print('Confidence 3: ', 100*value)

            cv2.imshow("Frame", img)

            key =cv2.waitKey(1)
            if key == ord("q"):
                #Drone.streamoff()
                cap.release()
                cv2.destroyAllWindows()
                break
            

    '''Different ways to see key input

    1. 
    key = cv2.waitKey(1)
    if key == ord("q"):
        do something

    2.
    key = cv2.waitKey(1)
    if k%256 == special interger number which represents a key:         (example: 27 corresponds to esc)
        do something
        '''
