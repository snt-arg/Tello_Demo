#This script works. Tello receives commmands which it follows, even if it isn't alwys working. The base idea is now working: A lot of space for refinement.

import cv2
from djitellopy import tello
from keras.models import load_model
import mediapipe as mp
import numpy as np
import time

Drone = tello.Tello()
Drone.connect()
Drone.streamon()


Time = round(time.time(),1)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

key = cv2.waitKey(1)

model = load_model('signlanguage(Dude).h5')
 
pTime = 0
cTime = 0

#cap = cv2.VideoCapture(0)
offset = 40
img_counter = 0
analysisframe = ''
Tello_commands = ['A', 'B', 'C', 'D','H','S','U','V','W','Y']

while True:

#getting stream+ begin of "hand"
    img = Drone.get_frame_read().frame
    #success, img = cap.read()
    h,w,c =img.shape
    #success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    hand_landmarks = results.multi_hand_landmarks
    analysisframe = img
    showframe = analysisframe

#hand + box            
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
            #cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (0, 255, 0), 2)
            #mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS)

#Using landmarks or boxes for the hand confuses the handsignrecognition and results in an unnecessary complicated error 
# message if my hand touches the borders (with the marks on them). It will probably be removed completely in the next script.

#frames per second on main image
    
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

#This is a timer which counts constatly and allows every 10 seconds an evaluation of the hand's position(if there is a hand)                        
            CurrentTime = round(time.time(),1)
            Time_since_detection = (CurrentTime - Time)% 10.00

#resizing shape for keras
        if round(Time_since_detection,1) == 0:
            time.sleep(1)
            analysisframe = analysisframe[y_min:y_max, x_min:x_max]
            analysisframe = cv2.resize(analysisframe,(128,128))
            analysisframe = np.reshape(analysisframe,[1,128,128,3])
            analysisframe = analysisframe/255.0

            #RESIZING PROBLEM, again....(screenshot saved)
            
            prediction = model.predict(analysisframe)
            predarray = np.array(prediction[0])
            letter_prediction_dict = {Tello_commands[i]: predarray[i] for i in range(len(Tello_commands))}
            predarrayordered = sorted(predarray, reverse=True)
            high1 = predarrayordered[0]
            high2 = predarrayordered[1]
            high3 = predarrayordered[2]
            for key,value in letter_prediction_dict.items():
                if value==high1:
                    print("Order: ", key)
                    print('Confidence 1: ', 100*value)

#This way of putting conditions surely has a simplification...
#Problem: after handsign gets recognised, the screen freezes
                    if key == 'A'and Drone.is_flying == True:
                        Drone.move("up",(20))
                        print("move up")
                        cv2.putText(img, str(key+"move_up"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)
                    if key == 'B' and Drone.is_flying == False:
                        Drone.takeoff() 
                        print("take off")
                        cv2.putText(img, str(key+"takeoff"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

                    if key == 'C'and Drone.is_flying == True:
                        Drone.move("up",(20))
                        print("move_down")
                        time.sleep(5)
                        cv2.putText(img, str(key+"move_down"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

                    if key == 'D'and Drone.is_flying == True:
                        Drone.move("up",(20))
                        print("move_left")
                        cv2.putText(img, str(key+"move_left"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

                    if key == 'H'and Drone.is_flying == True:
                        Drone.move("up",(20))
                        print("move_right")
                        cv2.putText(img, str(key+"move_right"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)
            
                    if key == 'S' and Drone.is_flying == True:
                        Drone.flip_forward()
                        print("flip_forward")
                        cv2.putText(img, str(key+"flip_forward"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

                    if key == 'U'and Drone.is_flying == True:
                        Drone.flip_back()
                        print("flip_back")
                        cv2.putText(img, str(key+"flip_back"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)
    
                    if key == 'V'and Drone.is_flying == True:
                        Drone.rotate_clockwise(20)
                        print("rotate_clockwise")
                        cv2.putText(img, str(key+"rotate_clockwise"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)
        
                    if key == 'W'and Drone.is_flying == True:
                        Drone.rotate_counter_clockwise(20)
                        print("rotate_counter_clockwise")
                        cv2.putText(img, str(key+"rotate_counter_clockwise"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)
                        
                    if key == 'Y'and Drone.is_flying == True:
                        Drone.land()
                        print("land")
                        cv2.putText(img, str(key+"land"), (10, 70), cv2.FONT_ITALIC, 3,
                        (255, 0, 255), 3)

                    else:
                        pass
                elif value==high2:
                    print("Order: ", key)
                    print('Confidence 2: ', 100*value)
                elif value==high3:
                    print("Order ", key)
                    print('Confidence 3: ', 100*value)
        cv2.imshow("Frame", img)
#information which is only important to look back to in case of errors (or to evaluate keras algorithm accuracy, or precision, didn't catch the difference)

    else:
          cv2.imshow("Frame", img)  
    key =cv2.waitKey(1)
    if key == ord("q"):
        Drone.streamoff()
        #cap.release()
        cv2.destroyAllWindows()
        break