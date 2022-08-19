#This script works. Tello receives commmands which it follows, even if it isn't alwys working. The base idea is now working: A lot of space for refinement.

from time import time
from google.protobuf.json_format import MessageToDict
import cv2
from djitellopy import tello
from keras.models import load_model
import mediapipe as mp
import numpy as np
import time

Drone = tello.Tello()
Drone.connect()
Drone.streamon()


StartTime = time.time()

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9,min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils


model = load_model('11Handsigns(30Ep).h5') 
pTime = 0
cTime = 0
#cap = cv2.VideoCapture(0)
offset = 40
img_counter = 0
analysisframe = ''
Tello_commands = ['Backwards', 'Down', 'Forwards', 'FrontFlip','Land','Left','Right','TakeOff','TurnLeft','TurnRight','Up']


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

#hand + box            
    if hand_landmarks:
         for i in results.multi_handedness:
            label = MessageToDict(i)[
                'classification'][0]['label']
            if label == 'Left':
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

#frames per second on main image
    
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3,
                            (255, 0, 255), 3)

    #This is a timer which counts constatly and allows every 10 seconds an evaluation of the hand's position(if there is a hand)                        
                CurrentTime = time.time()
                Time_since_detection = CurrentTime - StartTime
                print(Time_since_detection)

#resizing shape for keras
                if Time_since_detection >= 3 and len(analysisframe) != 0:
                    print(analysisframe.shape)
                    print(analysisframe)
                    analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                    if len(analysisframe) == 0:
                        break

                    analysisframe = cv2.resize(analysisframe,(128,128))
                    analysisframe = np.reshape(analysisframe,[1,128,128,3])
                    analysisframe = analysisframe/255.0
                    
                    prediction = model.predict(analysisframe)
                    predarray = np.array(prediction[0])
                    letter_prediction_dict = {Tello_commands[i]: predarray[i] for i in range(len(Tello_commands))}
                    predarrayordered = sorted(predarray, reverse=True)
                    high1 = predarrayordered[0]
                    for key,value in letter_prediction_dict.items():
                        if value==high1:
                            print("Order: ", key)
                            print('Confidence 1: ', 100*value)

                            if key == 'Backwards'and Drone.is_flying:
                                Drone.move_back(20)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)
                            if key == 'Forwards'and Drone.is_flying:
                                Drone.move_forward(40)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)

                            if key == 'Up'and Drone.is_flying:
                                Drone.move_up(40)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)

                            if key == 'TakeOff' and not Drone.is_flying:
                                Drone.takeoff() 
                                #Drone.go_xyz_speed(0,0,100,100)
                                time.sleep(4)
                                Drone.move_up(50)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)

                            if key == 'Down'and Drone.is_flying:
                                Drone.move_down(40)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)

                            if key == 'Left'and Drone.is_flying:
                                Drone.move_right(40)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)

                            if key == 'Right'and Drone.is_flying:
                                Drone.move_left(40)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)
                    
                            if key == 'FrontFlip' and Drone.is_flying:
                                Drone.flip_back()
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)
            
                            if key == 'TurnRight'and Drone.is_flying:
                                Drone.rotate_clockwise(20)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)
                
                            if key == 'TurnLeft'and Drone.is_flying:
                                Drone.rotate_counter_clockwise(20)
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)
                                
                            if key == 'Land'and Drone.is_flying:
                                Drone.land()
                                print(key)
                                cv2.putText(img, str(key), (10, 70), cv2.FONT_ITALIC, 3,
                                (255, 0, 255), 3)
                                cv2.imshow("Frame", img)



                    StartTime = time.time()
    cv2.imshow("Frame", img)

  
#information which is only important to look back to in case of errors (or to evaluate keras algorithm accuracy, or precision, didn't catch the difference)

    key =cv2.waitKey(1)
    if Drone.is_flying:
        Drone.send_rc_control(0,0,0,0)
    if key == ord("q"):
        Drone.land()
        Drone.streamoff()
        #cap.release()
        cv2.destroyAllWindows()
        break
    if key == ord("s"):
        Drone.takeoff() 