#Tello commanding with hand gestures and keyboard

from google.protobuf.json_format import MessageToDict
from keras.models import load_model
from djitellopy import tello
from cv2 import FONT_ITALIC
import mediapipe as mp
from time import time
import numpy as np
import time
import cv2


'''Drone = tello.Tello()
Drone.connect()
Drone.streamon()'''

Cnt = 0
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9,min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils


model = load_model('11Handsigns(50Ep).h5') 
pTime = 0
cTime = 0
offset = 40
StaticImgResize = 250
Orderlist = ["T"]

TAKEOFF = cv2.imread('Tello_Command_Images/TAKEOFF.png')
TAKEOFF_RESIZED = cv2.resize(TAKEOFF,(StaticImgResize,StaticImgResize))

LAND = cv2.imread('Tello_Command_Images/LAND.png')
LAND_RESIZED = cv2.resize(LAND,(StaticImgResize,StaticImgResize))

LEFT = cv2.imread('Tello_Command_Images/LEFT.png')
LEFT_RESIZED = cv2.resize(LEFT,(StaticImgResize,StaticImgResize))

RIGHT = cv2.imread('Tello_Command_Images/RIGHT.png')
RIGHT_RESIZED = cv2.resize(RIGHT,(StaticImgResize,StaticImgResize))

BACKWARDS = cv2.imread('Tello_Command_Images/BACKWARDS.jpg')
BACKWARDS_RESIZED = cv2.resize(BACKWARDS,(StaticImgResize,StaticImgResize))

FORWARDS = cv2.imread('Tello_Command_Images/FORWARDS.jpg')
FORWARDS_RESIZED = cv2.resize(FORWARDS,(StaticImgResize,StaticImgResize))

UP = cv2.imread('Tello_Command_Images/UP.png')
UP_RESIZED = cv2.resize(UP,(StaticImgResize,StaticImgResize))

DOWN = cv2.imread('Tello_Command_Images/DOWN.png')
DOWN_RESIZED = cv2.resize(DOWN,(StaticImgResize,StaticImgResize))

YAWcw = cv2.imread('Tello_Command_Images/YAWcw.jpeg')
YAWcw_RESIZED = cv2.resize(YAWcw,(StaticImgResize,StaticImgResize))

YAWccw = cv2.imread('Tello_Command_Images/YAWccw.jpeg')
YAWccw_RESIZED = cv2.resize(YAWccw,(StaticImgResize,StaticImgResize))

FLIP = cv2.imread('Tello_Command_Images/FLIP.jpeg')
FLIP_RESIZED = cv2.resize(FLIP,(StaticImgResize,StaticImgResize))

analysisframe = ''
Tello_commands = ['Backwards', 'Down', 'Forwards', 'FrontFlip','Land','Left','Right','TakeOff','TurnLeft','TurnRight','Up']


while True:
    success, img = cap.read()
    #img = Drone.get_frame_read().frame
    img = cv2.flip(img, 1) 
    h,w,c =img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    hand_landmarks = results.multi_hand_landmarks
    analysisframe = img.copy()

    '''Battery = Drone.get_battery()
    Height = Drone.get_height()
    FLightTime = Drone.get_flight_time()
    Temperature = Drone.get_highest_temperature()
    cv2.putText(img,"Temperature :"+str(Temperature),(10,150),FONT_ITALIC,1,(255,255,255),2,4)
    cv2.putText(img,"Battery: "+str(Battery),(10,200),FONT_ITALIC,1,(255,255,255),2,4)
    cv2.putText(img,"Height: "+str(Height),(10,250),FONT_ITALIC,1,(255,255,255),2,4)
    cv2.putText(img,"FLightTime: "+str(FLightTime),(10,300),FONT_ITALIC,1,(255,255,255),2,4)'''

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "fps: "+ str(int(fps)), (10, 50), cv2.FONT_ITALIC, 1,
                (255, 0, 255), 4)
#hand + box            
    if hand_landmarks:
        for i in results.multi_handedness:
            label = MessageToDict(i)[
                'classification'][0]['label']
            if label == 'Right':
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
                cv2.rectangle(img, (x_min - offset, y_min - offset), (x_max + offset, y_max + offset), (255, 255, 255), 2)
                mpDraw.draw_landmarks(img, handLMs, mpHands.HAND_CONNECTIONS)
                cv2.circle(img,(x_max,y_max),10,(255,0,0),10,0)
                cv2.circle(img,(x_min,y_min),10,(0,0,255),10,0)
                hBox = y_max-y_min
                wBox = x_max-x_min

                cv2.putText(img,"HeightOfHand: "+str(hBox),(10,350),FONT_ITALIC,1,(255,255,255),2,4)
                cv2.putText(img,"WidthOfHand: "+str(wBox),(10,400),FONT_ITALIC,1,(255,255,255),2,4)

             
               
                


#frames per second on main image
    
                
                      
                Cnt += 1
#resizing shape for keras
                if Cnt >= 50 and len(analysisframe) != 0:
                    cv2.destroyWindow("Order")
                    #print(analysisframe.shape)
                    #cv2.imshow("analysis frame", analysisframe)
                    
                    analysisframe = analysisframe[y_min - offset :y_max+ offset, x_min - offset:x_max + offset]
                    
                    if analysisframe.shape[0] == 0 or analysisframe.shape[1] == 0:
                        continue

                    cv2.imshow("analysisframe",analysisframe)

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
                            print('Confidence 1: ', round(100*value),1)

                            '''if key == 'TakeOff' and not Drone.is_flying:
                                Drone.takeoff()
                                Drone.send_rc_control(0,0,100,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                cv2.imshow("Order",TAKEOFF_RESIZED)
                                cv2.moveWindow("Order",10,450)
                                print(key)

                            if key == 'Land'and Drone.is_flying:
                                Drone.land()
                                print(key)
                                cv2.imshow("Order",LAND_RESIZED)

                            if key == 'Backwards'and Drone.is_flying:
                                Drone.send_rc_control(0,-100,0,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                            
                                cv2.imshow("Order", BACKWARDS_RESIZED)
                                
                            if key == 'Forwards'and Drone.is_flying:
                                Drone.send_rc_control(0,100,0,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
       
                                cv2.imshow("Order", FORWARDS_RESIZED)

                            if key == 'Up'and Drone.is_flying:
                                Drone.send_rc_control(0,0,100,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                
                                cv2.imshow("Order",UP_RESIZED )

                            if key == 'Down'and Drone.is_flying:
                                Drone.send_rc_control(0,0,-100,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                           
                                cv2.imshow("Order", DOWN_RESIZED)

                            if key == 'Left'and Drone.is_flying:
                                Drone.send_rc_control(-100,0,0,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                              
                                cv2.imshow("Order", LEFT_RESIZED)

                            if key == 'Right'and Drone.is_flying:
                                Drone.send_rc_control(100,0,0,0)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                               
                                cv2.imshow("Order", RIGHT_RESIZED)
                    
                            if key == 'FrontFlip' and Drone.is_flying:
                                Drone.flip_back()
                                print(key)
                                
                                cv2.imshow("Order",FLIP_RESIZED)
            
                            if key == 'TurnRight'and Drone.is_flying:
                                Drone.send_rc_control(0,0,0,-100)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                                
                                cv2.imshow("Order", YAWcw_RESIZED)
                
                            if key == 'TurnLeft'and Drone.is_flying:
                                Drone.send_rc_control(0,0,0,100)
                                time.sleep(0.5)
                                Drone.send_rc_control(0,0,0,0)
                                print(key)
                                
                                cv2.imshow("Order", YAWccw_RESIZED)
                            cv2.moveWindow("Order",10,450)'''
                                        

                    Cnt = 0  

    else: 

        cv2.destroyWindow("Order")
        key = cv2.waitKey(1)
        '''if key == ord("s") and Drone.is_flying:
            Drone.send_rc_control(0,-100,0,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
        
            cv2.imshow("Order", BACKWARDS_RESIZED)
            
        if key == ord("w") and Drone.is_flying:
            Drone.send_rc_control(0,100,0,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)

            cv2.imshow("Order", FORWARDS_RESIZED)

        if key == ord("t") and Drone.is_flying:
            Drone.send_rc_control(0,0,100,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)

            cv2.imshow("Order",UP_RESIZED )

        if key == ord("g") and Drone.is_flying:
            Drone.send_rc_control(0,0,-100,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
        
            cv2.imshow("Order", DOWN_RESIZED)

        if key == ord("d") and Drone.is_flying:
            Drone.send_rc_control(-100,0,0,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
            
            cv2.imshow("Order", LEFT_RESIZED)

        if key == ord("a") and Drone.is_flying:
            Drone.send_rc_control(100,0,0,0)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
            
            cv2.imshow("Order", RIGHT_RESIZED)

        if key == ord("y") and Drone.is_flying:
            Drone.flip_back()
            print(key)
            
            cv2.imshow("Order",FLIP_RESIZED)

        if key == ord("q")and Drone.is_flying:
            Drone.send_rc_control(0,0,0,-100)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
            
            cv2.imshow("Order", YAWcw_RESIZED)

        if key == ord("e") and Drone.is_flying:
            Drone.send_rc_control(0,0,0,100)
            time.sleep(0.5)
            Drone.send_rc_control(0,0,0,0)
            print(key)
            
            cv2.imshow("Order", YAWccw_RESIZED)'''
        cv2.moveWindow("Order",10,450) 
        cv2.imshow("Frame", img)

    
   
'''          
    #Drone stabaliser
   key =cv2.waitKey(1)
   if Drone.is_flying:
        Drone.send_rc_control(0,0,0,0)

    if key == 27:
        Drone.send_command_without_return("land")
        Drone.streamoff()
        cv2.destroyAllWindows()
        break
    
    if key == ord("r"):
        cv2.imshow("Order",TAKEOFF_RESIZED)
        Drone.takeoff()
        Drone.send_rc_control(0,0,100,0)
        time.sleep(0.5)
        Drone.send_rc_control(0,0,0,0)
'''