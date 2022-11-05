#!/usr/bin/env python
# The documents this code is built upon can be found on Git hub: https://github.com/kinivi/hand-gesture-recognition-mediapipe by Kinivi. I would recommend his program further since it is easy to modify and can be used in other ways too.
# This is an improved sketch for controlling a drone with handgestures. Now, it uses a handcombination (position1, position2) to identify commands.
# It is more childfriendly+ can have more commands now(around 500)

# -*- coding: utf-8 -*-
from cProfile import label
import csv
import copy
import argparse
import itertools
import time

import cv2 as cv
from cv2 import FONT_HERSHEY_COMPLEX
from cv2 import FILLED
from cv2 import LINE_AA
import numpy as np
import mediapipe as mp
from djitellopy import tello
from sklearn.metrics import classification_report


from utils import CvFpsCalc
from model import KeyPointClassifier
from google.protobuf.json_format import MessageToDict
Drone = tello.Tello()
Drone.connect()
Drone.streamon()
StaticImgResize = 300
offset = 100

# Getting+resizing respective images to display
TAKEOFF = cv.imread('Tello_Command_Images/TAKEOFF.png')
TAKEOFF_RESIZED = cv.resize(TAKEOFF, (StaticImgResize, StaticImgResize))

LAND = cv.imread('Tello_Command_Images/LAND.png')
LAND_RESIZED = cv.resize(LAND, (StaticImgResize, StaticImgResize))

LEFT = cv.imread('Tello_Command_Images/LEFT.png')
LEFT_RESIZED = cv.resize(LEFT, (StaticImgResize, StaticImgResize))

RIGHT = cv.imread('Tello_Command_Images/RIGHT.png')
RIGHT_RESIZED = cv.resize(RIGHT, (StaticImgResize, StaticImgResize))

BACKWARDS = cv.imread('Tello_Command_Images/BACKWARDS.jpg')
BACKWARDS_RESIZED = cv.resize(BACKWARDS, (StaticImgResize, StaticImgResize))

FORWARDS = cv.imread('Tello_Command_Images/FORWARDS.jpg')
FORWARDS_RESIZED = cv.resize(FORWARDS, (StaticImgResize, StaticImgResize))

UP = cv.imread('Tello_Command_Images/UP.png')
UP_RESIZED = cv.resize(UP, (StaticImgResize, StaticImgResize))

DOWN = cv.imread('Tello_Command_Images/DOWN.png')
DOWN_RESIZED = cv.resize(DOWN, (StaticImgResize, StaticImgResize))

YAWcw = cv.imread('Tello_Command_Images/YAWcw.jpeg')
YAWcw_RESIZED = cv.resize(YAWcw, (StaticImgResize, StaticImgResize))

YAWccw = cv.imread('Tello_Command_Images/YAWccw.jpeg')
YAWccw_RESIZED = cv.resize(YAWccw, (StaticImgResize, StaticImgResize))

FLIP = cv.imread('Tello_Command_Images/FLIP.jpeg')
FLIP_RESIZED = cv.resize(FLIP, (StaticImgResize, StaticImgResize))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)
    args = parser.parse_args()

    return args


def main():
    # Argument parsing
    use_brect = True
    #cap = cv.VideoCapture(0)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.9,
        min_tracking_confidence=0.8,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    mode, number = 0, 0
    Cnt = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(1)
        #success, image = cap.read()

        # Camera capture
        image = Drone.get_frame_read().frame

        if image is None:
            continue
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        HandSignList = []

        if results.multi_hand_landmarks is not None:

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                Cnt += 1
                # print(Cnt)
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                Right = False
                Left = False
                # making difference between left/right
                label = MessageToDict(handedness)[
                    'classification'][0]['label']
                if label == 'Right':
                    Right = True
                    current_right_handsign_id = keypoint_classifier(
                        pre_processed_landmark_list)
                    HandSignList.insert(1, str(current_right_handsign_id)+"R")
                if label == 'Left':
                    Left = True
                    current_left_handsign_id = keypoint_classifier(
                        pre_processed_landmark_list)
                    HandSignList.insert(0, str(current_left_handsign_id)+"L")
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    str(hand_sign_id),
                )

            if Cnt >= 50:
                # print(HandSignList)
                print(tuple(HandSignList))
                Cnt = 0
                cv.destroyWindow("Order")
                if not Drone.is_flying:
                    match tuple(HandSignList):
                        # 2_takeoff
                        case ('6L', '6R'):
                            print("takeoff")
                            print(hand_sign_id)
                            cv.imshow("Order", TAKEOFF_RESIZED)
                            # Drone.takeoff()
                            time.sleep(0.5)
                            # Drone.send_rc_control(0,0,100,0)
                            # Drone.send_rc_control(0,0,0,0)
                if Drone.is_flying:
                    match tuple(HandSignList):

                        # 1_land
                        case ('0L', '0R') | ('1R',):
                            print("land")
                            cv.imshow("Order", LAND_RESIZED)
                            # Drone.land()
                            time.sleep(0.5)
                            # Drone.send_rc_control(0,0,0,0)

                        # 3_right
                        case('0L', '1R'):
                            print("Right")
                            cv.imshow("Order", RIGHT_RESIZED)
                            #Drone.send_rc_control(100, 0, 0, 0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 4_left
                        case('0L', '2R'):
                            print("Left")
                            cv.imshow("Order", LEFT_RESIZED)
                            #Drone.send_rc_control(-100, 0, 0, 0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 5_up
                        case ('1L', '6R'):
                            print("Up")
                            cv.imshow("Order", UP_RESIZED)
                            #Drone.send_rc_control(0, 0, 100, 0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 6_down
                        case ('1L', '7R'):
                            print("Down")
                            cv.imshow("Order", DOWN_RESIZED)
                            #Drone.send_rc_control(0, 0, -100, 0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 7_forwards
                        case ('0L', '4R'):
                            print("forwards")
                            cv.imshow("Order", FORWARDS_RESIZED)
                            # Drone.send_rc_control(0,100,0,0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 8_backwards
                        case ('0L', '5R'):
                            print("backwards")
                            cv.imshow("Order", BACKWARDS_RESIZED)
                            # Drone.send_rc_control(0,-100,0,0)
                            time.sleep(0.5)
                            #Drone.send_rc_control(0, 0, 0, 0)

                        # 9_flip(a)
                        case ('6L', '2R'):
                            print("flip")
                            cv.imshow("Order", FLIP_RESIZED)
                            # Drone.flip_back()
                            time.sleep(0.5)

                        # 10_flip(b)
                        case ('6L', '4R'):
                            print("flip")
                            cv.imshow("Order", FLIP_RESIZED)
                            # Drone.flip_back()
                            time.sleep(0.5)

                        # 11 FLIP MAYHEM
                        case ('8L', '2R'):
                            print("flip")
                            cv.imshow("Order", FLIP_RESIZED)
                            '''Drone.flip_forward()
                            time.sleep(0.2)
                            Drone.flip_right()
                            time.sleep(0.2)
                            Drone.flip_back()
                            time.sleep(0.2)
                            Drone.flip_left()
                            time.sleep(0.2)
                            Drone.flip_forward()'''

                # Drawing part
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow('Hand Gesture Recognition', debug_image)

        Ukey = cv.waitKey(0)
        # drone stabiliser
        if Drone.is_flying:
            Drone.send_rc_control(0, 0, 0, 0)

        # keyboard control
        if Ukey == ord("q"):
            Drone.send_command_without_return("land")
            Drone.streamoff()
            cv.destroyAllWindows()
            break
        if key == ord("s"):
            Drone.takeoff()
            Drone.send_rc_control(0, 0, 100, 0)
            time.sleep(0.5)
            Drone.send_rc_control(0, 0, 0, 0)

        if key == ord("U"):
            Drone.send_rc_control(0, 0, -100, 0)
            time.sleep(0.5)
            Drone.send_rc_control(0, 0, 0, 0)
            print(key)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 6)

        # Index finger

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 4)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 4)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 4)

        # Little finger

        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 4)

        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 4)

        # Palm

        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 5)

        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 5)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 1:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 2:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 3:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 4:
            cv.circle(image, (landmark[0], landmark[1]), 8, (204, 0, 0),
                      -1)
        if index == 5:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 6:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 7:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 8:
            cv.circle(image, (landmark[0], landmark[1]), 8, (204, 0, 0),
                      -1)
        if index == 9:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 10:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 11:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 12:
            cv.circle(image, (landmark[0], landmark[1]), 8, (204, 0, 0),
                      -1)
        if index == 13:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 14:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 15:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 16:
            cv.circle(image, (landmark[0], landmark[1]), 8, (204, 0, 0),
                      -1)
        if index == 17:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 18:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 19:
            cv.circle(image, (landmark[0], landmark[1]), 5, (204, 0, 0),
                      -1)
        if index == 20:
            cv.circle(image, (landmark[0], landmark[1]), 8, (204, 0, 0),
                      -1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255, 20, 147), 4)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ': ' + hand_sign_text
    cv.putText(image, str(info_text), (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, info_text, (10, 350),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (127, 0, 255), 2,
                   cv.LINE_AA)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 390), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 255, 0), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
