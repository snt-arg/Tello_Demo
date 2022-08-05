#easy program to see Tello's stream

from djitellopy import tello
import cv2

Drone = tello.Tello()
Drone.connect()
Drone.streamon()

while True:
    img = Drone.get_frame_read().frame
    cv2.imshow("Output",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        Drone.streamoff()
        break