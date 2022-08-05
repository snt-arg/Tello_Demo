'''very simple test file which gives the start command and shares the screen of tell 
while projecting the percentage of the battery'''

from djitellopy import tello
import cv2
import time

Drone = tello.Tello()
Drone.connect()
Drone.streamon()

Commands = "s is takeoff\n s is "
while True:
    img = Drone.get_frame_read().frame
    Battery = Drone.get_battery()
    cv2.imshow(f'Tello: /n Battery={Battery}',img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        Drone.streamoff()
        break
    if key == ord("s"):
        Drone.takeoff()
    if key == ord("l"):
        Drone.land()
        ''' If you want to comment multiple lines of a document, you should use 3 of these marks on each end and start(best if it is in the same line as the order'''