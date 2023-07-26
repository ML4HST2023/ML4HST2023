import os
import cv2
import time

import HiWonder.HiwonderSDK.Board as board

from uwyo_controller import controller
import uwyo_camera as camera
import uwyo_common as common

# Select the location to save the data to
directory = 'datasets/line_following/'
common.Validate_Dir(directory)

# Set the camera position
servox = 2000 # [500 - 2000] -> [up - down] -> 1500 is straight ahead
servoy = 1500 # [500 - 2500] -> [right - left] -> 1500 is straight ahead

# Create a controller to reference the camera servos
turbopi = controller()
turbopi.set_cam(x=servox, y=servoy)

# Start the camera thread
cam = camera.Camera()
cam.start_thread()

# Track the number of frames to save and the number saved
frames_saved = 0
frames_to_save = 100

while frames_saved < frames_to_save:
    frame = cam.frame # Store the frame locally
    if frame is not None:
        cv2.imshow('frame',frame) # Display the frame for visual validation
        # Allow exit with escape key
        key = cv2.waitKey(1) 
        if key == 27:
            break
        else:
            # Save the frame and delay the camera to the desired save speed
            cv2.imwrite(f'{directory}{frames_saved}.jpg', frame)
            frames_saved += 1
            time.sleep(0.1)
    else:
        time.sleep(1)

# Properly close camera
cam.close()
cv2.destroyAllWindows()

# Properly shutdown robot and reset servos
turbopi.stop()
turbopi.reset_cam()