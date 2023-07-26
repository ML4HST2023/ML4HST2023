'''
    A program that tracks and runs the TurboPi with user's face.
    When the user shows index finger gesture, the TurboPi stops running.
    
'''
# Import system path
import sys
sys.path.append('/home/turbopi/TurboPi/')
# Impport cv2 for computer vision operations
import cv2
# Import time for delaying the program
import time
# Import camera
import Camera
# Import threading for running vehicle and AI functions in parallel
import threading
# Import numpy for numeric computation
import numpy as np
# Import mediapipe for face tracking and gesture recognition
import mediapipe as mp
# Import HiWonder libraries
import HiwonderSDK.PID as PID
import HiwonderSDK.Board as Board
import HiwonderSDK.mecanum as mecanum
# Import hand angle and gesture recognition functions from GestureRecognition library
from GestureRecognition import hand_angle, gesture
# Import warnings to remove unnecessary warnings
import warnings

# Initalize the car
car = mecanum.MecanumChassis()
# Initialize face detection object
Face = mp.solutions.face_detection
# Create a 70% confidence face tracking function
faceDetection = Face.FaceDetection(min_detection_confidence=0.7)

# Initialize gesture recognition object
mp_hands = mp.solutions.hands
# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# Create a hand tracking module with 60% confidence
hands = mp_hands.Hands(model_complexity=0,min_detection_confidence=0.6,min_tracking_confidence=0.6)

# Set the servo parameters
servo1 = 1500
servo2 = 1500
servo_x = servo2
servo_y = servo1

# Create flag for car operations
stop_st = False
results_lock = False
gesture_num = None
# Set the size of the camera window
size = (640, 480)
# Set the car running flag to False
__isRunning = False
# Set the default center and area of camera
center_x, center_y, area = -1, -1, 0

# Initialize the car in x direction
car_x_pid = PID.PID(P=0.150, I=0.001, D=0.0001)
# Initialize the car in y direction
car_y_pid = PID.PID(P=0.002, I=0.001, D=0.0001)
# Initialize the servo in x direction
servo_x_pid = PID.PID(P=0.05, I=0.0001, D=0.0005)
# Initialize the servo in y direction
servo_y_pid = PID.PID(P=0.05, I=0.0001, D=0.0005)

# Function to set servo in default position
def initMove():
    Board.setPWMServoPulse(1, servo1, 1000)
    Board.setPWMServoPulse(2, servo2, 1000)

# Function to stop the car
def car_stop():
    car.set_velocity(0,90,0)

# Function to reset the car's running parameters
def reset():
    # Set variables to global 
    global servo1, servo2
    global servo_x, servo_y
    global center_x, center_y, area
    
    # Set the servo parameters to default
    servo1 = 1185
    servo2 = 1500
    servo_x = servo2
    servo_y = servo1
    
    # Clear the x and y direction parameters of car and servo
    car_x_pid.clear()
    car_y_pid.clear()
    servo_x_pid.clear()
    servo_y_pid.clear()
    center_x, center_y, area = -1, -1, 0

# Function to start the program
def start():
    global __isRunning
    reset()
    __isRunning = True
    print("FaceTracking Start")

# Set the flag to False
car_en = False

# Function to move the car according to face tracked
def move():
    # Set the global variables
    global __isRunning,car_en, stop_st
    global servo_x, servo_y
    global center_x, center_y, area
    global gesture_num, results_lock
    
    # Set the image width and height
    img_w, img_h = size[0], size[1]
    
    # Loop until car stops running
    while True:
        # If the car is running
        if __isRunning:
            # If the camera is not centered properly
            if center_x != -1 and center_y != -1:
                # If the camera is out of face frame
                if abs(center_x - img_w/2.0) < 15:
                    # Half the center x-coordinates
                    center_x = img_w/2.0
                # Set the x-servo point
                servo_x_pid.SetPoint = img_w/2.0
                # Update the x-servo motor
                servo_x_pid.update(center_x)
                # Keep on updating the x-servo motor depending on face coordinates
                servo_x += int(servo_x_pid.output)
                
                # Set the servo x according to different conditions
                servo_x = 800 if servo_x < 800 else servo_x
                servo_x = 2200 if servo_x > 2200 else servo_x
                
                # If the camera is out of face frame
                if abs(center_y - img_h/2.0) < 10:
                    # Half the center y-coordinates
                    center_y = img_h/2.0
                # Set the y-servo point
                servo_y_pid.SetPoint = img_h/2.0
                # Update the y-servo motor
                servo_y_pid.update(center_y)
                # Keep on updating the y-servo motor depending on face coordinates
                servo_y -= int(servo_y_pid.output)
                # Set the servo y according to different conditions
                servo_y = 1000 if servo_y < 1000 else servo_y
                servo_y = 1900 if servo_y > 1900 else servo_y
                
                # Change the x-servo position
                Board.setPWMServoPulse(1, servo_y, 20)
                # Change the y-servo position
                Board.setPWMServoPulse(2, servo_x, 20)
                
                # If the area becomes out of designated area
                if abs(area - 30000) < 2000 or servo_y < 1100:
                    # Set the area to default value
                    car_y_pid.SetPoint = area
                else:
                    # Else, set the y value to 30000 (Highest)
                    car_y_pid.SetPoint = 30000
                # Update the car
                car_y_pid.update(area)
                # Set the y area output to dy
                dy = car_y_pid.output
                # If the dy is less than 20, set dy to zero
                dy = 0 if abs(dy) < 20 else dy
                
                # If the value is less than 15
                if abs(servo_x - servo2) < 15:
                    # Set the point to servo_x
                    car_x_pid.SetPoint = servo_x
                else:
                    # Else, set the value to default
                    car_x_pid.SetPoint = servo2
                # Update the servo parameters
                car_x_pid.update(servo_x)
                # Set the x area output to dx
                dx = car_x_pid.output
                # If the dx is less than 20, set dx to zero
                dx = 0 if abs(dx) < 20 else dx
                # Run the car according to face coordinates
                car.translation(dx, dy)
                # Set the car engine to true
                car_en = True
            else:
                # If the car engine is found true without faces detected
                if car_en:
                    # Set the car top stop
                    car_stop()
                    # Set car engine to false
                    car_en = False
                # Delay the program
                time.sleep(0.01)
            # If the hand gesture has been detected
            if results_lock and gesture_num:
                # If the index finger is detected
                if gesture_num == 1:
                    # Set the car to stop
                    car.set_velocity(0,90,0)
                    # Stop the car from running
                    __isRunning = False
                # Disable locking the gesture results
                results_lock = False
                
            else:
                # If the car has been stopped
                if stop_st:
                    # Initialize the car 
                    initMove()
                    # Stop the car again
                    car_stop()
                    # Set the stop to false
                    stop_st = False
                    # Delay the program
                    time.sleep(0.5)
                # Delay the program
                time.sleep(0.01)
        
        else:
            # If car is still running
            if car_en:
                # Stop the car
                car_stop()
                # Set the engine to false
                car_en = False
            # Delay the program
            time.sleep(0.01)
# Create a thread to run the function 'move' in parallel with main program 
th = threading.Thread(target=move)
# Set the Daemon to true
th.setDaemon(True)
# Run the program
th.start()
# Create an array to store the results
results_list = []

# Function to perform computer vision operations: Face tracking and Gesture Recognition
def run(img):
    # Set the global variables
    global __isRunning, area
    global gesture_num
    global center_x, center_y
    global center_x, center_y, area
    global results_lock
    global results_list
    
    # If the car is not running, return the image on camera window
    if not __isRunning:   
        return img
    # Create a copy of image
    img_copy = img.copy()
    # Extract the height and width of the image
    img_h, img_w = img.shape[:2]
    # Set a vairable to store the gesture number
    gesture_num = 0
    # Convert the image to RGB
    imgRGB = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    # Perform face detection
    results = faceDetection.process(imgRGB)
    # Perform hand detection
    results_hand = hands.process(imgRGB)
    # If face has been detected
    if results.detections:
        # Loop through face coordinates
        for index, detection in enumerate(results.detections):
            # Create a bounding box over face
            bboxC = detection.location_data.relative_bounding_box
            # Get the bounding box variables
            bbox = (int(bboxC.xmin * img_w), int(bboxC.ymin * img_h),  
                   int(bboxC.width * img_w), int(bboxC.height * img_h))
            # Draw the bounding box over face
            mp_drawing.draw_detection(img,detection)
            # Set the bounding box to x, y, width and height
            x, y, w, h = bbox
            # Get the central x-coorindates
            center_x =  int(x + (w/2))
            # Get the central y-coorindates
            center_y =  int(y + (h/2))
            # Find the area
            area = int(w * h)
    else:
        # Set the default central values and area
        center_x, center_y, area = -1, -1, 0
    # If hand is detected
    if results_hand.multi_hand_landmarks:
        # Loop through hand coordinates
        for hand_landmarks in results_hand.multi_hand_landmarks:
            # Draw the landmarks over detected hand
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get the cooridnates of the hand
            hand_local = [(landmark.x * img_w, landmark.y * img_h) for landmark in hand_landmarks.landmark]
            # If coordinates have been detected
            if hand_local:
                # Get the current angle of the hand
                angle_list = hand_angle(hand_local)
                # Get the gesture using the angle
                gesture_results = gesture(angle_list)
                # Display the numbers of fingers raised
                cv2.putText(img, str(gesture_results), (20, 50), 0, 2, (255, 100, 0), 3)
                # If gestures have been recognized
                if gesture_results:
                    # Add the gesture result on the list
                    results_list.append(gesture_results)
                    # If five fingers are detected
                    if len(results_list) == 5:
                        # find the gesture number
                        gesture_num = np.mean(np.array(results_list))
                        # lock the results
                        results_lock = True
                        # empty the results list
                        results_list = []
    # Return the image for displaying
    return img
# Main function
if __name__ == '__main__':
    # Start the car and set the servos 
    start()
    # Initialize the camera
    camera = Camera.Camera()
    # Open the camera
    camera.camera_open(correction=True)
    # Filter unnecessary warnings
    warnings.filterwarnings("ignore")
    # Loop until the car is running
    while __isRunning:
        # Get the frames from camera
        img = camera.frame
        # If the frame has been detected
        if img is not None:
            # Copy the frame
            frame = img.copy()
            # Run the copied frame for computer vision operations
            Frame = run(frame)
            # Resize the copied frame
            frame_resize = cv2.resize(Frame, (640, 480))
            # Display the window with the video
            cv2.imshow('frame', frame_resize)
            # Wait until user hits button q
            key = cv2.waitKey(1)
            # If user hits button q
            if key == 27:
                # Set the running to False
                __isRunning = False
                # Stop the program
                car_stop()
                # Break the loop
                break
        else:
            # Delay the program
            time.sleep(0.01)
    # Close the camera           
    camera.camera_close()
    # Stop the car
    car_stop()
    # Destroy the camera window
    cv2.destroyAllWindows()