'''
    A GUI program that enables user to manually orient the camera using X and Y servo motors.
'''
# Import the system to create a path
import sys
sys.path.append('/home/turbopi/TurboPi/')
# Import threading for running camera and servo in parallel
import threading
# Import cv2 for CV operations
import cv2q
# Import board to control servo motors
import HiwonderSDK.Board as Board

# Initialize the current servo y position
Board.setPWMServoPulse(1, 1100, 1000)
# Initialize the current servo x position
Board.setPWMServoPulse(2, 1100, 1000)

# Function to orient servo x positon
def servo_x_position(value):
    # Convert servo value from range (0-180) to (1100-2000)
    servo_value = int((value / 180) * (2000 - 1100) + 1100)
    # Update the servo x position based on the slider value
    Board.setPWMServoPulse(2, servo_value, 100)
    # Display the current servo position
    print(f"Servo X position: {value}")

# Function to orient servo y position
def servo_y_position(value):
    # Convert servo value from range (0-180) to (1100-2000)
    servo_value = int((value / 180) * (2000 - 1100) + 1100)
    # Update the servo y position based on the slider value
    Board.setPWMServoPulse(1, servo_value, 100)
    # Display the current servo position
    print(f"Servo Y position: {value}")

# Function to run the camera thread
def camera_thread():
    # Create a camera object to get frames
    cap = cv2.VideoCapture(-1)  
    # Create a window to display the camera feed
    cv2.namedWindow("Camera Control")
    # Create sliders for controlling x-servo
    cv2.createTrackbar("Servo X", "Camera Control", 0, 180, servo_x_position)
    # Create sliders for controlling y-servo
    cv2.createTrackbar("Servo Y", "Camera Control", 0, 180, servo_y_position)
    # Read each camera frames
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        # If frame has been detected
        if ret:
            # Display the text
            text = "Press q to exit"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Display the frame in the window
            cv2.imshow("Camera Control", frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release the camera
    cap.release()
    # Destroy the camera window
    cv2.destroyAllWindows()
    
# Main function
if __name__ == '__main__':
    # Create a separate thread for the camera
    camera_thread = threading.Thread(target=camera_thread)
    # Start the thread
    camera_thread.start()
    # Wait for the camera thread to exit
    camera_thread.join()