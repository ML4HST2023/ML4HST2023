"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/19/2023

Purpose:
Utilizes a camera and machine learning to identify lines in the environment
and attempts to follow them. Ideally the line should make a loop such that
the robot can follow it continuously.
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

# Outside Dependencies
import os
import cv2
import math
import time
import signal
import threading

import torch
from torchvision import transforms

# In House Dependencies
import uwyo_camera as camera
import uwyo_common as common
from uwyo_controller import controller


class line_follower():
    def __init__(self, threshold=0.95, subimages=2, mean=[0.5404], std=[0.1433], speed=40, turning=0.4, cam_x=2000, cam_y=1500, image_offset=64):
        # Set the classification threshold
        self.threshold = threshold
        
        # Create a controller object to interface with the HiWonder hat
        self.turbopi = controller()
        
        # Setup forward and turning speeds
        self.speed = speed
        self.turning = turning
        
        # Set the camera position
        self.cam_x = cam_x
        self.cam_y = cam_y
        self.turbopi.set_cam(x=self.cam_x, y=self.cam_y)
        
        # Manually set the 
        self.mean = mean
        self.std = std
        
        # Load the model and disable gradient prop
        self.model = torch.jit.load('line_follower.pt', map_location=torch.device('cpu'))
        self.model.eval()
        
        # Set the size of the image to give the ML model
        self.rows = 30
        self.cols = 40
        
        # Setup variables to track the size of the cropped images
        self.sub_images = subimages
        self.row_crop = None
        self.col_crop = None
        self.col_crop2 = None
        self.image_offset = image_offset
        
        # Setup variables to track the original image sizes 
        self.width = None
        self.height = None
        
        # Setup a variable to track the object detection state
        self.logits = []
        for i in range(subimages):
            self.logits.append(0)
            
        # Setup a transform to convert the input images into ML ready inputs
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((self.rows, self.cols), antialias=True),
                                             transforms.Grayscale(),
                                             transforms.Normalize(self.mean,self.std)])
           
        # Set the running flag and the stop command (ctrl + c)
        self.running = True
        signal.signal(signal.SIGINT, self.stop)
        
        
    """
        drive()
        A function to define what happens when each logit is true
    """
    def drive(self):
        if self.logits[0]:
            self.turbopi.turn_left(turning=self.turning)
        elif self.logits[1]:
            self.turbopi.turn_right(turning=self.turning)
        else:
            self.turbopi.forward(self.speed)


    """
        preprocess(frame)
        A function to perform image cropping and transforming so that the output is
        correctly formatted for the ML models input
        
        inputs:
         - frame (numpy array): The original frame to crop and transform
        outputs:
         - frames (numpy array): An array of cropped and transformed frames
    """
    def preprocess(self, frame):
        frames = torch.zeros([self.sub_images,1,self.rows,self.cols], dtype=torch.float32)
        for i in range(self.sub_images):
            temp = None
            j1 = i*(self.col_crop + self.col_crop2)
            j2 = j1 + self.col_crop
            #cv2.imshow(f'{j1} : {j2}', frame[self.height-self.row_crop:self.height,j1:j2,[2,1,0]])
            frames[i,:,:,:] = self.transform(frame[self.height-self.row_crop:self.height,j1:j2,[2,1,0]])
        return frames
    
    
    """
        run_model()
        A function to always grab the latest frame, preprocess it, and update the logits using the
        ML model
    """
    def run_model(self):
        while True:
            frame = self.cam.frame
            self.logits = self.model(self.preprocess(frame)) >= self.threshold if frame is not None else self.logits
            
    """
        run()
        A function to always act on the latest logits update to control the motors
        as well as update the battery voltage measurement
    """
    def run(self,):
        print()
        disp_index = 0
        while self.running:
            try:
                self.drive() 
                if disp_index > 120:
                    self.turbopi.display_battery()
                    disp_index = 0
                else:
                    disp_index += 1                   
            except Exception as e:
                common.Print_Error('Line Follower -> main loop', e)
                break
                    
    
    """
        start_ann()
        A function to setup and start the ML model thread
    """
    def start_ann(self,):
        self.ann = threading.Thread(target=self.run_model, args=(), daemon=True)
        self.ann.start()
        
        
    """
        start_cam()
        A function to setup the camera parameters including the cropping dimensions and original
        frame dimensions
    """
    def start_cam(self,):
        self.cam = camera.Camera()
        self.cam.start_thread()
        
        while self.row_crop is None:
            try:
                frame = self.cam.frame
                if frame is not None:
                    self.width = frame.shape[1]
                    self.height = frame.shape[0] - self.image_offset
                    self.row_crop = int(self.height/4) if self.sub_images <= 4 else int(self.height/self.sub_images)
                    self.col_crop = int(self.width/4) if self.sub_images <= 4 else int(self.width/self.sub_images)
                    #self.col_crop2 = int((self.width-(self.col_crop*self.sub_images))/(self.sub_images+1)) if self.sub_images < 4 else int(0)
                    self.col_crop2 = 2 * self.col_crop
            except Exception as e:
                common.Print_Error('Line Follower -> setup',e)
                
                    
    """
        stop(signum, var)
        A function to catch the stop key input and close everything properly
    """
    def stop(self, signum=None, var=None):
        try:
            self.running = False
            self.cam.close()
            self.turbopi.stop()
            self.turbopi.reset_cam()
            self.ann_thread.close()
            cv2.destroyAllWindows()
        except Exception as e:
            cv2.destroyAllWindows()
        
        
if __name__=='__main__':
    follower = line_follower(threshold=0.85,    # Classification threshold
                             subimages=2,       # number of sub images to crop
                             mean=[0.5404],     # z-normalization mean
                             std=[0.1433],      # z-normalization std
                             speed=40,          # Forward speed
                             turning=0.4,       # Turning speed
                             cam_x=2000,        # Camera x position
                             cam_y=1500,        # Camera y position
                             image_offset=64)   # Offset from bottom to crop images from
                             
    follower.start_cam()        # Start the camera thread
    follower.start_ann()        # Start the ML thread
    follower.run()              # Start the control thread
    follower.stop()             # When the program stops close everything properly
    
