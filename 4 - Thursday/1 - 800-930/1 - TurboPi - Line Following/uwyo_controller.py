# -*- coding: utf-8 -*-
"""
Project: ML4HST
Advisor: Dr. Suresh Muknahallipatna
Author: Josh Blaney

Last Update: 06/19/2023

Purpose:
Used to provide common functionality such as wheel control and servo control
for the HiWonder TurboPi. This file is a wrapper of the board tha mecanum
files provided by HiWonder.
 
Notes:
For more information about the project contact Dr. Suresh Muknahallipatna.
"""

import sys
import time
sys.path.append('/home/turbopi/TruboPi/')
import HiWonder.HiwonderSDK.mecanum as mecanum
import HiWonder.HiwonderSDK.Board as board

__name__ = 'uwyo_controller'

class controller():
    def __init__(self):
        self.robot = mecanum.MecanumChassis()
    
    def display_battery(self,):
        voltage = board.getBattery() / 1000
        voltage = voltage / 10 if voltage > 10 else voltage
        print('\r', f'Battery voltage {voltage:<.2f}V', end="\r")
        
    def drive(self, speed=25, direction=0, turning=0):
        self.robot.set_velocity(speed, direction, turning)
    
    def drive_raw(self, v1, v2, v3, v4):
        board.setMotor(1, v1)
        board.setMotor(2, v2)
        board.setMotor(3, v3)
        board.setMotor(4, v4)
        
    def drift_left(self, speed=25, direction=180):
        self.robot.set_velocity(speed, direction, 0)
        
    def drift_right(self, speed=25, direction=0):
        self.robot.set_velocity(speed, direction, 0)
        
    def forward(self, speed=25):
        self.robot.set_velocity(speed, 90, 0)
        
    def reset_cam(self):
        board.setPWMServoPulse(1, 1500, 1000)
        board.setPWMServoPulse(2, 1500, 1000)
        board.setBuzzer(0)
        board.setBuzzer(1)
        time.sleep(0.05)
        board.setBuzzer(0)
        
    def reverse(self, speed=25):
        self.robot.set_velocity(speed, 270, 0)
    
    def set_cam(self, x=1500, y=1500):
        board.setPWMServoPulse(1, x, 1000)
        board.setPWMServoPulse(2, y, 1000)
        
    def stop(self):
        self.robot.set_velocity(0,0,0)
        
    def turn_left(self, turning=-1):
        turning = turning if turning <= 0 else -turning
        self.robot.set_velocity(0, 0, turning)
        
    def turn_right(self, turning=1):
        turning = turning if turning >=0 else -turning
        self.robot.set_velocity(0, 0, turning)
        
