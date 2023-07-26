#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/turbopi/TurboPi/')
import time
import signal
import HiwonderSDK.mecanum as mecanum

chassis = mecanum.MecanumChassis()

if __name__ == '__main__':
    # Power 50(0~100)， Direction 90(0~360)，Turning 0(-2~2)
    chassis.set_velocity(50,90,0) 
    time.sleep(1)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
    
    chassis.set_velocity(75,180,0)
    time.sleep(2)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
    
    chassis.set_velocity(50,270,0)
    time.sleep(1)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
    
    chassis.set_velocity(75,0,0)
    time.sleep(2)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
    
    chassis.set_velocity(50,0,-2)
    time.sleep(1)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
    
    chassis.set_velocity(50,0,2)
    time.sleep(1)
    
    chassis.set_velocity(0,0,0)
    time.sleep(1)
