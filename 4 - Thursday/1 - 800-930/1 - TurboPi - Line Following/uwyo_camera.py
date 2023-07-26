#!/usr/bin/env python3
# encoding:utf-8
import cv2
import time
import threading
import numpy as np
from multiprocessing import Process
from HiWonder.CameraCalibration.CalibrationConfig import *

import uwyo_common as common

__name__ = 'uwyo_camera'

class Camera:
    def __init__(self, resolution=[640,480]):
        self.cap = None
        self.width = resolution[0]
        self.height = resolution[1]
        self.frame = None
        self.opened = False
        
        # Adjust camera calibration using stored values
        self.param_data = np.load(calibration_param_path + '.npz')
        dim = tuple(self.param_data['dim_array'])
        k = np.array(self.param_data['k_array'].tolist())
        d = np.array(self.param_data['d_array'].tolist())
        p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(k, d, dim ,None).copy()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), p, dim, cv2.CV_16SC2)
    
    def auto(self):
        try:
            self.cap = cv2.VideoCapture(-1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_SATURATION, 40)
            self.correction = correction
            self.opened = True
            self.query()
        except Exception as e:
            common.Print_Error('Camera -> start process', e)
        
    def close(self):
        try:
            self.opened = False
            time.sleep(0.2)
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.05)
            self.cap = None
        except Exception as e:
            common.Print_Error('Camera -> Close', e)
    
    def query(self):
        while True:
            try:
                if self.opened and self.cap.isOpened():
                    ret, frame_tmp = self.cap.read()
                    if ret:
                        self.frame = frame_tmp
                    else:
                        self.frame = None
                        self.cap.release()
                        cap = cv2.VideoCapture(-1)
                        ret, _ = cap.read()
                        if ret:
                            self.cap = cap
                elif self.opened:
                    self.cap.release()
                    cap = cv2.VideoCapture(-1)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap              
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                common.Print_Error('Camera -> Query', e)
                time.sleep(0.01)
                
    
    def start_thread(self, correction=False):
        try:
            self.cap = cv2.VideoCapture(-1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_SATURATION, 40)
            self.correction = correction
            self.opened = True
            self.th = threading.Thread(target=self.query, args=(), daemon=True)
            self.th.start()
        except Exception as e:
            common.Print_Error('Camera -> start thread', e)
            
    def stop(self):
        try:
            if self.th is not None:
                self.th.kill()
        except Exception as e:
            common.Print_Error('Camera -> stop', e)
