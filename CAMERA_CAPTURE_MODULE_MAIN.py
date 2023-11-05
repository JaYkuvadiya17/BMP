# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:59:25 2022

@author: admin
"""

import pyrealsense2 as rs
import numpy as np
import cv2



class ComeraCapture_RealSense():
    def __init__(self, CAMERA_CONFIG_OBJ=None):
        ## Initialization of RealSense Camera
        #self.pipe = rs.pipeline()
        # Configure IR stream
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        
        # Start input streaming
        profile = pipeline.start(config)
        
        # Get emitter information
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]
        depth_sensor.set_option(rs.option.emitter_enabled, 0)
        self.pipeline = pipeline
        # Ignore first 1sec for camera warm-up
        for i in range(30):
            frames = self.pipeline.wait_for_frames()
        print('camera started...')

        
    def get_IR_FRAME_SET(self):
        frames = self.pipeline.wait_for_frames()
        infrared_frame_left = frames.get_infrared_frame(1)
        infrared_frame_right = frames.get_infrared_frame(2)
        L_img = np.asanyarray(infrared_frame_left.get_data())
        R_img = np.asanyarray(infrared_frame_right.get_data())
        if (L_img is not None) and (R_img is not None):
            ret = True
        return ret, L_img, R_img
    
    def get_COLOR_FRAME(self):
        frames = self.pipe.wait_for_frames()
        C_img = np.empty((0,0))
        ret = True
        return ret, C_img
    
    def get_IR_COLOR_FRAME_SET(self):
        frames = self.pipe.wait_for_frames()
        L_img = np.empty((0,0))
        R_img = np.empty((0,0))
        C_img = np.empty((0,0))
        ret = True
        return ret, L_img, R_img, C_img
    
class CameraCapture_USB():
    def __init__(self, CAMERA_CONFIG_OBJ=None):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,3040)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ramp_frames = 3 # initial images to discard to adjust the camera
        for i in range(ramp_frames):
            retval, frame = self.cap.read()
            
    def get_IR_FRAME_SET(self):
        L_img = np.empty((0,0))
        R_img = np.empty((0,0))
        ret, frame = self.cap.read()
        if ret:
            L_img = frame[:,:1520]
            R_img = frame[:,1520:]
        return ret, L_img, R_img
            