# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:18:59 2022

@author: JEET
"""
import cv2
import numpy as np
import pickle
import itertools
from scipy.spatial.distance import pdist, cdist
import pyrealsense2 as rs
# import cv2.aruco as aruco
from camera_calibrate import StereoCalibration
# from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense, CameraCapture_USB
import time


'''
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
'''
class ArUco():
    def __init__(self):
        self.markersize = 4
        self.totalMarkers = 50      
        self.tool_vector =  None
        
    def set_tool_vector(self,tool_list):
        self.tool_vector = tool_list
        
    def get_corner_world_points(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img)
        wp_list = [] 
        
        #print('------right corner points-------')
        #print(matched_right_corner_points)
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                wp = StereoObj.get_world_points(matched_left_corner_points[i], matched_right_corner_points[i])
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
    
    def get_center_world_points(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img)
        wp_list = [] 
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                left_cnt_pnts = np.expand_dims(np.mean(matched_left_corner_points[i], axis=0), axis=0)
                right_cnt_pnts = np.expand_dims(np.mean(matched_right_corner_points[i], axis=0), axis=0)
                wp = StereoObj.get_world_points(left_cnt_pnts, right_cnt_pnts)
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
    
    def get_corner_world_points_v2(self,StereoObj, left_img, right_img):
        tool_ret, matched_left_corner_points, matched_right_corner_points = self.get_matched_corner_points_pairs(left_img, right_img)
        wp_list = [] 
        for i in range(len(tool_ret)):
            if tool_ret[i]==True:
                wp = StereoObj.get_world_points(matched_left_corner_points[i], matched_right_corner_points[i])
                #print('wp',wp)
                wp_list.append(wp)
            else:
                wp_list.append(None)       
        return tool_ret, wp_list
                
    def findArucoMarkers(self, img, draw=True):
        key = getattr(cv2.aruco, f'DICT_{self.markersize}X{self.markersize}_{self.totalMarkers}')
        cv2.aruco_dict = cv2.aruco.Dictionary_get(key)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arucoParam = cv2.aruco.DetectorParameters_create()
        arucoParam .cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        bboxes, ids, rejected = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=arucoParam)
    
        # bboxes_npArray = np.squeeze(np.array(bboxes))
        bboxes_npArray = np.array(bboxes)
        #print(bboxes_npArray.shape,ids)
    
        if draw:
            cv2.aruco.drawDetectedMarkers(img, bboxes)
        return bboxes_npArray, ids
    
    def get_matched_corner_points_pairs(self, left_img, right_img):
        
        num_tools = len(self.tool_vector)
        
        matched_left_corner_points = np.empty((len(self.tool_vector),4,2))
        matched_right_corner_points = np.empty((len(self.tool_vector),4,2))
        tool_ret = [False]*num_tools
        
        corner_left, ids_left  =   self.findArucoMarkers(left_img)
        corner_right, ids_right  = self.findArucoMarkers(right_img)
        
        
        for i in range(num_tools):
            idx_l = np.where(ids_left==self.tool_vector[i])[0]
            idx_r = np.where(ids_right==self.tool_vector[i])[0]      
            #print(idx_l, idx_r)
            if (idx_l.size == 1) and (idx_r.size == 1):
                tool_ret[i] = True
                matched_left_corner_points[i,:,:] = corner_left[idx_l[0],:,:]
                matched_right_corner_points[i,:,:] = corner_right[idx_r[0],:,:]
            else:
                # print('error: tool id-', self.tool_vector[i], ' is not detected...')
                pass

        
        return tool_ret, matched_left_corner_points, matched_right_corner_points
                    
                
            
'''
    def getWPDistance(corner_points_left, ids_left, corner_points_right, ids_right):
        if ids_left is None:
            return
        if ids_right is None:
            return
        for pos,ele in enumerate(tool_vector):
            left_query = np.where(ids_left == ele)
            found_left = left_query[0]
            right_query = np.where(ids_right == ele)
            found_right = right_query[0]
            # print(ids_left,"\n", ids_right)
            # print(corner_points_left, "\n", corner_points_right)
            if (found_left.size >= len(tool_vector)) and (found_right.size >= len(tool_vector)):
                left_corners = corner_points_left[found_left[0], :, :]
                left_corners = np.squeeze(left_corners)
    
                right_corners = corner_points_right[found_right[0], :, :]
                right_corners = np.squeeze(right_corners)
    
                wp = StereoObj.get_world_points(left_corners, right_corners)
                print(wp)
                flag_vector[pos] = (True, True)
                point_distance = pdist(wp)
            elif found_right.size <= 0:
                flag_vector[pos] = (True, False)
                print("aruco not detected in right camera")
            elif found_left.size <= 0:
                flag_vector[pos] = (False, True)
                print("element not detected by left camera")
            else:
                print("aruco not detected")
            print(flag_vector)
'''
        
    
class SterioParameter():
    def __init__(self,stereo_para_file):
        with open(stereo_para_file, 'rb') as file_handle:
            print('stereo parameter object loaded...')
            self.cal = pickle.load(file_handle)
            
        self.wL=1280
        self.hL=720
        self.New_M1, tmp = cv2.getOptimalNewCameraMatrix(self.cal.camera_model['M1'],self.cal.camera_model['dist1'],(self.wL,self.hL),1,(self.wL,self.hL))
        self.New_M2, tmp = cv2.getOptimalNewCameraMatrix(self.cal.camera_model['M2'],self.cal.camera_model['dist2'],(self.wL,self.hL),1,(self.wL,self.hL))
                      
        self.projMat_L = self.cal.camera_model['M1'] @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
        self.projMat_R= self.cal.camera_model['M2'] @ cv2.hconcat([self.cal.camera_model['R'], self.cal.camera_model['T']]) # R, T from stereoCalibrate
        
        self.sr_rotation1, self.sr_rotation2, self.sr_pose1, self.sr_pose2 = \
        cv2.stereoRectify(cameraMatrix1 = self.cal.M1, 
                          distCoeffs1 = self.cal.d1,
                          cameraMatrix2 = self.cal.M2, 
                          distCoeffs2 = self.cal.d2, 
                          imageSize = (self.wL,self.hL),
                          R = self.cal.camera_model['R'], 
                          T = self.cal.camera_model['T']                                                   
                          )[0:4]
        
        self.scale_factor = 24.7 #mm
        
    def get_world_points(self, L_key_pnts, R_key_pnts):       
        points3d = np.empty((0,0))
        L_key_pnts = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
        R_key_pnts = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)
        L_key_pnts = cv2.undistortPoints(L_key_pnts,self.cal.M1,self.cal.d1,None, self.cal.camera_model['M1'])
        R_key_pnts = cv2.undistortPoints(R_key_pnts,self.cal.M2,self.cal.d2,None, self.cal.camera_model['M2'])
        
   
        points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts, R_key_pnts)        
        #points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts.reshape(L_key_pnts.shape[0],1,2), R_key_pnts.reshape(R_key_pnts.shape[0],1,2))            
        points3d = (points4d[:3, :]/points4d[3, :]).T
        points3d = points3d*self.scale_factor  
            
        return points3d
        
        
    def get_world_points_product(self, L_key_pnts, R_key_pnts):       
        L_key_pnts = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
        R_key_pnts = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)
        L_key_pnts = cv2.undistortPoints(L_key_pnts,self.cal.M1,self.cal.d1,None, self.cal.camera_model['M1'])
        R_key_pnts = cv2.undistortPoints(R_key_pnts,self.cal.M2,self.cal.d2,None, self.cal.camera_model['M2'])
        

        comb_idx = np.asarray(list(itertools.product(range(L_key_pnts.shape[0]),range(L_key_pnts.shape[0]))))
        #print(comb_idx)
        L_key_pnts = L_key_pnts[comb_idx[:,0],:,:]
        R_key_pnts = R_key_pnts[comb_idx[:,1],:,:]
        
        #print(L_key_pnts.shape)
        #print(R_key_pnts.shape)
        
        points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts, R_key_pnts)
        #points4d = cv2.triangulatePoints(self.projMat_L, self.projMat_R, L_key_pnts.reshape(L_key_pnts.shape[0],1,2), R_key_pnts.reshape(R_key_pnts.shape[0],1,2))
        points3d = (points4d[:3, :]/points4d[3, :]).T
        points3d = points3d*self.scale_factor 
        return points3d
    
    def KeyPointsPairing(self, L_key_pnts, R_key_pnts):
        ret = False
        if L_key_pnts.shape[0]==R_key_pnts.shape[0]:
        #L_key_pnts is N_L X 2
        #R_key_pnts is N_R X 2
    
            
            L_key_pnts_N12 = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
            R_key_pnts_N12 = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)  
            
            # Note: here undistortpoints is to stereorectify points                                          
            L_key_pnts_N12 = cv2.undistortPoints(L_key_pnts_N12,self.cal.M1,self.cal.d1,R=self.sr_rotation1,P=self.sr_pose1)
            R_key_pnts_N12 = cv2.undistortPoints(R_key_pnts_N12,self.cal.M2,self.cal.d2,R=self.sr_rotation2,P=self.sr_pose2)
            
            L_sr_pnts = np.squeeze(L_key_pnts_N12)
            R_sr_pnts = np.squeeze(R_key_pnts_N12)
        
        
            #print(np.mean(L_sr_pnts, axis=0) -  np.mean(R_sr_pnts, axis=0))           
            #print(np.concatenate((L_sr_pnts,R_sr_pnts),axis=1))            
            L_sr_pnts = (L_sr_pnts - np.mean(L_sr_pnts, axis=0))
            R_sr_pnts = (R_sr_pnts - np.mean(R_sr_pnts, axis=0))            
            d = cdist(L_sr_pnts,R_sr_pnts)
            #print(np.min(d, axis=0))
            idx = np.argmin(d, axis=0)
            #print(idx)           
            R_key_pnts = R_key_pnts[idx,:]
            ret = True
        else:
            print('error: unbalanace blobs...')
            
            
        return ret, L_key_pnts, R_key_pnts
 

    
    def KeyPointsPairing_v2(self, L_key_pnts, R_key_pnts):
        #L_key_pnts is NP1 X 2
        #R_key_pnts is NP2 X 2
        
        L_key_pnts_N12 = L_key_pnts.reshape(L_key_pnts.shape[0],1,2)
        R_key_pnts_N12 = R_key_pnts.reshape(R_key_pnts.shape[0],1,2)
        
        L_key_pnts_N12 = cv2.undistortPoints(L_key_pnts_N12,self.cal.M1,self.cal.d1,None, self.cal.camera_model['M1'])
        R_key_pnts_N12 = cv2.undistortPoints(R_key_pnts_N12,self.cal.M2,self.cal.d2,None, self.cal.camera_model['M2'])
        #L_key_pnts_N12 = cv2.undistortPoints(L_key_pnts_N12,self.cal.M1,self.cal.d1,None, self.New_M1)
        #R_key_pnts_N12 = cv2.undistortPoints(R_key_pnts_N12,self.cal.M2,self.cal.d2,None, self.New_M2)
        
        #find epilines corresponding to points in left image and draw them on the right image
        epilines_in_R_for_L_pnts = cv2.computeCorrespondEpilines(L_key_pnts_N12, 1, self.cal.camera_model['F'])
        d_RxL = dist_points_to_lines(R_key_pnts,np.squeeze(epilines_in_R_for_L_pnts))           
        
        #find epilines corresponding to points in right image and draw them on the left image
        epilines_in_L_for_R_pnts = cv2.computeCorrespondEpilines(R_key_pnts_N12, 2, self.cal.camera_model['F'])   
        d_LxR = dist_points_to_lines(L_key_pnts,np.squeeze(epilines_in_L_for_R_pnts))
        
        
        d = d_RxL #+ d_LxR.T
        
        print(d)
        
        #idx = np.argmin(d, axis=0)
        #L_key_pnts = L_key_pnts[idx,:]
        
        idx = np.argmin(d, axis=0)
        R_key_pnts = R_key_pnts[idx,:]
        
        print(idx)
        
        ret = True
                        
        return ret, L_key_pnts, R_key_pnts  # NP1 or NP2 X 2  
        
# ## It is a Object for directly getting 3D point cloud
# class ArUco_MARKER3D():
#     def __init__(self, CONFIG_Obj=None): 
#         # set camera boject
#         if (True):
#             self.FrameCapture = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ=None)   
#             #self.FrameCapture = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
#         self.SteroObj = SterioParameter('IntelD455_Calibration_para.pickle')
#         #self.SteroObj = SterioParameter('stereo_para_v3.pickle')
#         self.ArUco_Obj = ArUco()

          
#     def get_aruco_point_cloud(self):
#         tool_ret = [None]*len(self.ArUco_Obj.tool_vector)
#         wp_list = [None]*len(self.ArUco_Obj.tool_vector)
#         ret, left_img, right_img= self.FrameCapture.get_IR_FRAME_SET()
#         if ret:
#             tool_ret, wp_list = self.ArUco_Obj.get_corner_world_points(self.SteroObj, left_img, right_img)
#         else:
#             print('camera frame not captured...')
#         return tool_ret, wp_list
    
#     def get_aruco_center_point_wp(self):
#         tool_ret = [None]*len(self.ArUco_Obj.tool_vector)
#         wp_list = [None]*len(self.ArUco_Obj.tool_vector)
#         ret, left_img, right_img= self.FrameCapture.get_IR_FRAME_SET()
#         if ret:
#             tool_ret, wp_list = self.ArUco_Obj.get_center_world_points(self.SteroObj, left_img, right_img)
#         else:
#             print('camera frame not captured...')
#         return tool_ret, wp_list
    
# class BLOB_MARKER3D():
#     def __init__(self, CONFIG_Obj=None): 
#         # set camera boject
#         if (True):
#             self.FrameCapture = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ=None) 
#             #self.FrameCapture = CameraCapture_USB(CAMERA_CONFIG_OBJ=None)
#         self.SteroObj = SterioParameter('IntelD455_Calibration_para.pickle')
#         #self.SteroObj = SterioParameter('stereo_para_v3.pickle')
        
#     def get_blob_point_cloud(self):
#         ret = False
#         point_cloud = np.empty((0,0))
#         ret, L_img, R_img = self.FrameCapture.get_IR_FRAME_SET()        
#         try:
#             if ret==True:       
#                 #IntelDispImagePair(L_img, R_img)           
#                 L_key_pnts = blob_detect(L_img, plot_flag=True)
#                 R_key_pnts = blob_detect(R_img, plot_flag=True)   
                
#                 print(L_key_pnts.shape, R_key_pnts.shape)
                
#                 ret, L_key_pnts, R_key_pnts = self.SteroObj.KeyPointsPairing(L_key_pnts, R_key_pnts)
#                 if ret==True:
#                     point_cloud = self.SteroObj.get_world_points(L_key_pnts, R_key_pnts)  
#             else:
#                 print('error: frames is not captured...')        
#             return ret, point_cloud
#         finally:
#             pass
          

        
def dist_points_to_lines(pnts, lines):
    #lines : N x 3 (A,B,C)
    #pnts: M x 2
    pnts = np.asarray(pnts)    
    pnts = np.append(pnts,np.ones((pnts.shape[0],1)),axis=1)
    print('pnts', pnts.shape)    
    numerator = pnts@lines.T
    print('numerator', numerator.shape) 
    denominator = np.sqrt(np.sum(lines.T[:2,:]**2,axis=0))
    print('denominator', denominator.shape)        
    d = numerator/denominator

    return np.abs(d) # size: M X N
    


def blob_detect(image, plot_flag=False):
    
    params = cv2.SimpleBlobDetector_Params()
     
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 120
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.66
     
    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2
         
    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
     
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
         
    # Detect blobs    
    im = cv2.GaussianBlur(image, (3, 3), 0)
    MinThr=(200)
    MaxThr=(256)
    mask = cv2.inRange(im, MinThr, MaxThr)
    
    keypoints = detector.detect(255-mask)   
    blob_pnts = np.zeros((len(keypoints),2))    
    for i in range(len(keypoints)):
        blob_pnts[i,:]= keypoints[i].pt    
        
    

    if plot_flag == True:
     
        # Draw blobs on our image as red circles
        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
        text = "Number of Circular Blobs: " + str(len(keypoints))
        cv2.putText(blobs, text, (20, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
         
        # Show blobs
        cv2.imshow("Filtering Circular Blobs Only", cv2.resize(blobs,(1520,540)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return blob_pnts # matrix NX2


# def IntelDispImagePair(IR_image_left, IR_image_right):
#     resize_left = cv2.resize(IR_image_left, (640, 360))
#     resize_right = cv2.resize(IR_image_right, (640, 360))
#     cv2.imshow('IR image left', resize_left)
#     cv2.imshow('IR image right', resize_right)

# #########################################################################
# '''
# from matplotlib import pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim(-300,300)
# ax.set_ylim(-300,300)
# ax.set_zlim(-300,300)

# M3D = BLOB_MARKER3D()
# ret, p3c = M3D.get_blob_point_cloud()
# sc = ax.scatter(p3c[:,0],p3c[:,1],p3c[:,2])

# while(True):
#     ret, p3c = M3D.get_blob_point_cloud()
#     if ret:
#         sc._offsets3d = (p3c[:,0],p3c[:,1],p3c[:,2])
  
# '''
# '''
# try:
#     while True:
#         #ret, p3c = M3D.get_blob_point_cloud()
#         st = time.time()
#         p3d = M3D.get_blob_point_cloud()
#         print(time.time() - st)
#         print(tool_ret)
#         for i in range(len(tool_ret)):
#             if tool_ret[i]:
#                 print('--------',i)
#                 print(pdist(wp_list[i]))
#         time.sleep(0.025)
# finally:
#     print('done...')
#  '''