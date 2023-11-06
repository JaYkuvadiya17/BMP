from save_images import SaveImages
import configparser
import pickle
from camera_calibrate import StereoCalibration
import numpy as np
import math
from KEYPOINTS_DETECT_LIB import SterioParameter , ArUco
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense

config = configparser.ConfigParser()
config.read('config.ini')


def calculate_distance(point1, point2):
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]
    z1 = point1[2]
    z2 = point2[2]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return d

def angle_between_two_aruco(d1,d2):
    d1 = np.array(d1)
    d2 = np.array(d2)

    print(d1.shape)
    n1 = np.cross(d1[2,:]-d1[1,:],d1[2,:]-d1[3,:])    
    n2 = np.cross(d2[2,:]-d2[1,:],d2[2,:]-d2[3,:])   
    v1_u = n1 / np.linalg.norm(n1)
    v2_u = n2 / np.linalg.norm(n2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    angle_degree = np.rad2deg(angle_rad)
    return angle_degree


def CalibrateCamera():
    path = config['PATHS']['CalibrationImagePath']
    img_format = config['IMAGE_FORMAT']['CalibrationImageFormat']

    # calibrating camera and saving calibration file
    cal = StereoCalibration(filepath=path, img_format=img_format)
    try :
        with open('cal.pkl' , 'wb' )as fp:
            pickle.dump(cal,fp)
    except:
        print("Unable to save")

def Detection(ids):
    StereoObj = SterioParameter("cal.pkl")
    obj = ArUco()
    obj.set_tool_vector(ids)

    # img_path = config['PATHS']['ArucoImagePath']
    # img_format = config['IMAGE_FORMAT']['ArucoImageFormat']

    # # reading images from folder
    # left_images =  glob.glob(img_path + 'LEFT/*' + img_format)
    # right_images = glob.glob(img_path + 'RIGHT/*' + img_format)
    # left_images.sort()
    # right_images.sort()

    dist =[]
    angle = []
    far = []

    obj2 = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ=None)
    while True:  
        ret, left, right = obj2.get_IR_FRAME_SET()
        corner_tool_ret , corner_point = obj.get_corner_world_points(StereoObj , left , right)
        center_tool_ret , center_points = obj.get_center_world_points(StereoObj , left , right)
        if(corner_tool_ret == [True]*len(ids) and center_tool_ret == [True]*len(ids)):
            f = np.linalg.norm((center_points[0][0] + center_points[1][0])/2)
            d = calculate_distance(point1=center_points[0][0] , point2 = center_points[1][0])
            a = angle_between_two_aruco(corner_point[0] , corner_point[1])
            far.append(f)
            dist.append(d)
            angle.append(a)
            print("far : " , f)
            print("Distance : " , d) 
            print("angle : " , a)
           
    return dist , far , angle

def plot(dist,far,angle):
    plt.subplot(1,2,1)
    plt.scatter(far,angle)
    plt.xlabel("Far")
    plt.ylabel("Angle")
    plt.subplot(1,2,2)
    plt.scatter(far, dist)
    plt.xlabel("Far")
    plt.ylabel("Distance")
    plt.show()

if __name__ == '__main__':

    # saving Calibration images
    SaveImages.save_images(1)

    # Calibrate Camera
    CalibrateCamera()

    # saving Aruco images
    SaveImages.save_images(2)

    # Detecting Aruco and calculating distance and angle

    ids = [5,8] #Change accordingly 
    dist,far,angle =  Detection(ids)

    # plotting distance and angle

    plot(dist,far,angle)
    