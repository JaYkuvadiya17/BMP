import cv2
from CAMERA_CAPTURE_MODULE_MAIN import ComeraCapture_RealSense
import time
import configparser


class SaveImages:
    def __init__(self):
        pass
    def save_images(key):
        obj = ComeraCapture_RealSense(CAMERA_CONFIG_OBJ=None)
        config = configparser.ConfigParser()
        config.read('config.ini')
        if key == '1':
            path = config['PATHS']['CalibrationImagePath']
            img_format = config['IMAGE_FORMAT']['CalibrationImageFormat']
        elif key == '2':
            path = config['PATHS']['ArucoImagePath']
            img_format = config['IMAGE_FORMAT']['ArucoImageFormat']
        else:
            print("Invalid choice")
        i = 0
        while True:
            ret, L_img, R_img = obj.get_IR_FRAME_SET()
            if ret:
                cv2.imshow('L_img', L_img)
                cv2.imwrite(f'{path}/L_img_{i}.{img_format}', L_img)
                cv2.imshow('R_img', R_img)
                cv2.imwrite(f'{path}/R_img_{i}.{img_format}', R_img)
                ## Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                i += 1
                time.sleep(0.1)
            else:
                print('No image')
                break