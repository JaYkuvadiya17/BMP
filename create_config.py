import configparser

config = configparser.ConfigParser()

#change according to your path
config["PATHS"] = {
    'ArucoImagePath' : '/Users/jaykuvadiya/DAIICT/BMP/Stereo_final/Aruco_Images/',
    'CalibrationImagePath' : '/Users/jaykuvadiya/DAIICT/BMP/Stereo_final/Calibration_Images/',
}

# change according to your image format
config["IMAGE_FORMAT"] = {
    'ArucoImageFormat' : '.png',
    'CalibrationImageFormat' : '.png',
}


with open('config.ini', 'w') as configfile:
    config.write(configfile)