"""
File containing all the helper functions for 3D reconstruction
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import operator


# ........................Image Registration.....................................
def get_camera_pose(image, objectPoints, cameraMatrix):   #passes in image to be registered

    # PnP + RANSAC to solve camera pose and orientation

    #https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e
    #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation
    rvecs, tvecs, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoefficients=[])

    x = tvecs[0]
    y = tvecs[1]
    z = tvecs[2]
    pitch = rvecs[0]
    roll = rvecs[1]
    yaw = rvecs[2]

    camera_pose = [x, y, z, pitch, roll, yaw]
    return camera_pose



def get_camera_matrix():

    return []



def get_object_points(images_init):

    return []




# ........................Triangulation.....................................
def Triangulation():


    return




# ........................Bundle Adjustment.....................................
def Bundle_Adjustment():

    return
