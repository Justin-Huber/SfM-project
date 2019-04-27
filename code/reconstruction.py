"""
File containing all the helper functions for 3D reconstruction
"""

import cv2
import numpy as np


def get_camera_pose(pts2d, pts3d, K):
    # TODO fix threshold, currently hardcoded
    _, rvec, tvec, _ = cv2.solvePnPRansac(np.array(pts3d), np.array(pts2d), cameraMatrix=K, distCoeffs=np.zeros((4, 1)), reprojectionError=0.004*1980)
    rotM = cv2.Rodrigues(rvec)[0]
    return rotM, tvec


def get_angle(a, b, c):
    """
    Returns angle where pt2 is the middle one
    :param pt1:
    :param pt2:
    :param pt3:
    :return:
    """

    a = a.squeeze()
    b = b.squeeze()
    c = c.squeeze()

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
