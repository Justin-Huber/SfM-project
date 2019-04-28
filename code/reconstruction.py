"""
File containing all the helper functions for 3D reconstruction
"""

import cv2
import numpy as np
import open3d
import time
import scipy
from scipy.optimize import leastsq

from feature_extraction import get_human_readable_exif



def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False


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


def get_pcd(points):
    """
    Get pcd from points
    """
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd


def get_gt_points(obj_filename):
    """
    Function for returning the ground truth point cloud

    :param obj_filename: filename for .obj file
    :return: ground truth point cloud points as array
    """

    x, y, z = [], [], []
    with open(obj_filename, 'r') as f:
        for line in f.readlines():
            vals = line.split()
            if len(vals) is 4:
                t, x_i, y_i, z_i = vals

                if t is 'v':
                    x.append(x_i)
                    y.append(y_i)
                    z.append(z_i)

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    points = np.array([x, y, z])
    return points

