"""
File containing all the evaluation functions for the SfM Pipeline
"""
from open3d import *
import numpy as np
import copy
import math

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def align_pcd(source, target):
    """

    :param rc_pcd: reconstructed point cloud
    :param gt_pcd: ground truth point cloud
    :return: aligned_pcd: aligned point cloud
    """

    '''...
    T1 = None
    T2 = None

    T = T2 @ T1
    ...'''

    '''...
    mean_dist, pcd_pair_dict = evaluate_pcd(rc_pcd, gt_pcd)

    #select 16 points:
    #selecting first 16 points in rc_pcd, and the corresponding closest distance points in gt_pcd
    gt = np.zeros((4,16))
    rc = np.zeros((4,16))

    for i in range(16):
        rc[0][i] = rc_pcd[i].x
        rc[1][i] = rc_pcd[i].y
        rc[2][i] = rc_pcd[i].z
        rc[3][i] = 1

        j = pcd_pair_dict.get(i)

        gt[0][i] = gt_pcd[j].x
        gt[1][i] = gt_pcd[j].y
        gt[2][i] = gt_pcd[j].z
        gt[3][i] = 1


    aligned_pcd = T @ rc_pcd
    return aligned_pcd
    ...'''

    threshold = 0.02
    trans_init = np.asarray(
                [[0.862, 0.011, -0.507,  0.5],
                [-0.139, 0.967, -0.215,  0.7],
                [0.487, 0.255,  0.835, -1.4],
                [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = evaluate_registration(source, target,
            threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    draw_registration_result(source, target, reg_p2l.transformation)
    return




def refine_pcd(rc_pcd, gt_pcd):
    """

    :param rc_pcd: reconstructed point cloud
    :param gt_pcd: ground truth point cloud
    :return: ref_pcd: refined point cloud
    """
    ref_pcd = None
    return ref_pcd


def evaluate_pcd(rc_pcd, gt_pcd):
    """

    :param rc_pcd: reconstructed point cloud
    :param gt_pcd: ground truth point cloud
    :return:
    """

    '''...
    num_points_gt = len(gt_pcd)
    num_points_rc = len(rc_pcd)

    pcd_pair_dict = {}

    distances = []
    for i in range(num_points_rc):
        min_distance = float("inf")
        min_point_gt = -1
        for j in range(num_points_gt):
            distance = math.sqrt((rc_pcd[i].x - gt_pcd[i].x)**2 + (rc_pcd[i].y - gt_pcd[i].y)**2 + (rc_pcd[i].z - gt_pcd[i].z)**2)
            if distance < min_distance:
                min_distance = distance
                min_point_gt = j
                pcd_pair_dict.setdefault(i, min_point_gt)
        distances.append(min_distance)


    distances = np.array(distances)
    mean_dist = distances.mean()
    return mean_dist, pcd_pair_dict
    #raise NotImplementedError
    ...'''


def evaluate_cameras(cams, gt_cams):
    """

    :param cams:
    :param gt_cams:
    :return:
    """
    raise NotImplementedError


def main():
    rc_pcd = read_point_cloud("../office1.pcd")
    gt_pcd = read_point_cloud("../office2.pcd")
    align_pcd(rc_pcd, gt_pcd)



if __name__ == "__main__":
    main()
