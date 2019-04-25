"""
File containing all the evaluation functions for the SfM Pipeline
"""

def align_pcd(rc_pcd, gt_pcd):
    """

    :param rc_pcd: reconstructed point cloud
    :param gt_pcd: ground truth point cloud
    :return: aligned_pcd: aligned point cloud
    """

    T1 = None
    T2 = None

    T = T2 @ T1

    aligned_pcd = T @ rc_pcd
    return aligned_pcd


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
    raise NotImplementedError


def evaluate_cameras(cams, gt_cams):
    """

    :param cams:
    :param gt_cams:
    :return:
    """
    raise NotImplementedError
