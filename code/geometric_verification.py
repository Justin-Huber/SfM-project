"""
File containing all the helper functions for geometric verification
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d
import random


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 15, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 15, color, -1)
    return img1, img2


def draw_epipolar(img1, img2, F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show(block=False)


def get_K_from_exif(exif_data):
    width = exif_data['ExifImageWidth']
    height = exif_data['ExifImageHeight']
    focal_length = exif_data['FocalLength'][0]
    focal_plane_res_unit = exif_data['FocalPlaneResolutionUnit']
    focal_plane_x_res = exif_data['FocalPlaneXResolution'][0]
    focal_plane_y_res = exif_data['FocalPlaneXResolution'][0]

    # TODO figure out what FocalPlaneResolutionUnit
    # if focal_plane_res_unit == 1:
    #     conversion = 1
    # elif focal_plane_res_unit == 2:
    #     conversion =
    # elif focal_plane_res_unit == 3:
    #     conversion =
    # elif focal_plane_res_unit == 4:
    #     conversion =
    # TODO what we're currently doing, seems decent
    conversion = focal_plane_res_unit

    return np.array([[focal_length * width * conversion / focal_plane_x_res, 0, width / 2],
                      [0, focal_length * height * conversion / focal_plane_y_res, height / 2],
                      [0, 0, 1]])


def compute_F_matrix_residual(F, pt1, pt2):
    """
    Estimates the distance between pt2 and the projected epipolar line created by F and pt1

    :param F:
    :param pt1:
    :param pt2:
    :return:
    """

    if F.shape != (3,3):
        F = np.append(F, 1)
        F = F.reshape((3,3))  # optimizer reshapes to (9,)

    line2 = cv2.computeCorrespondEpilines(pt1.reshape(-1, 1, 2), 1, F)
    line2 = line2.squeeze()
    pt2 = cv2.convertPointsToHomogeneous(np.array([pt2]))
    pt2 = pt2.squeeze()
    dist = np.abs(line2.dot(pt2) / np.sqrt(line2[0]**2 + line2[1]**2))
    return dist


def F_matrix_residuals(F, inlier_pts1, inlier_pts2):
    res = []
    for pt1, pt2, in zip(inlier_pts1, inlier_pts2):
        res.append(compute_F_matrix_residual(F, pt1, pt2))
    return res


from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage import io
from skimage.transform import FundamentalMatrixTransform
from skimage import img_as_ubyte


def findPointCloud(K1, K2, R, t, pts1, pts2):
    I = np.diag(np.ones(R.shape[1]))
    I_zeros = np.append(I, np.zeros((I.shape[0], 1)), axis=1)
    R_t = np.append(R, t, axis=1)

    P1 = K1 @ I_zeros
    P2 = K2 @ R_t

    X = cv2.triangulatePoints(P1[:3], P2[:3], pts1.T.astype(float), pts2.T.astype(float))
    X /= X[3]

    X_prime1 = P1 @ X
    X_prime2 = P2 @ X

    mask1 = X_prime1[2] > 0
    mask2 = X_prime2[2] > 0

    mask = [all(tup) for tup in zip(mask1, mask2)]

    return X[:, mask]


def visualize_pcd(points):
    """
    Visualize the point cloud.
    """
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points)
    # Uncomment this line if you want to paint some color
    pcd.paint_uniform_color([0, 0, 1])
    open3d.draw_geometries([pcd])


def get_best_configuration(K1, K2, F, pts1, pts2):
    E = K2.T @ F @ K1

    R1, R2, t = cv2.decomposeEssentialMat(E)

    X = np.array(([0], [0]))

    best_config = None
    for args in [[R1, t], [R1, -t], [R2, t], [R2, -t]]:
        tmp = findPointCloud(K1, K2, *args, pts1, pts2)

        if tmp.shape[1] > X.shape[1]:
            X = tmp
            best_config = args

    return best_config


def visualize_gv_from_F(K1, K2, F, pts1, pts2):
    E = K2.T @ F @ K1

    R1, R2, t = cv2.decomposeEssentialMat(E)

    X = np.array(([0], [0]))

    best_R, best_t = None, None
    for args in [[R1, t], [R1, -t], [R2, t], [R2, -t]]:
        tmp = findPointCloud(K1, K2, *args, pts1, pts2)

        if tmp.shape[1] > X.shape[1]:
            X = tmp
            best_R, best_t = args

    visualize_pcd(X[:3].astype(float).T)

    return best_R, best_t


def visualize_gv(K1, K2, R, t, pts1, pts2):
    X = findPointCloud(K1, K2, R, t, pts1, pts2)

    try:  # For visualizing 3D plots on Google Colab
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False

    if IN_COLAB:
        from plotly.offline import iplot
        import plotly.graph_objs as go

        import numpy as np

        trace = go.Scatter3d(
            x=X[0],
            y=X[1],
            z=X[2],
            mode='markers',
            marker=dict(
                size=1,
                line=dict(
                    color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                opacity=0.8
            )
        )

        data = [trace]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            )
        )

        iplot(data)
    else:
        visualize_pcd(X[:3].astype(float).T)


def visualize_gt(obj_filename):
    """
    Function for visualizing the ground truth point cloud

    :param obj_filename: filename for .obj file
    :return:
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
    visualize_pcd(points.astype(float).T)


def visualize_matches(img_left, img_right, keypoints_left, keypoints_right, matches):
    # Plot all

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img_left, img_right, keypoints_left, keypoints_right,
                 matches, only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Matches")

    plt.show(block=False)

    plt.cla()


def visualize_cv_matches(img_left, img_right, keypoints_left, keypoints_right, matches, inliers):
    cv_keypoints_left = np.array([cv2.KeyPoint(x=point[1], y=point[0],
                                               _size=0, _angle=0,
                                               _response=0, _octave=0,
                                               _class_id=0) for point in keypoints_left])
    cv_keypoints_right = np.array([cv2.KeyPoint(x=point[1], y=point[0],
                                                _size=0, _angle=0,
                                                _response=0, _octave=0,
                                                _class_id=0) for point in keypoints_right])
    cv_matches = np.array([cv2.DMatch(_distance=0, _imgIdx=0,
                                      _queryIdx=match[0], _trainIdx=match[1]) for match in matches])
    cv_inlier_matches = cv_matches[inliers]
    img = cv2.drawMatches(img_left, cv_keypoints_left, img_right, cv_keypoints_right, cv_inlier_matches, None,
                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img), plt.show(block=False)


import time
from scipy.optimize import minimize, leastsq, least_squares
global_pts1 = []
global_pts2 = []

def cv_impl(file1, file2, visualize_all_matches, visualize_good_matches, visualize_epipoles, n_keypoints):
    start_time = time.time()
    # Find sparse feature correspondences between left and right image.
    im1 = cv2.imread(file1, 0)
    im2 = cv2.imread(file2, 0)

    # sift = cv2.xfeatures2d.SIFT_create(n_keypoints)
    # kp1, des1 = sift.detectAndCompute(im1, None)
    # kp2, des2 = sift.detectAndCompute(im2, None)

    orb = cv2.ORB_create(n_keypoints)
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    pts1 = np.array([kp.pt for kp in kp1])
    pts2 = np.array([kp.pt for kp in kp2])

    # # FLANN parameters
    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # TODO set params differently?
    # search_params = dict(checks=50)  # or pass empty dictionary
    # matcher = cv2.FlannBasedMatcher(index_params, search_params)
    # cv_matches = matcher.knnMatch(des1, des2, k=2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    cv_matches = bf.match(des1, des2)
    cv_matches = np.array(cv_matches)
    matches = np.array([(match.queryIdx, match.trainIdx) for match in cv_matches])
    cv_end_time = time.time()

    print('CV Runtime: ', cv_end_time - start_time)

    print("Number of matches:", len(cv_matches))

    threshold = 0.001 * max(im1.shape)
    F, inliers_mask = cv2.findFundamentalMat(pts1[matches[:, 0]],
                                             pts2[matches[:, 1]],
                                             method=cv2.FM_RANSAC,
                                             ransacReprojThreshold=threshold,
                                             confidence=0.999)

    print("Number of inliers:", inliers_mask.sum())
    inliers_mask = inliers_mask.ravel() == 1

    inlier_pts1 = pts1[matches[inliers_mask, 0]]
    inlier_pts1 = inlier_pts1.astype(int)
    inlier_pts2 = pts2[matches[inliers_mask, 1]]
    inlier_pts2 = inlier_pts2.astype(int)

    if visualize_good_matches:
        img = cv2.drawMatches(im1, kp1, im2, kp2, cv_matches[inliers_mask], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img), plt.show(block=False)

    if visualize_epipoles:
        draw_epipolar(im1, im2, F, inlier_pts1, inlier_pts2)

    global global_pts1
    global global_pts2
    global_pts1 = inlier_pts1
    global_pts2 = inlier_pts2
    # TODO optimize F according to the inliers using Levenberg-Marquardt
    F, _ = leastsq(func=F_matrix_residuals, x0=F.flatten()[:-1].reshape(-1, 1), args=(inlier_pts1, inlier_pts2))

    F = np.append(F, 1).reshape(3, 3)

    U, SIGMA, V_T = np.linalg.svd(F)
    SIGMA[2] = 0
    F = U @ np.diag(SIGMA) @ V_T

    print(F)

    # TODO get inliers to optimized F
    inliers_mask_conf = np.zeros(len(inliers_mask))

    for i, match in enumerate(matches):
        pt1 = pts1[match[0]]
        pt2 = pts2[match[1]]
        dist = compute_F_matrix_residual(F, pt1, pt2)
        if dist < threshold:
            inliers_mask_conf[i] = 1

    inliers_mask_conf = inliers_mask_conf == 1

    num_diff = 0
    for i, i_conf in zip(inliers_mask, inliers_mask_conf):
        if i != i_conf:
            num_diff += 1

    print('Number of inliers in disagreement: ', num_diff)

    inlier_pts1 = pts1[matches[inliers_mask_conf, 0]]
    inlier_pts1 = inlier_pts1.astype(int)
    inlier_pts2 = pts2[matches[inliers_mask_conf, 1]]
    inlier_pts2 = inlier_pts2.astype(int)

    if visualize_all_matches:
        img = cv2.drawMatches(im1, kp1, im2, kp2, cv_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img), plt.show(block=False)

    if visualize_good_matches:
        img = cv2.drawMatches(im1, kp1, im2, kp2, cv_matches[inliers_mask_conf], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img), plt.show(block=False)

    if visualize_epipoles:
        draw_epipolar(im1, im2, F, inlier_pts1, inlier_pts2)

    # Draw 3D
    from feature_extraction import get_human_readable_exif
    exif_data1 = get_human_readable_exif(file1)
    exif_data2 = get_human_readable_exif(file2)

    # TODO ask about this
    K1 = get_K_from_exif(exif_data1)
    K2 = get_K_from_exif(exif_data2)

    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    K1 = np.array([[2100.0000, 0.0000, 960.0000],
                   [0.0000, 2100.0000, 540.0000],
                   [0.0000, 0.0000, 1.0000]])
    K2 = np.array([[2100.0000, 0.0000, 960.0000],
                   [0.0000, 2100.0000, 540.0000],
                   [0.0000, 0.0000, 1.0000]])

    R, t = visualize_gv_from_F(K1, K2, F, inlier_pts1, inlier_pts2)
    visualize_gv(K1, K2, R, t, inlier_pts1, inlier_pts2)


if __name__ == '__main__':
    file1 = '../datasets/Jeep/images/0000.jpg'
    file2 = '../datasets/Jeep/images/0002.jpg'
    obj_filename = '../datasets/Jeep/Jeep-model.obj'

    visualize_all_matches = False
    visualize_good_matches = True
    visualize_epipoles = True
    visualize_ground_truth = False

    n_keypoints = 8000

    np.random.seed(0)

    if visualize_ground_truth:
        visualize_gt(obj_filename)
    cv_impl(file1, file2, visualize_all_matches, visualize_good_matches, visualize_epipoles, n_keypoints)


def sk_impl(file1, file2, visualize_all_matches, visualize_good_matches, visualize_epipoles, n_keypoints):
    start_time = time.time()
    # Find sparse feature correspondences between left and right image.
    img_left, img_right = io.imread(file1), io.imread(file2)
    img_left, img_right = map(rgb2gray, (img_left, img_right))

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(img_left)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_right)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)
    sk_end_time = time.time()

    print('SK Runtime: ', sk_end_time - start_time)

    print("Number of matches:", matches.shape[0])

    if visualize_all_matches:
        visualize_matches(img_left, img_right, keypoints_left, keypoints_right, matches)

    img_left = img_as_ubyte(img_left)
    img_right = img_as_ubyte(img_right)

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=0.001*max(img_left.shape), max_trials=5000)

    print("Number of inliers:", inliers.sum())

    if visualize_good_matches:
        visualize_cv_matches(img_left, img_right, keypoints_left, keypoints_right, matches, inliers)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_left = inlier_keypoints_left.astype(int)
    inlier_keypoints_left = np.array([(x, y) for (y, x) in inlier_keypoints_left])
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]
    inlier_keypoints_right = inlier_keypoints_right.astype(int)
    inlier_keypoints_right = np.array([(x, y) for (y, x) in inlier_keypoints_right])

    # TODO how to properly select
    CV8_F, mask = cv2.findFundamentalMat(inlier_keypoints_left, inlier_keypoints_right,
                                         method=cv2.FM_8POINT)
    print('CV8 F: ', CV8_F)
    if visualize_epipoles:
        draw_epipolar(img_left, img_right,
                      CV8_F,
                      inlier_keypoints_left, inlier_keypoints_right)

    sample_from = np.random.choice(inlier_keypoints_left.shape[0], 20)
    sampled_kp_left = inlier_keypoints_left[sample_from, :]
    sampled_kp_right = inlier_keypoints_right[sample_from, :]

    # TODO potentially another round of RANSAC here to initialize F correctly
    # Create F matrices
    CV8_F, mask = cv2.findFundamentalMat(sampled_kp_left, sampled_kp_right, method=cv2.FM_8POINT)

    CV8_F, mask = cv2.findFundamentalMat(inlier_keypoints_left, inlier_keypoints_right,
                                         method=cv2.FM_8POINT)

    # Draw 3D
    from feature_extraction import get_human_readable_exif
    exif_data1 = get_human_readable_exif(file1)
    exif_data2 = get_human_readable_exif(file2)

    K1 = np.array([[1520.4, 0, 302.3],
                   [0, 1525.9, 246.9],
                   [0, 0, 1]])
    K2 = np.array([[1520.4, 0, 302.3],
                   [0, 1525.9, 246.9],
                   [0, 0, 1]])

    # # TODO ask about this
    K1 = get_K_from_exif(exif_data1)
    K2 = get_K_from_exif(exif_data2)

    visualize_gv_from_F(K1, K2, CV8_F, inlier_keypoints_left, inlier_keypoints_right)
