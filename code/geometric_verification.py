import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d


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
    plt.show()


def get_K_from_exif(exif_data):
    width = exif_data['ExifImageWidth']
    height = exif_data['ExifImageHeight']
    focal_length = exif_data['FocalLength'][0]
    focal_plane_res_unit = exif_data['FocalPlaneResolutionUnit']
    focal_plane_x_res = exif_data['FocalPlaneXResolution'][0]
    focal_plane_y_res = exif_data['FocalPlaneXResolution'][0]

    # if focal_plane_res_unit == 1:
    #     conversion = 1
    # elif focal_plane_res_unit == 2:
    #     conversion =
    # elif focal_plane_res_unit == 3:
    #     conversion =
    # elif focal_plane_res_unit == 4:
    #     conversion =

    conversion = focal_plane_res_unit

    return np.array([[focal_length * width * conversion / focal_plane_x_res, 0, width / 2],
                      [0, focal_length * height * conversion / focal_plane_y_res, height / 2],
                      [0, 0, 1]])


from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage import io
from skimage.transform import FundamentalMatrixTransform
from skimage import img_as_ubyte


def findFundamentalMat(im1, im2, pts1, pts2, eig_then_svd=False):
    ones = np.ones((pts1.shape[0], 1))
    homog_pts1 = np.append(pts1, ones, axis=1)
    homog_pts2 = np.append(pts2, ones, axis=1)

    T = np.diag(np.ones(homog_pts1.shape[1]) / np.max(im1.shape))
    T[-1, -1] = 1
    T_prime = np.diag(np.ones(homog_pts2.shape[1]) / np.max(im2.shape))
    T_prime[-1, -1] = 1

    # mean1 = np.mean(pts1, axis=0)
    # mean2 = np.mean(pts2, axis=0)
    # T = np.diag(np.ones(homog_pts1.shape[1]))
    # T[0, 2] = -mean1[1]
    # T[1, 2] = -mean1[0]
    # T_prime = np.diag(np.ones(homog_pts2.shape[1]))
    # T_prime[0, 2] = -mean2[1]
    # T_prime[1, 2] = -mean2[0]

    pts1_norm = homog_pts1.dot(T)
    pts2_norm = homog_pts2.dot(T_prime)

    A = np.array(
        [(pt1.reshape(1, *pt1.shape) * pt2.reshape(*pt2.shape, 1)).flatten() for pt1, pt2 in zip(pts1_norm, pts2_norm)])

    if eig_then_svd:
        eig_vals, eig_vecs = np.linalg.eig(A.T.dot(A))
        idx = np.argmin(eig_vals)
        F = eig_vecs[:, idx].reshape(3, 3)
    else:
        U, SIGMA, V_T = np.linalg.svd(A.T @ A, full_matrices=True)
        F = V_T[8].reshape(3, 3)

    U, SIGMA, V_T = np.linalg.svd(F, full_matrices=True)
    SIGMA[2:] = 0
    F = U @ np.diag(SIGMA) @ V_T
    F /= F[2, 2]
    F = T_prime.T @ F @ T

    return F





def findPointCloud(R, t, pts1, pts2):
    #####################################################################
    # 4
    #####################################################################
    I = np.diag(np.ones(R.shape[1]))
    I_zeros = np.append(I, np.zeros((I.shape[0], 1)), axis=1)
    R_t = np.append(R, t, axis=1)

    P1 = K1 @ I_zeros
    P2 = K2 @ R_t

    #####################################################################
    # 5
    #####################################################################
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
    # pcd.paint_uniform_color([0, 0, 1])
    open3d.draw_geometries([pcd])


def visualize_gv(K1, K2, F, pts1, pts2):
    E = K2.T @ F @ K1

    R1, R2, t = cv2.decomposeEssentialMat(E)

    X = np.array(([0], [0]))

    for args in [[R1, t], [R1, -t], [R2, t], [R2, -t]]:
        tmp = findPointCloud(*args, pts1, pts2)

        if tmp.shape[1] > X.shape[1]:
            X = tmp
        # visualize_pcd(tmp[:3].astype(float).T)

    visualize_pcd(X[:3].astype(float).T)


def visualize_matches(img_left, img_right, keypoints_left, keypoints_right, matches):
    # Plot all

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img_left, img_right, keypoints_left, keypoints_right,
                 matches, only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Matches")

    plt.show()
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
    plt.imshow(img), plt.show()


if __name__ == '__main__':
    # TODO pixel bug?
    np.random.seed(0)
    file1 = '../datasets/EmpireVase/images/0090.jpg'
    file2 = '../datasets/EmpireVase/images/0098.jpg'
    # file1 = '../data/im1.png'
    # file2 = '../data/im2.png'
    visualize_all_matches = False
    visualize_good_matches_sk = False
    visualize_good_matches_cv = True
    visualize_8pt = False
    visualize_ransac = False
    visualize_mine = True

    temple = np.load('../data/temple.npz')
    K1, K2, pts1, pts2 = temple['K1'], temple['K2'], temple['pts1'], temple['pts2']
    im1, im2 = cv2.imread('../data/im1.png', cv2.IMREAD_GRAYSCALE), cv2.imread('../data/im2.png', cv2.IMREAD_GRAYSCALE)

    false = True
    if false:
        img_left, img_right = io.imread(file1), io.imread(file2)
        # img_left, img_right, _ = data.stereo_motorcycle()
        img_left, img_right = map(rgb2gray, (img_left, img_right))

        # Find sparse feature correspondences between left and right image.

        descriptor_extractor = ORB(n_keypoints=8000)

        descriptor_extractor.detect_and_extract(img_left)
        keypoints_left = descriptor_extractor.keypoints
        descriptors_left = descriptor_extractor.descriptors

        descriptor_extractor.detect_and_extract(img_right)
        keypoints_right = descriptor_extractor.keypoints
        descriptors_right = descriptor_extractor.descriptors

        matches = match_descriptors(descriptors_left, descriptors_right,
                                    cross_check=True)
        print("Number of matches:", matches.shape[0])

        if visualize_all_matches:
            visualize_matches(img_left, img_right, keypoints_left, keypoints_right, matches)

        model, inliers = ransac((keypoints_left[matches[:, 0]],
                                 keypoints_right[matches[:, 1]]),
                                FundamentalMatrixTransform, min_samples=8,
                                residual_threshold=0.001*max(img_left.shape), max_trials=5000)
        print("Number of inliers:", inliers.sum())

        if visualize_good_matches_sk:
            visualize_matches(img_left, img_right, keypoints_left, keypoints_right, matches[inliers])

        img_left = img_as_ubyte(img_left)
        img_right = img_as_ubyte(img_right)

        if visualize_good_matches_cv:
            visualize_cv_matches(img_left, img_right, keypoints_left, keypoints_right, matches, inliers)

        inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
        inlier_keypoints_left = inlier_keypoints_left.astype(int)
        inlier_keypoints_left = np.array([(x, y) for (y, x) in inlier_keypoints_left])
        inlier_keypoints_right = keypoints_right[matches[inliers, 1]]
        inlier_keypoints_right = inlier_keypoints_right.astype(int)
        inlier_keypoints_right = np.array([(x, y) for (y, x) in inlier_keypoints_right])
        model.params /= model.params[2,2]

        sample_from = np.random.choice(inlier_keypoints_left.shape[0], 20)
        sampled_kp_left = inlier_keypoints_left[sample_from, :]
        sampled_kp_right = inlier_keypoints_right[sample_from, :]

        # TODO potentially another round of RANSAC here to initialize F correctly
        # Create F matrices
        CV8_F, mask = cv2.findFundamentalMat(sampled_kp_left, sampled_kp_right, method=cv2.FM_8POINT)
        CVR_F, mask = cv2.findFundamentalMat(sampled_kp_left, sampled_kp_right, method=cv2.FM_RANSAC)
        MY_F = findFundamentalMat(img_left, img_right, sampled_kp_left, sampled_kp_right)
    else:
        img_left = im1
        img_right = im2
        inlier_keypoints_left = pts1
        inlier_keypoints_right = pts2

    CV8_F, mask = cv2.findFundamentalMat(inlier_keypoints_left, inlier_keypoints_right, method=cv2.FM_8POINT)
    CVR_F, mask = cv2.findFundamentalMat(inlier_keypoints_left, inlier_keypoints_right, method=cv2.FM_RANSAC)
    MY_F = findFundamentalMat(img_left, img_right, inlier_keypoints_left, inlier_keypoints_right, True)
    F = findFundamentalMat(im1, im2, pts1, pts2, True)

    print('CV8 F: ', CV8_F)
    if visualize_8pt:
        draw_epipolar(img_left, img_right,
                      CV8_F,
                      inlier_keypoints_left, inlier_keypoints_right)
    print('CVR F: ', CVR_F)
    if visualize_ransac:
        draw_epipolar(img_left, img_right,
                      CVR_F,
                      inlier_keypoints_left, inlier_keypoints_right)
    print('My F: ', MY_F)
    if visualize_mine:
        draw_epipolar(img_left, img_right,
                      MY_F,
                      inlier_keypoints_left, inlier_keypoints_right)

    # Draw 3D
    from feature_extraction import get_human_readable_exif
    import os
    exif_data1 = get_human_readable_exif(file1)
    exif_data2 = get_human_readable_exif(file2)

    K1 = np.array([[1520.4, 0, 302.3],
                   [0, 1525.9, 246.9],
                   [0, 0, 1]])
    K2 = np.array([[1520.4, 0, 302.3],
                   [0, 1525.9, 246.9],
                   [0, 0, 1]])

    # TODO focal length?
    K1 = np.array([[1520.4, 0, img_left.shape[1] / 2],
                   [0, 1520.4, img_left.shape[0] / 2],
                   [0, 0, 1]])
    K2 = np.array([[1520.4, 0, img_right.shape[1] / 2],
                   [0, 1520.4, img_right.shape[0] / 2],
                   [0, 0, 1]])

    # TODO ask about this
    K1 = get_K_from_exif(exif_data1)
    K2 = get_K_from_exif(exif_data2)

    # TODO doesnt work
    # K1 = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]])
    # K2 = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]])

    visualize_gv(K1, K2, F, inlier_keypoints_left, inlier_keypoints_right)
