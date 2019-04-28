"""
File containing the SfM Pipeline
"""

import os
import warnings
import pickle
import matplotlib.pyplot as plt
import cv2
try:  # For using tqdm on Google Colab
  import google.colab
  IN_COLAB = True
  from tqdm import tqdm_notebook as tqdm
except:
  IN_COLAB = False
  from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed
from itertools import combinations, product
import numpy as np
import networkx as nx
from scipy.optimize import leastsq
import open3d

from feature_extraction import get_human_readable_exif, populate_keypoints_and_descriptors, deserialize_keypoints
from feature_matching import serialize_matches, deserialize_matches
from geometric_verification import draw_epipolar, get_K_from_exif, visualize_pcd, visualize_gv,\
                                    get_best_configuration, F_matrix_residuals, compute_F_matrix_residual, findPointCloud
from reconstruction import get_camera_pose, get_angle, rotate_view

FLANN = 0
BF = 1


class Pipeline:
    """
    A SfM Pipeline for reconstructing a 3D scene from a set of 2D images.

    """
    def __init__(self, images_dir, output_dir=os.path.abspath(os.path.join(os.getcwd(), '..')), **kwargs):
        """

        :param images_dir: directory containing set of 2D images
                           must only be filled with images
        :param output_dir: directory where pipeline/* will be created
        """
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.n_keypoints = kwargs.pop('n_keypoints', 100)
        self.verbose = kwargs.pop('verbose', False)
        self.gv_threshold = kwargs.pop('gv_threshold', 20)
        self.init_threshold = kwargs.pop('init_threshold', 100)
        self.dos_threshold = kwargs.pop('dos_threshold', 2)
        self.n_jobs = kwargs.pop('n_jobs', 1)

        self._init_pipeline_file_structure()

        # check valid image dir
        self.images = []

    def _init_pipeline_file_structure(self):
        if not os.path.isdir(self.output_dir):
            raise RuntimeError("Invalid output directory")

        self.pipeline_dir = os.path.join(self.output_dir, 'pipeline')
        self.feature_extraction_dir = os.path.join(self.pipeline_dir, 'feature_extraction')
        self.feature_matching_dir = os.path.join(self.pipeline_dir, 'feature_matching')
        self.geometric_verification_dir = os.path.join(self.pipeline_dir, 'geometric_verification')
        self.reconstruction_dir = os.path.join(self.pipeline_dir, 'reconstruction')

        if not os.path.exists(self.pipeline_dir):
            os.mkdir(self.pipeline_dir)
        if not os.path.exists(self.feature_extraction_dir):
            os.mkdir(self.feature_extraction_dir)
        if not os.path.exists(self.feature_matching_dir):
            os.mkdir(self.feature_matching_dir)
        if not os.path.exists(self.geometric_verification_dir):
            os.mkdir(self.geometric_verification_dir)
        if not os.path.exists(self.reconstruction_dir):
            os.mkdir(self.reconstruction_dir)

    def run(self):
        pickled_pcd = os.path.join(self.reconstruction_dir, 'pcd.pkl')
        if os.path.exists(pickled_pcd):
            with open(pickled_pcd, 'rb') as f:
                self.pcd, self.camera3Dpose = pickle.load(f)
        else:
            self._extract_features()  # extract features using ORB
            self._match_features()  # match features using FLANN
            self._geometric_verification()
            self._init_reconstruction()
            self._reconstruct3d()
            with open(pickled_pcd, 'wb') as f:
                pickle.dump((self.pcd, self.camera3Dpose), f)

    def _load_images_impl(self):
        """

        :return: returns list of images in the images directory
                in the same order they appear in the directory
                and a list of exif information for those images
        """
        if not os.path.exists(self.images_dir):
            raise RuntimeError("Invalid image directory")

        img_filenames = os.listdir(self.images_dir)

        exif_data = [get_human_readable_exif(os.path.join(self.images_dir, img_filename)) for img_filename in img_filenames]

        images = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(cv2.imread)(os.path.join(self.images_dir, img_filename), 0)
                                                            for img_filename in tqdm(img_filenames, desc='Loading images'))

        if self.verbose and input("Visualize image loading? (y/n) ") == 'y':
            plt.imshow(images[0]), plt.show()

        return images, exif_data

    def _load_images(self):
        pickled_images = os.path.join(self.feature_extraction_dir, 'images_and_exif.pkl')
        if os.path.exists(pickled_images):
            with open(pickled_images, 'rb') as f:
                images, exif_data = pickle.load(f)
        else:
            images, exif_data = self._load_images_impl()
            with open(pickled_images, 'wb') as f:
                pickle.dump((images, exif_data), f)

        return images, exif_data

    def _extract_features_impl(self):
        # TODO move implementation of feature extraction from feature_extraction.py to here
        raise NotImplementedError

    def _extract_features(self):
        # TODO add check that self.n_keypoints and loaded keypoints agree
        self.images, self.exif_data = self._load_images()
        self.num_images = len(self.images)

        pickled_keypoints = os.path.join(self.feature_extraction_dir, 'keypoints.pkl')
        if os.path.exists(pickled_keypoints):
            with open(pickled_keypoints, 'rb') as f:
                self.keypoints_and_descriptors = pickle.load(f)
        else:
            self.keypoints_and_descriptors = populate_keypoints_and_descriptors(self.images, self.n_keypoints, self.n_jobs)
            with open(pickled_keypoints, 'wb') as f:
                pickle.dump(self.keypoints_and_descriptors, f)

        # deserialize the keypoints
        self.keypoints_and_descriptors = [deserialize_keypoints(kps_and_des) for kps_and_des in self.keypoints_and_descriptors]
        # self.keypoints_and_descriptors = np.array(self.keypoints_and_descriptors)

        if self.verbose and input("Visualize keypoints? (y/n) ") == 'y':
            img = self.images[0]
            vis_keypoints = self.keypoints_and_descriptors[0][0]
            vis_img = cv2.drawKeypoints(img, vis_keypoints, img, flags=4)  # draws rich keypoints
            plt.imshow(vis_img), plt.show()

    def visualize_matches(self, i, j, mask=None):
        """
        Function to visualize matches between two images by their indices

        :param i: image i
        :param j: image j
        :return:
        """

        # TODO make more elegant if possible
        if mask is None:
            img = cv2.drawMatches(self.images[i], self.keypoints_and_descriptors[i][0],
                                   self.images[j], self.keypoints_and_descriptors[j][0],
                                   self.matches[i][j], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        else:
            # TODO change all other lists to numpy ArrayItemContainers
            matches_as_np = np.array(self.matches[i][j])[mask == 1]
            img = cv2.drawMatches(self.images[i], self.keypoints_and_descriptors[i][0],
                                  self.images[j], self.keypoints_and_descriptors[j][0],
                                  matches_as_np, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img), plt.show(block=False), plt.pause(0.001)

    def _match_features_parallel(self):
        raise NotImplementedError

    def _match_features_impl(self):
        # TODO add debug option to visualize
        if self.verbose and input("Visualize matches impl? (y/n) ") == 'y':
            visualize = True
        else:
            visualize = False

        type = FLANN

        if type == FLANN:
            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=200)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # TODO upper triangular matrix of matches between all combinations of images
        image_matches = []
        been_matched = set()  # for keeping track of what has already been fully matched

        for i, kp_and_des1 in tqdm(enumerate(self.keypoints_and_descriptors), desc='Matching image features'):
            been_matched.add(i)
            i_matches = []
            for j, kp_and_des2 in enumerate(self.keypoints_and_descriptors):
                i_with_j_matches = []
                if j not in been_matched:
                    kp1, des1 = kp_and_des1
                    kp2, des2 = kp_and_des2

                    matches = matcher.knnMatch(des1, des2, k=2)

                    # Need to draw only good matches, so create a mask
                    matchesMask = [[0, 0] for _ in range(len(matches))]

                    # ratio test as per Lowe's paper
                    for k, (m, n) in enumerate(matches):
                        if m.distance < 0.6 * n.distance:
                            matchesMask[k] = [1, 0]

                    # remove matches who map to same keypoint in j
                    counts = {match.trainIdx: 0 for match, _ in matches}
                    for match, _ in matches:
                        counts[match.trainIdx] += 1

                    i_with_j_matches = []
                    for (match, _), (matchMask, _) in zip(matches, matchesMask):
                        if matchMask and counts[match.trainIdx] == 1:
                            i_with_j_matches.append(match)

                    if visualize:
                        img1 = self.images[i]
                        img2 = self.images[j]

                        draw_params = dict(matchColor=(0, 255, 0),
                                           singlePointColor=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, None, **draw_params)
                        plt.imshow(img3), plt.show()

                i_matches.append(serialize_matches(i_with_j_matches))
            image_matches.append(i_matches)

        return image_matches

    def _match_features(self):
        pickled_matches = os.path.join(self.feature_matching_dir, 'matches.pkl')
        if os.path.exists(pickled_matches):
            with open(pickled_matches, 'rb') as f:
                self.matches = pickle.load(f)
        else:
            self.matches = self._match_features_impl()
            with open(pickled_matches, 'wb') as f:
                pickle.dump(self.matches, f)

        self.matches = [[deserialize_matches(ij_matches) for ij_matches in i_matches] for i_matches in self.matches]
        self.matches = np.array([[np.array(sorted(ij_matches, key=lambda match: match.distance))
                                  for ij_matches in i_matches] for i_matches in self.matches])

        if self.verbose and input("Visualize matches? (y/n) ") == 'y':
            for i in range(self.num_images):
                for j in range(self.num_images):
                    if len(self.matches[i][j]) > 0:
                        self.visualize_matches(i, j)
            # TODO add visualization

    def _geometric_verification_impl(self, i, j):
        # TODO multithreading accessing a list, how does it work?
        # TODO check that GIL isn't preventing parallelism
        kp1 = self.keypoints_and_descriptors[i][0]
        kp2 = self.keypoints_and_descriptors[j][0]

        pts1 = np.array([kp.pt for kp in kp1])
        pts2 = np.array([kp.pt for kp in kp2])

        exif_data1 = self.exif_data[i]
        exif_data2 = self.exif_data[j]

        height, width = exif_data1['ExifImageHeight'], exif_data1['ExifImageWidth']

        ransacReprojThreshold = 0.006*max(height, width)

        matches = np.array([(match.queryIdx, match.trainIdx) for match in self.matches[i, j]])

        if len(matches) > 8:
            # Args:
            # - ransacReprojThreshold: threshold for inliers
            #                            1-3 recommended by OpenCV documentation
            #                            0.006*max(img dimensions) recommended by Bundler Paper
            # - confidence: desired probability that the estimated matrix is correct
            F, inliers_mask = cv2.findFundamentalMat(pts1[matches[:, 0]], pts2[matches[:, 1]],
                                                     method=cv2.FM_RANSAC,
                                                     ransacReprojThreshold=ransacReprojThreshold, confidence=0.999)

            inliers_mask = inliers_mask.ravel() == 1

            inlier_pts1 = pts1[matches[inliers_mask, 0]]
            inlier_pts1 = inlier_pts1.astype(int)
            inlier_pts2 = pts2[matches[inliers_mask, 1]]
            inlier_pts2 = inlier_pts2.astype(int)

            # must have enough points to constrain the problem
            if inlier_pts1.shape[0] >= 8 and inlier_pts2.shape[0] >= 8 and F is not None:
                # optimize F according to the inliers using Levenberg-Marquardt
                F, _ = leastsq(func=F_matrix_residuals, x0=F.flatten()[:-1].reshape(-1, 1), args=(inlier_pts1, inlier_pts2))
                F = np.append(F, 1).reshape(3, 3)

                U, SIGMA, V_T = np.linalg.svd(F)
                SIGMA[2] = 0
                F = U @ np.diag(SIGMA) @ V_T

                # get inliers to optimized F
                for k, match in enumerate(matches):
                    pt1 = pts1[match[0]]
                    pt2 = pts2[match[1]]
                    dist = compute_F_matrix_residual(F, pt1, pt2)
                    if dist < ransacReprojThreshold:
                        inliers_mask[k] = 1
                    else:
                        inliers_mask[k] = 0

                self.gv_masks[i][j] = inliers_mask.ravel() == 1
                self.scene_graph[i, j] = np.sum(inliers_mask)
                # Scene graph is undirected so j,i and i,j have same weight
                self.scene_graph[j, i] = self.scene_graph[i, j]

                inlier_pts1 = pts1[matches[inliers_mask, 0]]
                inlier_pts1 = inlier_pts1.astype(int)
                inlier_pts2 = pts2[matches[inliers_mask, 1]]
                inlier_pts2 = inlier_pts2.astype(int)

                K1 = get_K_from_exif(exif_data1)
                K2 = get_K_from_exif(exif_data2)

                # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
                K1 = np.array([[2100.0000, 0.0000, 960.0000],
                               [0.0000, 2100.0000, 540.0000],
                               [0.0000, 0.0000, 1.0000]])
                K2 = np.array([[2100.0000, 0.0000, 960.0000],
                               [0.0000, 2100.0000, 540.0000],
                               [0.0000, 0.0000, 1.0000]])

                if inlier_pts1.shape[0] > 0 and inlier_pts2.shape[0] > 0:
                    # get E from exif data and F
                    self.im2im_configs[i][j] = get_best_configuration(K1, K2, F, inlier_pts1, inlier_pts2)

                if self.verbose and input("Visualize matches impl? (y/n) ") == 'y':
                    im1, im2 = self.images[i], self.images[j]
                    # self.visualize_matches(i, j)
                    # self.visualize_matches(i, j, inliers_mask.ravel())
                    # TODO incorrect epipolar lines?
                    draw_epipolar(im1, im2, F, inlier_pts1, inlier_pts2)
                    visualize_gv(K1, K2, F, inlier_pts1, inlier_pts2)

    def _geometric_verification(self):
        pickled_gv = os.path.join(self.geometric_verification_dir, 'geometric_verification.pkl')
        if os.path.exists(pickled_gv):
            with open(pickled_gv, 'rb') as f:
                self.scene_graph, self.im2im_configs, self.gv_masks = pickle.load(f)
        else:
            # a weighted adjacency list where an edge's weight is indicated
            # by the number of geometrically verified matches the two images share
            self.scene_graph = np.zeros((self.num_images, self.num_images))

            # R, t matrices between all image combinations
            # None when images don't share an edge
            self.im2im_configs = np.array([[None for _ in range(self.num_images)] for _ in range(self.num_images)])

            # geometrically verified matches stored as inlier masks
            self.gv_masks = np.array([[np.zeros(self.matches[i, j].shape) == 1 for j in range(self.num_images)] for i in range(self.num_images)])

            ij_combs = list(combinations(range(self.num_images), 2))
            Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._geometric_verification_impl)(i, j)
                    for (i, j) in tqdm(ij_combs, desc='Pairwise geometric verification'))

            with open(pickled_gv, 'wb') as f:
                pickle.dump((self.scene_graph, self.im2im_configs, self.gv_masks), f)

    def _find_homog_inlier_ratio(self, i, j):
        inlier_mask = self.gv_masks[i, j]
        matches = self.matches[i, j][inlier_mask]

        if len(matches) == 0:
            return np.inf

        kp1 = self.keypoints_and_descriptors[i][0]
        kp2 = self.keypoints_and_descriptors[j][0]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 0.4% of max dimension of image as described in "Modeling The World"
        threshold = 0.004 * max(*self.images[i].shape, *self.images[j].shape)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

        # use these mappings to find % of matches that are inliers to the homography
        return np.mean(mask)

    def _init_reconstruction_impl(self):
        ij_combs = list(combinations(range(self.num_images), 2))

        # find homography mappings between all image pairs
        homog_ratios = Parallel(n_jobs=1, backend='threading')(delayed(self._find_homog_inlier_ratio)(i, j)
                                                for i, j in tqdm(ij_combs, desc='Finding homography inlier ratios'))

        # finally find lowest % with at least 100 matches to start reconstruction from

        sorted_idxs = np.argsort(np.array(homog_ratios))

        for idx in sorted_idxs:
            i, j = ij_combs[idx]
            num_matches = np.sum(self.gv_masks[i, j])

            if num_matches >= self.init_threshold:
                return i, j

        # if we get here, it means there were no image pairs that had enough matches
        raise RuntimeError("No image pairs with 100 matches to start reconstruction")

    def _prune_tracks(self):
        consistent_tracks = [True for _ in self.tracks]
        for i, track in enumerate(self.tracks):
            kps = []
            imgs = []

            for kp, img in track:
                kps.append(kp)
                imgs.append(img)

            imgs_set = set(imgs)

            if len(imgs_set) != len(imgs):
                consistent_tracks[i] = False

        self.tracks = np.array(self.tracks)[consistent_tracks]

    def _build_tracks(self):
        self.track_adj_list = [[] for _ in range(self.num_images * self.n_keypoints)]

        for i in range(self.num_images):
            for j in range(self.num_images):
                if self.scene_graph[i, j]:
                    mask = self.gv_masks[i, j]
                    for match in self.matches[i, j][mask]:
                        kp1, kp2 = match.queryIdx, match.trainIdx

                        i_node = i * self.n_keypoints + kp1
                        j_node = j * self.n_keypoints + kp2

                        self.track_adj_list[i_node].append(j_node)
                        self.track_adj_list[j_node].append(i_node)

        self.track_graph = nx.Graph()
        for i in range(len(self.track_adj_list)):
            for j in self.track_adj_list[i]:
                self.track_graph.add_edge(i, j)

        # list of all the tracks where each track is a set of (image idx, keypoint idx) pairs
        self.tracks = list(nx.connected_components(self.track_graph))
        # converting the (img_num * n_keypoints + keypoint) to (img, keypoint)
        self.tracks = [set([(int(element / self.n_keypoints), element % self.n_keypoints) for element in track]) for track in self.tracks]
        self._prune_tracks()  # get rid of tracks which have more than one keypoint per image
        self.image_tracks = [set() for _ in range(self.num_images)]  # each image's keypoints' trackID (idx in self.tracks)
        for i, track in enumerate(self.tracks):
            for img, _ in track:
                self.image_tracks[img].add(i)
        warnings.warn('NotImplementedWarning')

    def _register_tracks(self, tracks):
        self.registered_tracks.update(tracks)
        # TODO do more here?

    def visualize_kp(self, i, kps):
        img = cv2.drawKeypoints(self.images[i], kps, None)
        plt.imshow(img), plt.show(block=False), plt.pause(0.001)

    def visualize_pcd(self, final=False):
        if final:
            pcd = self.pcd
            rc_pcd = open3d.PointCloud()
            rc_pcd.points = open3d.Vector3dVector(pcd)
            rc_pcd.paint_uniform_color([0, 0, 1])

            cams3dpose = np.array([-np.matrix(pose[0]).T * np.matrix(pose[1]) for pose in self.camera3Dpose if
                                   pose is not None]).squeeze()
            cam_pcd = open3d.PointCloud()
            cam_pcd.points = open3d.Vector3dVector(cams3dpose)
            cam_pcd.paint_uniform_color([1, 0, 0])

            open3d.draw_geometries([rc_pcd, cam_pcd])
        else:
            pcd = self.pcd[:-(self.num_points_added_last_itr + 1)]
            rc_pcd = open3d.PointCloud()
            rc_pcd.points = open3d.Vector3dVector(pcd)
            rc_pcd.paint_uniform_color([0, 0, 1])

            pcd2 = self.pcd[-self.num_points_added_last_itr:]
            rc2_pcd = open3d.PointCloud()
            rc2_pcd.points = open3d.Vector3dVector(pcd2)
            rc2_pcd.paint_uniform_color([0, 1, 0])

            cams3dpose = np.array([-np.matrix(pose[0]).T * np.matrix(pose[1]) for pose in self.camera3Dpose if pose is not None]).squeeze()
            cam_pcd = open3d.PointCloud()
            cam_pcd.points = open3d.Vector3dVector(cams3dpose)
            cam_pcd.paint_uniform_color([1, 0, 0])

            cams3dpose2 = np.array(-np.matrix(self.camera3Dpose[self.last_cam_added][0]).T * np.matrix(self.camera3Dpose[self.last_cam_added][1])).squeeze()
            cam_pcd2 = open3d.PointCloud()
            cam_pcd2.points = open3d.Vector3dVector([cams3dpose2])
            cam_pcd2.paint_uniform_color([1, 0.706, 0])
            open3d.draw_geometries([rc_pcd, rc2_pcd, cam_pcd2, cam_pcd])
            #open3d.draw_geometries_with_animation_callback([rc_pcd, rc2_pcd, cam_pcd2, cam_pcd], rotate_view)

            # vis = open3d.Visualizer()
            # vis.create_window()
            # vis.draw_geometry([rc_pcd, rc2_pcd, cam_pcd2, cam_pcd])
            # vis.register_animation_callback(rotate_view)
            # vis.run()
            # vis.destroy_window()

    def _get_track_kp_between_images(self, shared_tracks, i, j):
        pts1 = []
        pts2 = []

        # find both images' keypoints in each track
        for trackID in shared_tracks:
            trackDict = dict(self.tracks[trackID])
            kp1 = trackDict[i]
            kp2 = trackDict[j]
            pts1.append(self.keypoints_and_descriptors[i][0][kp1].pt)
            pts2.append(self.keypoints_and_descriptors[j][0][kp2].pt)

        return np.array(pts1), np.array(pts2)

    def _triangulate_tracks(self, tracks, i, j):
        # TODO incorporate tracks, currently triangulating just matches between i and j which is incorrect

        return

    def _init_reconstruction(self):
        # prune pairs that don't have enough verified matches
        self.scene_graph[self.scene_graph < self.gv_threshold] = 0
        self._build_tracks()

        pickled_rc_init = os.path.join(self.reconstruction_dir, 'init.pkl')
        if os.path.exists(pickled_rc_init):
            with open(pickled_rc_init, 'rb') as f:
                init_imgs = pickle.load(f)
        else:
            init_imgs = self._init_reconstruction_impl()

            with open(pickled_rc_init, 'wb') as f:
                pickle.dump(init_imgs, f)

        i, j = init_imgs

        self.unregistered_imgs = set(range(self.num_images))
        self.registered_tracks = set()
        self.track2pcd = {}  # trackID to index in self.pcd
        self.camera3Dpose = [None for _ in range(self.num_images)]

        # register i, j together
        self.unregistered_imgs.remove(i), self.unregistered_imgs.remove(j)
        i_tracks = self.image_tracks[i]
        j_tracks = self.image_tracks[j]
        shared_tracks = i_tracks.intersection(j_tracks)
        self._register_tracks(shared_tracks)

        # Initialize point cloud data
        K1, K2 = get_K_from_exif(self.exif_data[i]), get_K_from_exif(self.exif_data[j])
        R2, t2 = self.im2im_configs[i][j]
        R1, t1 = np.diag(np.ones(R2.shape[1])), np.zeros(t2.shape)  # camera i is the origin of this coordinate system
        pts1, pts2 = self._get_track_kp_between_images(shared_tracks, i, j)

        # TODO add a way to associate 3D pts with the cameras that observe it
        self.pcd = findPointCloud(K1, K2, R1, t1, R2, t2, pts1, pts2)[:3].T
        self._bundle_adjustment()  # perform two frame bundle adjustment

        i_pts2d = []
        j_pts2d = []
        pts3d = []
        for pt3d_idx, trackID in enumerate(shared_tracks):  # associating each trackID to its index in self.pcd
            self.track2pcd[trackID] = pt3d_idx

            trackDict = dict(self.tracks[trackID])
            kp1 = trackDict[i]
            kp2 = trackDict[j]
            i_pts2d.append(self.keypoints_and_descriptors[i][0][kp1].pt)
            j_pts2d.append(self.keypoints_and_descriptors[j][0][kp2].pt)
            pts3d.append(self.pcd[pt3d_idx])

        # solve for camera positions in 3D space
        self.camera3Dpose[i] = get_camera_pose(i_pts2d, pts3d, K1)
        self.camera3Dpose[j] = get_camera_pose(j_pts2d, pts3d, K2)

        if self.verbose and input("Visualize Scene Graph? (y/n) ") == 'y':
            G = nx.from_numpy_matrix(np.array(self.scene_graph))
            nx.draw(G)
            plt.show()

        if self.verbose and input("Visualize initialization images? (y/n) ") == 'y':
            inliers_mask = self.gv_masks[i][j]
            self.visualize_matches(i, j, inliers_mask)
        self.num_points_added_last_itr = self.pcd.shape[0]
        self.last_cam_added = i
        self.visualize_pcd()

    def _register_next_img(self):
        """
        Registers an image into the 3D reconstruction by adding its unregistered points into the point cloud
        """

        num_points_start = self.pcd.shape[0]

        img_registered_tracks = [(i, self.image_tracks[i].intersection(self.registered_tracks)) for i in self.unregistered_imgs]
        i, i_reg_tracks = sorted(img_registered_tracks, key=lambda x: len(x[1]), reverse=True)[0]
        # if there are no images left which observe any registered tracks
        if i_reg_tracks == set():
            self.unregistered_imgs.remove(i)
            return
        candidate_tracks = self.image_tracks[i].difference(i_reg_tracks)  # tracks observed by i that aren't in the reconstruction yet

        i_pts2d = []
        pts3d = []
        for trackID in i_reg_tracks:
            pt3d_idx = self.track2pcd[trackID]
            trackDict = dict(self.tracks[trackID])
            kp1 = trackDict[i]
            i_pts2d.append(self.keypoints_and_descriptors[i][0][kp1].pt)
            pts3d.append(self.pcd[pt3d_idx])

        if len(pts3d) < 4:  # need 4 points to use solvePnP
            self.unregistered_imgs.remove(i)
            return

        K1 = get_K_from_exif(self.exif_data[i])
        # solve for camera positions in 3D space TODO maybe refine later
        R1, t1 = get_camera_pose(i_pts2d, pts3d, K1)
        i_cam_pt3d = -R1.T @ t1
        self.camera3Dpose[i] = R1, t1

        self.last_cam_added = i
        for trackID in candidate_tracks:
            # add candidate tracks if they are observed by at least one registered camera
            # and if adding it produces a well-conditioned estimate of its location
            trackDict = dict(self.tracks[trackID])
            kp1 = trackDict[i]
            pt1 = self.keypoints_and_descriptors[i][0][kp1].pt

            best_pt3d = None
            best_dos = 0
            for j, kp2 in trackDict.items():
                i_node = i * self.n_keypoints + kp1
                j_node = j * self.n_keypoints + kp2
                # if j is registered and is adjacent to i
                if j not in self.unregistered_imgs and j_node in self.track_adj_list[i_node]:
                    K2 = get_K_from_exif(self.exif_data[j])
                    R2, t2 = self.camera3Dpose[j]
                    pt2 = self.keypoints_and_descriptors[j][0][kp2].pt
                    pt3d = findPointCloud(K1, K2, R1, t1, R2, t2, np.array([pt1]), np.array([pt2]))

                    if pt3d.size > 0:
                        rotM, tvec = self.camera3Dpose[j]
                        j_cam_pt3d = -rotM.T @ tvec

                        dos = get_angle(np.array(i_cam_pt3d), np.array(pt3d), np.array(j_cam_pt3d))

                        if dos > best_dos:
                            best_pt3d = pt3d.reshape(1, -1)
                            best_dos = dos

            # add best triangulation
            if best_dos > self.dos_threshold:
                self.pcd = np.append(self.pcd, best_pt3d, axis=0)
                self.track2pcd[trackID] = self.pcd.shape[0] - 1  # maps to point we just added
                self._register_tracks([trackID])

        num_points_end = self.pcd.shape[0]

        self.num_points_added_last_itr = num_points_end - num_points_start
        # TODO remap every image's observed tracks with the new added pts

        self.unregistered_imgs.remove(i)  # finally, remove this image from the set of unregistered images

    def _bundle_adjustment(self):
        warnings.warn('NotImplementedWarning')

    def _reconstruct3d(self):
        while self.unregistered_imgs != set():
            self._register_next_img()
            #self.visualize_pcd()

    def evaluate(self):
        raise NotImplementedError


if __name__ == '__main__':
    with warnings.catch_warnings():  # TODO how to not display deprecation warnings?
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        pipeline = Pipeline('../datasets/Statue/images/',
                            n_keypoints=8000, verbose=False,
                            n_jobs=-1, init_threshold=100)
        pipeline.run()
        pipeline.visualize_pcd(final=True)
        pipeline.evaluate()
