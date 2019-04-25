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
import time
from itertools import combinations
import numpy as np
import networkx as nx

from feature_extraction import get_human_readable_exif, populate_keypoints_and_descriptors, deserialize_keypoints
from feature_matching import serialize_matches, deserialize_matches
from geometric_verification import draw_epipolar, get_K_from_exif, visualize_gv, get_best_configuration


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
        self._extract_features()  # extract features using SIFT
        self._match_features()  # match features using FLANN (TODO decide)
        self._geometric_verification()
        self._init_reconstruction()
        self._reconstruct3d()
        raise NotImplementedError

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
        self.keypoints_and_descriptors = np.array([deserialize_keypoints(kps_and_des) for kps_and_des in self.keypoints_and_descriptors])

        # TODO add debug option to visualize
        if self.verbose and input("Visualize keypoints? (y/n) ") == 'y':
            img = self.images[0]
            vis_keypoints = self.keypoints_and_descriptors[0][0]
            vis_img = cv2.drawKeypoints(img, vis_keypoints, img, flags=4)  # draws rich keypoints
            plt.imshow(vis_img), plt.show()

    def _match_features_impl(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # TODO set params differently?
        search_params = dict(checks=50)  # or pass empty dictionary

        matcher = cv2.FlannBasedMatcher(index_params, search_params)  # TODO use cv2.BFMatcher() ?

        # TODO upper triangular matrix of matches between all combinations of images
        image_matches = []
        been_matched = set()  # for keeping track of what has already been fully matched

        for i, kp_and_des1 in tqdm(enumerate(self.keypoints_and_descriptors), desc='Matching image features'):
            been_matched.add(i)
            i_matches = []
            for j, kp_and_des2 in enumerate(self.keypoints_and_descriptors):
                # TODO save homography here? or in geometric verification?
                i_with_j_matches = []
                if j not in been_matched:
                    kp1, des1 = kp_and_des1
                    kp2, des2 = kp_and_des2

                    matches = matcher.knnMatch(des1, des2, k=2)

                    # Need to draw only good matches, so create a mask
                    matchesMask = [[1, 0] for _ in range(len(matches))]

                    i_with_j_matches = []
                    for match, matchMask in zip(matches, matchesMask):
                        if matchMask[0]:
                            i_with_j_matches.append(match[0])

                    # TODO add debug option to visualize
                    if self.verbose and input("Visualize matches impl? (y/n) ") == 'y':
                        img1 = self.images[i]
                        img2 = self.images[j]

                        draw_params = dict(matchColor=(0, 255, 0),
                                           singlePointColor=(255, 0, 0),
                                           matchesMask=matchesMask,
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
                        plt.imshow(img3), plt.show()

                i_matches.append(serialize_matches(i_with_j_matches))
            image_matches.append(i_matches)
        return image_matches

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
            kps_i_as_np = np.array(self.keypoints_and_descriptors[i][0])[mask == 1]
            kps_j_as_np = np.array(self.keypoints_and_descriptors[j][0])[mask == 1]
            img = cv2.drawMatches(self.images[i], self.keypoints_and_descriptors[i][0],
                                  self.images[j], self.keypoints_and_descriptors[i][0],
                                  matches_as_np, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(img), plt.show()

    def _match_features(self):
        # TODO decide between saving all matches in one go or incrementally for each image's matches
        pickled_matches = os.path.join(self.feature_matching_dir, 'matches.pkl')
        if os.path.exists(pickled_matches):
            with open(pickled_matches, 'rb') as f:
                self.matches = pickle.load(f)
        else:
            self.matches = self._match_features_impl()
            with open(pickled_matches, 'wb') as f:
                pickle.dump(self.matches, f)

        self.matches = [[deserialize_matches(ij_matches) for ij_matches in i_matches] for i_matches in self.matches]
        self.matches = np.array([[sorted(ij_matches, key=lambda match: match.distance)
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
        pts1 = np.array([self.keypoints_and_descriptors[i][0][match.queryIdx].pt for match in self.matches[i][j]])
        pts2 = np.array([self.keypoints_and_descriptors[j][0][match.trainIdx].pt for match in self.matches[i][j]])

        # TODO
        exif_data1 = self.exif_data[i]
        exif_data2 = self.exif_data[j]

        height, width = exif_data1['ExifImageHeight'], exif_data1['ExifImageWidth']

        # Args:
        # - ransacReprojThreshold: threshold for inliers
        #                            1-3 recommended by OpenCV documentation
        #                            0.006*max(img dimensions) recommended by Bundler Paper
        # - confidence: desired probability that the estimated matrix is correct
        F, inliers_mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC,
                                                 ransacReprojThreshold=0.001*max(height, width), confidence=0.999)

        # TODO extra RANSAC stage to make sure F is correct and not prone to outliers
        self.gv_masks[i][j] = inliers_mask.ravel() == 1
        self.scene_graph[i, j] = np.sum(inliers_mask)
        # Scene graph is undirected so j,i and i,j have same weight
        self.scene_graph[j, i] = self.scene_graph[i, j]

        gv_pts1 = pts1[inliers_mask.ravel() == 1].astype(int)
        gv_pts2 = pts2[inliers_mask.ravel() == 1].astype(int)

        K1 = get_K_from_exif(exif_data1)
        K2 = get_K_from_exif(exif_data2)

        # TODO get E from exif data and F
        self.im2im_configs[i][j] = get_best_configuration(K1, K2, F, gv_pts1, gv_pts2)

        # TODO add a debug option for visualization
        if self.verbose and input("Visualize matches impl? (y/n) ") == 'y':
            im1, im2 = self.images[i], self.images[j]
            # self.visualize_matches(i, j)
            # self.visualize_matches(i, j, inliers_mask.ravel())
            # TODO incorrect epipolar lines?
            draw_epipolar(im1, im2, F, gv_pts1, gv_pts2)

            visualize_gv(K1, K2, F, gv_pts1, gv_pts2)

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
            self.gv_masks = np.array([[None for _ in range(self.num_images)] for _ in range(self.num_images)])

            ij_combs = list(combinations(range(self.num_images), 2))
            Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._geometric_verification_impl)(i, j)
                    for (i, j) in tqdm(ij_combs, desc='Pairwise geometric verification'))

            with open(pickled_gv, 'wb') as f:
                pickle.dump((self.scene_graph, self.im2im_configs, self.gv_masks), f)

    def _find_homog_inlier_ratio(self, i, j):
        inlier_mask = self.gv_masks[i, j]
        matches = self.matches[i, j][inlier_mask]
        kp1 = self.keypoints_and_descriptors[i, 0]
        kp2 = self.keypoints_and_descriptors[j, 0]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 0.4% of max dimension of image as described in "Modeling The World"
        threshold = 0.004 * max(self.images[i].shape, self.images[j].shape)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
        matchesMask = mask.ravel().tolist()
        # use these mappings to find % of matches that are inliers to the homography

        raise NotImplementedError

    def _init_reconstruction_impl(self):
        ij_combs = list(combinations(range(self.num_images), 2))

        # find homography mappings between all image pairs
        homog_ratios = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(self._find_homog_inlier_ratio)(i, j)
                                                for i, j in tqdm(ij_combs, desc='Finding homography inlier ratios'))

        # finally find lowest % with at least 100 matches to start reconstruction from

        sorted_idxs = np.argsort(np.array(homog_ratios))

        for i, j in sorted_idxs:
            num_matches = np.sum(self.gv_masks[i, j])

            if num_matches >= self.init_threshold:
                return i, j

        # if we get here, it means there were no image pairs that had enough matches
        raise RuntimeError("No matches with 100 matches to start reconstruction")

    def _init_reconstruction(self):
        pickled_rc_init = os.path.join(self.reconstruction_dir, 'init.pkl')
        if os.path.exists(pickled_rc_init):
            with open(pickled_rc_init, 'rb') as f:
                init = pickle.load(f)
        else:
            init = self._init_reconstruction_impl()

            with open(pickled_rc_init, 'wb') as f:
                pickle.dump(init, f)

        raise NotImplementedError

    def _init_reconstruction_old(self):
        # TODO old implementation of initializing
        # prune pairs that don't have enough verified matches
        self.scene_graph[self.scene_graph < self.gv_threshold] = 0

        if self.verbose and input("Visualize Scene Graph? (y/n) ") == 'y':
            G = nx.from_numpy_matrix(np.array(self.scene_graph))
            nx.draw(G)
            plt.show()

        # start at the image which has the highest weighted edges
        # pick it and the image it shares the highest weighted edge with as the first images to register
        i = np.argmax(np.sum(self.scene_graph, axis=1))
        j = np.argmax(self.scene_graph[i])

        i, j = sorted([i, j])

        # TODO register i, j together
        # TODO visualize i, j in 3D
        # TODO add a debug option for visualization
        if self.verbose and input("Visualize initialization images? (y/n) ") == 'y':
            K1 = get_K_from_exif(self.exif_data[i])
            K2 = get_K_from_exif(self.exif_data[j])
            R, t = self.im2im_configs[i][j]
            inliers_mask = self.gv_masks[i][j]
            kps1 = np.array(self.keypoints_and_descriptors[i][0])[inliers_mask]
            kps2 = np.array(self.keypoints_and_descriptors[j][0])[inliers_mask]
            pts1 = np.array([kp.pt for kp in kps1])
            pts2 = np.array([kp.pt for kp in kps2])
            visualize_gv(K1, K2, R, t, pts1, pts2)

    def _register_img(self, i):
        """
        Registers an image into the 3D reconstruction by adding its unregistered points into the point cloud
        :param i: index of image i
        :return:
        """
        raise NotImplementedError

    def _reconstruct3d(self):
        self.point_cloud = []  # list of 3d points in the scene
        self.feature_in_point_cloud = [[]]  #

        raise NotImplementedError


if __name__ == '__main__':
    with warnings.catch_warnings():  # TODO how to not display dep warnings?
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        pipeline = Pipeline('../datasets/Bicycle/images/', n_keypoints=100, verbose=False, n_jobs=-1)
        pipeline.run()
