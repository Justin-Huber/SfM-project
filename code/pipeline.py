import os
import warnings
import pickle
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.externals.joblib import Parallel, delayed
import time
from itertools import combinations
import numpy as np

# import globals
from feature_extraction import get_human_readable_exif, populate_keypoints_and_descriptors, deserialize_keypoints
from feature_matching import serialize_matches, deserialize_matches
from geometric_verification import draw_epipolar


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
        self._init_pipeline_file_structure()
        self.verbose = kwargs.pop('verbose', False)

        # check valid image dir
        self.images = []

    def _init_pipeline_file_structure(self):
        if not os.path.isdir(self.output_dir):
            raise RuntimeError("Invalid output directory")

        self.pipeline_dir = os.path.join(self.output_dir, 'pipeline')
        self.feature_extraction_dir = os.path.join(self.pipeline_dir, 'feature_extraction')
        self.feature_matching_dir = os.path.join(self.pipeline_dir, 'feature_matching')

        if not os.path.exists(self.pipeline_dir):
            os.mkdir(self.pipeline_dir)
        if not os.path.exists(self.feature_extraction_dir):
            os.mkdir(self.feature_extraction_dir)
        if not os.path.exists(self.feature_matching_dir):
            os.mkdir(self.feature_matching_dir)

    def run(self, stage=0):
        if stage == 0:
            self._extract_features()  # extract features using SIFT
        if stage < 1:
            self._match_features()  # match features using FLANN (TODO decide)
        if stage < 2:
            self._geometric_verification()
        if stage < 3:
            self._init_reconstruction()
        if stage < 4:
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

        images = Parallel(n_jobs=-1, backend='threading')(delayed(cv2.imread)(os.path.join(self.images_dir, img_filename), 0)
                                                            for img_filename in tqdm(img_filenames, desc='Loading images'))

        # TODO add debug option to visualize
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

    def _extract_features(self):
        self.images, self.exif_data = self._load_images()
        self.num_images = len(self.images)

        pickled_keypoints = os.path.join(self.feature_extraction_dir, 'keypoints.pkl')
        if os.path.exists(pickled_keypoints):
            with open(pickled_keypoints, 'rb') as f:
                self.keypoints_and_descriptors = pickle.load(f)
        else:
            self.keypoints_and_descriptors = populate_keypoints_and_descriptors(self.images)
            with open(pickled_keypoints, 'wb') as f:
                pickle.dump(self.keypoints_and_descriptors, f)

        # deserialize the keypoints
        self.keypoints_and_descriptors = [deserialize_keypoints(kps_and_des) for kps_and_des in self.keypoints_and_descriptors]

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

        matcher = cv2.FlannBasedMatcher(index_params, search_params)  # cv2.BFMatcher()

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
                    matchesMask = [[0, 0] for _ in range(len(matches))]

                    # ratio test as per Lowe's paper
                    for k, (m, n) in enumerate(matches):
                        if m.distance < 0.75 * n.distance:  # TODO do we still want to filter out these? e.g. for bike, many useful points map to many others so they won't be included
                            matchesMask[k] = [1, 0]

                    i_with_j_matches = []
                    for match, matchMask in zip(matches, matchesMask):
                        if matchMask[0]:
                            i_with_j_matches.append(match[0])

                    if self.verbose and input("Visualize matches impl? (y/n) ") == 'y':
                        # TODO add debug option to visualize
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
        self.matches = [[sorted(ij_matches, key=lambda match: match.distance) for ij_matches in i_matches] for i_matches in self.matches]

        if self.verbose and input("Visualize matches? (y/n) ") == 'y':
            for i in range(self.num_images):
                for j in range(self.num_images):
                    if len(self.matches[i][j]) > 0:
                        img3 = cv2.drawMatches(self.images[i], self.keypoints_and_descriptors[i][0],
                                                self.images[j], self.keypoints_and_descriptors[j][0],
                                                self.matches[i][j], None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                        plt.imshow(img3), plt.show()
            # TODO add visualization

    def _geometric_verification_impl(self, i, j):
        # TODO multithreading accessing a list, how does it work?
        # TODO check that GIL isn't preventing parallelism
        pts1 = np.array([self.keypoints_and_descriptors[i][0][match.trainIdx].pt for match in self.matches[i][j]])
        pts2 = np.array([self.keypoints_and_descriptors[j][0][match.queryIdx].pt for match in self.matches[i][j]])

        # TODO do our own implementation
        F, inliers_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        im1, im2 = self.images[i], self.images[j]
        gv_pts1 = pts1[inliers_mask.ravel() == 1].astype(int)
        gv_pts2 = pts2[inliers_mask.ravel() == 1].astype(int)
        print(i, j)
        # TODO incorrect epipolar lines?
        draw_epipolar(im1, im2, F, pts1.astype(int), pts2.astype(int))
        draw_epipolar(im1, im2, F, gv_pts1, gv_pts2)


        # TODO add a debug option for visualization
        # TODO get E from exif data and F

    def _geometric_verification(self):
        # a weighted adjacency list where an edge's weight is indicated
        # by the number of geometrically verified matches the two images share
        self.scene_graph = [[0 for _ in range(self.num_images)] for _ in range(self.num_images)]

        # E matrices between all image combinations
        # None when images don't share an edge
        self.essential_matrices = [[None for _ in range(self.num_images)] for _ in range(self.num_images)]

        # geometrically verified matches TODO mask?
        self.gv_matches = [[None for _ in range(self.num_images)] for _ in range(self.num_images)]

        # TODO old implementation
        # ij_combs = [(i, j) for i in range(self.num_images) for j in range(self.num_images)]
        # ij_combs = list(set([tuple(sorted(list(tup))) for tup in ij_combs]))
        ij_combs = list(combinations(range(self.num_images), 2))
        Parallel(n_jobs=1, backend='threading')(delayed(self._geometric_verification_impl)(i, j)
                for (i, j) in tqdm(ij_combs, desc='Pairwise geometric verification'))

        raise NotImplementedError

    def _init_reconstruction(self):
        # start at the image which has the highest weighted edges
        # pick it and the image it shares the highest weighted edge with as the first images to register
        raise NotImplementedError

    def _reconstruct3d(self):
        raise NotImplementedError


if __name__ == '__main__':
    # TODO may not be necessary
    # globals.init('../datasets/Bicycle/images/')

    with warnings.catch_warnings():  # TODO how to not display dep warnings?
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        pipeline = Pipeline('../datasets/Bicycle/images/', verbose=True)
        pipeline.run()

        # globals.pipeline.run()
