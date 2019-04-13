import os
import pickle
import matplotlib.pyplot as plt
import cv2

from feature_extraction import load_images, populate_keypoints_and_descriptors,\
                                deserialize_keypoints


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
            self._init_reconstruction()
        if stage < 3:
            self._reconstruct3d()
        raise NotImplementedError

    def _load_images(self):
        pickled_images = os.path.join(self.feature_extraction_dir, 'images.pkl')
        if os.path.exists(pickled_images):
            with open(pickled_images, 'rb') as f:
                images = pickle.load(f)
        else:
            images = load_images(self.images_dir)
            with open(pickled_images, 'wb') as f:
                pickle.dump(images, f)

        return images

    def _extract_features(self):
        images = self._load_images()  # TODO move to the else?
        pickled_keypoints = os.path.join(self.feature_extraction_dir, 'keypoints.pkl')
        if os.path.exists(pickled_keypoints):
            with open(pickled_keypoints, 'rb') as f:
                self.keypoints_and_descriptors = pickle.load(f)
        else:
            self.keypoints_and_descriptors = populate_keypoints_and_descriptors(images)
            with open(pickled_keypoints, 'wb') as f:
                pickle.dump(self.keypoints_and_descriptors, f)

        # deserialize the keypoints
        self.keypoints_and_descriptors = [deserialize_keypoints(kps_and_des) for kps_and_des in self.keypoints_and_descriptors]

        # TODO add debug option to visualize
        img = images[0]
        vis_keypoints = self.keypoints_and_descriptors[0][0]
        vis_img = cv2.drawKeypoints(img, vis_keypoints, img, flags=4)  # draws rich keypoints
        plt.imshow(vis_img), plt.show()

    def _match_features(self):
        raise NotImplementedError  # TODO below implementation not done yet
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # TODO set params differently?
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        plt.imshow(img3, ), plt.show()
        raise NotImplementedError

    def _init_reconstruction(self):
        raise NotImplementedError

    def _reconstruct3d(self):
        raise NotImplementedError


if __name__ == '__main__':
    pipeline = Pipeline('../datasets/Bicycle/images/')
    pipeline.run()
