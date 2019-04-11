import os
import pickle

from feature_extraction import load_images, populate_keypoints_and_descriptors


class Pipeline():
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

        p_dir = os.path.join(self.output_dir, 'pipeline')
        fe_dir = os.path.join(p_dir, 'feature_extraction')
        fm_dir = os.path.join(p_dir, 'feature_matching')

        if not os.path.exists(p_dir):
            os.mkdir(p_dir)
        if not os.path.exists(fe_dir):
            os.mkdir(fe_dir)
        if not os.path.exists(fm_dir):
            os.mkdir(fm_dir)

    def run(self, stage=0):
        if stage == 0:
            self._extract_features()  # extract features using SIFT
        if stage < 1:
            self._match_features()  # match features using ANN (TODO decide)
        if stage < 2:
            self._init_reconstruction()
        if stage < 3:
            self._reconstruct3d()
        raise NotImplementedError

    def _extract_features(self):
        images = load_images(self.images_dir)
        self.features_and_descriptors = populate_keypoints_and_descriptors(images)
        raise NotImplementedError

    def _match_features(self):
        raise NotImplementedError

    def _init_reconstruction(self):
        raise NotImplementedError

    def _reconstruct3d(self):
        raise NotImplementedError


if __name__ == '__main__':
    pipeline = Pipeline('../datasets/Bicycle/images/')
    pipeline.run()
