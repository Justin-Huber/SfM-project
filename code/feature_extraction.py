import os
from sklearn.externals.joblib import Parallel, delayed
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

def populate_keypoints_and_descriptors(images):
    """

    :param images: list of images as numpy arrays
    :return: returns a list of keypoints and descriptors where the
            index is the index of that image in the images list
    """
    # Initiate SIFT detector
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kps_and_des = Parallel(n_jobs=16)(delayed(sift.detectAndCompute)(img, None) for img in
                        tqdm(images, desc='Extracting features and descriptors'))

    # TODO add debug option to visualize
    return kps_and_des


def load_images(images_dir):
    """

    :return: returns list of images in the images directory
            in the same order they appear in the directory
    """
    if not os.path.exists(images_dir):
        raise RuntimeError("Invalid image directory")

    img_filenames = os.listdir(images_dir)
    images = Parallel(n_jobs=16)(delayed(cv2.imread)(os.path.join(images_dir, img_filename), 0) for img_filename in
                                   tqdm(img_filenames, desc='Loading images'))

    # TODO add debug option to visualize
    print(len(images))
    plt.imshow(images[85]), plt.show()

    return images
