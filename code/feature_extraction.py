import os
from sklearn.externals.joblib import Parallel, delayed
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

NUM_KEYPOINTS = 100 # TODO 100, 1 just for testing


# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def serialize_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        ++i
        temp_array.append(temp)
    return temp_array

# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def deserialize_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1],
                                    _size=point[1], _angle=point[2],
                                    _response=point[3], _octave=point[4],
                                    _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


def pickleable_detect_and_compute(img):
    """
    Wrapper function for sift.detectAndCompute which uses pickleable keypoints
    by serializing the keypoints
    :param sift: sift object
    :param img: np array
    :return: returns a tuple of the serialized keypoint and descriptor
    """
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(NUM_KEYPOINTS)
    kp, des = sift.detectAndCompute(img, None)
    return serialize_keypoints(kp, des)


def populate_keypoints_and_descriptors(images):
    """

    :param images: list of images as numpy arrays
    :return: returns a list of serialized keypoints and descriptors where the
            index is the index of that image in the images list
    """
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(NUM_KEYPOINTS)
    # find the keypoints and descriptors with SIFT
    kps_and_des = Parallel(n_jobs=16)(delayed(pickleable_detect_and_compute)(img) for img in
                            tqdm(images, desc='Extracting features and descriptors'))

    # kps_and_des = []
    # for img in tqdm(images, desc='Extracting features and descriptors'):
    #     kps_and_des.append(sift.detectAndCompute(img, None))

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
    plt.imshow(images[0]), plt.show()

    return images
