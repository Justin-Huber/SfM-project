"""
File containing all the helper functions for feature extraction
"""

from sklearn.externals.joblib import Parallel, delayed
import cv2
try:  # For using tqdm on Google Colab
  import google.colab
  IN_COLAB = True
  from tqdm import tqdm_notebook as tqdm
except:
  IN_COLAB = False
  from tqdm import tqdm
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def get_human_readable_exif(filename):
    """

    :param exif_data: dict of nonsense numbers to values
    :return: a dict with tags switched out for their human readable versions
    """

    exif_data = Image.open(filename)._getexif()
    return {TAGS[t]: v for (t, v) in exif_data.items()}


def serialize_keypoints(keypoints, descriptors):
    """
    # https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python

    :param keypoints:
    :param descriptors:
    :return:
    """
    temp_array = []
    for kp, des in zip(keypoints, descriptors):
        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave,
                kp.class_id, des)
        temp_array.append(temp)
    return temp_array


def deserialize_keypoints(array):
    """
    # https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python

    :param array:
    :return:
    """
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


def pickleable_detect_and_compute(img, n_keypoints):
    """
    Wrapper function for sift.detectAndCompute which uses pickleable keypoints
    by serializing the keypoints
    :param sift: sift object
    :param img: np array
    :return: returns a tuple of the serialized keypoint and descriptor
    """
    # Initiate SIFT detector
    detector = cv2.xfeatures2d.SIFT_create(n_keypoints)
    kp, des = detector.detectAndCompute(img, None)
    return serialize_keypoints(kp, des)


def populate_keypoints_and_descriptors(images, n_keypoints, n_jobs):
    """

    :param images: list of images as numpy arrays
    :return: returns a list of serialized keypoints and descriptors where the
            index is the index of that image in the images list
    """
    # Initiate SIFT detector
    # find the keypoints and descriptors with SIFT
    kps_and_des = Parallel(n_jobs=n_jobs, backend='threading')(delayed(pickleable_detect_and_compute)(img, n_keypoints)
                                        for img in tqdm(images, desc='Extracting features and descriptors'))
    # TODO add debug option to visualize
    return kps_and_des
