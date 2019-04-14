import os
from sklearn.externals.joblib import Parallel, delayed
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

NUM_KEYPOINTS = 100


# https://isotope11.com/blog/storing-surf-sift-orb-keypoints-using-opencv-in-python
def serialize_keypoints(keypoints, descriptors):
    temp_array = []
    for kp, des in zip(keypoints, descriptors):
        temp = (kp.pt, kp.size, kp.angle, kp.response, kp.octave,
                kp.class_id, des)
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


def serialize_matches(matches):
    temp_array = []
    for match in matches:
        temp = (match.distance, match.imgIdx, match.queryIdx, match.trainIdx)
        temp_array.append(temp)
    return temp_array


def deserialize_matches(matches):
    temp_array = []
    for match in matches:
        if len(match) > 0:
            temp = cv2.DMatch(_distance=match[0], _imgIdx=match[1],
                              _queryIdx=match[2], _trainIdx=match[3])
        else:
            temp = None
        temp_array.append(temp)
    return temp_array


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
    # find the keypoints and descriptors with SIFT
    kps_and_des = Parallel(n_jobs=4)(delayed(pickleable_detect_and_compute)(img) for img in
                            tqdm(images, desc='Extracting features and descriptors'))


    # sift = cv2.xfeatures2d.SIFT_create(NUM_KEYPOINTS)
    # kps_and_des = []
    # for img in tqdm(images, desc='Extracting features and descriptors'):
    #     kps_and_des.append(pickleable_detect_and_compute(img))

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
