import cv2


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



######### TODO need below? ########
import os
import numpy as np
import math
import matplotlib.pyplot as plt

def read_img(path):
    #read image in grayscale (the 0 lfag indicates grayscale)
    return cv2.imread(path, 0)


def extract_and_match_draw(gray1, gray2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1,None)
    kp2, des2 = orb.detectAndCompute(gray2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    cropped_matches = []
    for match in matches:
        if match.distance < 26:
            cropped_matches.append(match)
        else:
            break

    imgDebug = cv2.drawMatches(gray1,kp1,gray2,kp2,cropped_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imgDebug),plt.show()

    return cropped_matches


def extract_and_match(gray1, gray2):
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1,None)
    kp2, des2 = orb.detectAndCompute(gray2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    cropped_matches = []
    for match in matches:
        if match.distance < 26:
            cropped_matches.append(match)
        else:
            break

    return cropped_matches
