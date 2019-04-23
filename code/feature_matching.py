"""
File containing all the helper functions for feature matching
"""

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
