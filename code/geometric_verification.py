import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import linalg
import operator


#.....................................RANSAC..........................................
def get_h(putative_matches, kp1, kp2, samples):

    p1_1 = kp1[samples[0][0]].pt
    p1_2 = kp2[samples[0][1]].pt
    p2_1 = kp1[samples[1][0]].pt
    p2_2 = kp2[samples[1][1]].pt
    p3_1 = kp1[samples[2][0]].pt
    p3_2 = kp2[samples[2][1]].pt
    p4_1 = kp1[samples[3][0]].pt
    p4_2 = kp2[samples[3][1]].pt

    A = np.array([[p1_1[0], p1_1[1], 1,0,0,0, -p1_2[0]*p1_1[0], -p1_2[0]*p1_1[1], -p1_2[0]],
                    [0,0,0, p1_1[0], p1_1[1], 1, -p1_2[1]*p1_1[0], -p1_2[1]*p1_1[1], -p1_2[1]],
                    [p2_1[0], p2_1[1], 1,0,0,0, -p2_2[0]*p2_1[0], -p2_2[0]*p2_1[1], -p2_2[0]],
                    [0,0,0, p2_1[0], p2_1[1], 1, -p2_2[1]*p2_1[0], -p2_2[1]*p2_1[1], -p2_2[1]],
                    [p3_1[0], p3_1[1], 1,0,0,0, -p3_2[0]*p3_1[0], -p3_2[0]*p3_1[1], -p3_2[0]],
                    [0,0,0, p3_1[0], p3_1[1], 1, -p3_2[1]*p3_1[0], -p3_2[1]*p3_1[1], -p3_2[1]],
                    [p4_1[0], p4_1[1], 1,0,0,0, -p4_2[0]*p4_1[0], -p4_2[0]*p4_1[1], -p4_2[0]],
                    [0,0,0, p4_1[0], p4_1[1], 1, -p4_2[1]*p4_1[0], -p4_2[1]*p4_1[1], -p4_2[1]]])

    ATA = A.transpose() @ A
    w, v = linalg.eig(ATA)
    h = v[:,len(w)-1].reshape((3,3))
    return h



def homography_mapping(putative_matches, kp1, kp2):
    best_average_res = 0
    best = None
    bestcount =  -1
    bestset = []

    for trial in range(100):

        inliers = 0
        average_res = 0

        samples = random.sample(putative_matches,4)
        h = get_h(putative_matches, kp1, kp2, samples)

        for pair in putative_matches:
            pt_1 = kp1[pair[0]].pt
            pt_2 = kp2[pair[1]].pt
            homo_pt_1 = np.append(pt_1,[1])
            pred_homo_right = h @ homo_pt_1
            pred_right = np.array([pred_homo_right[0]/pred_homo_right[2], pred_homo_right[1]/pred_homo_right[2]])
            diff = np.subtract(pt_2, pred_right)
            error = diff @ diff.transpose()

            if error < 10:
                average_res += error
                inliers+=1

        if inliers > bestcount:
            best = h
            bestcount = inliers
            bestset = samples
            best_average_res = average_res

    best_average_res = best_average_res / bestcount
    print("best_average_res", best_average_res)

    return best, bestcount, bestset



#..........................Compute Distance Matrix....................................
def Compute_and_Store_Distance_Matrix(kp1, kp2, des1, des2, file_num):
    #Normalizing:
    for i in range(0, len(des1)):
        mean = np.mean(des1[i])
        des1[i] = des1[i] - mean
        des1[i] = des1[i] / np.std(des1[i])

    for i in range(0, len(des2)):
        mean = np.mean(des2[i])
        des2[i] = des2[i] - mean
        des2[i] = des2[i] / np.std(des2[i])

    distance_matrix = np.zeros((len(kp1), len(kp2)))

    for i in range(0, len(kp1)):
        for j in range(0, len(kp2)):
            distanceCurr = 0
            for k in range(0, len(des1[i])):
                distanceCurr += (des1[i][k] - des2[j][k]) ** 2
            distanceCurr = distanceCurr ** (1/2)
            distance_matrix[i][j] = distanceCurr
    #Storing the matrix to a npy file
    np.save("outfile" + file_num + ".npy", distance_matrix)
    return



def Load_Distance_Matrix(file_num):
    return np.load("outfile" + file_num + ".npy")



def get_putative_matches(kp1, kp2, distance_matrix):
    distance_dict = {}
    coor_dict = {}
    pair_dict = {}


    for i in range(0, distance_matrix.shape[0]):
        distance_dict.setdefault(i, distance_matrix[i].min())
        pair_dict.setdefault(i, np.argmin(distance_matrix[i]))
        coor_dict.setdefault(i, kp2[np.argmin(distance_matrix[i])].pt)

    distance_list = sorted(distance_dict.items(), key=operator.itemgetter(1))

    #select the top 200 min distances:
    putative_matches = {}

    for i in range(0, len(distance_list)):
        #print(distance_dict.get(key))
        #print("distance: ", distance_list[i][1])
        putative_matches.setdefault(distance_list[i][0], distance_list[i][1])
        if i == 13:
            break

    delKeys = []
    for key in coor_dict:
        if key not in putative_matches:
            delKeys.append(key)

    for key in delKeys:
        del coor_dict[key]
        del pair_dict[key]

    pair_list = []
    for i in pair_dict:
       k = (i,pair_dict[i])
       pair_list.append(k)

    return pair_list


def draw_matches(kp1, kp2, gray1, gray2, best_inlier_inx):
    matches = []
    dist = 0
    for i in range(0, len(best_inlier_inx)):
        match = cv2.DMatch(best_inlier_inx[i][0], best_inlier_inx[i][1], dist)
        matches.append(match)


    img3 = cv2.drawMatches(gray1,kp1,gray2,kp2,matches, None)
    plt.imshow(img3),plt.show()
    return
