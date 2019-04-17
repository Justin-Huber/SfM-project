import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import linalg
import operator
import time


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 15, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 15, color, -1)
    return img1, img2


def draw_epipolar(img1, img2, F, pts1, pts2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()


def _geometric_verification_impl(i, j):
    """

    :param i:
    :param j:
    :return:
    """

    # globals.pipeline._geometric_verification_impl(i, j)
    time.sleep(0.001)


#.....................................RANSAC..........................................
def get_f(putative_matches, kp1, kp2, samples):

    p1_1 = kp1[samples[0][0]].pt
    p1_2 = kp2[samples[0][1]].pt
    p2_1 = kp1[samples[1][0]].pt
    p2_2 = kp2[samples[1][1]].pt
    p3_1 = kp1[samples[2][0]].pt
    p3_2 = kp2[samples[2][1]].pt
    p4_1 = kp1[samples[3][0]].pt
    p4_2 = kp2[samples[3][1]].pt

    #added for f, bc we need 8 correspondences
    p5_1 = kp1[samples[4][0]].pt
    p5_2 = kp2[samples[4][1]].pt
    p6_1 = kp1[samples[5][0]].pt
    p6_2 = kp2[samples[5][1]].pt
    p7_1 = kp1[samples[6][0]].pt
    p7_2 = kp2[samples[6][1]].pt
    p8_1 = kp1[samples[7][0]].pt
    p8_2 = kp2[samples[7][1]].pt

    '''...
    #for h matrix
    A = np.array([[p1_1[0], p1_1[1], 1,0,0,0, -p1_2[0]*p1_1[0], -p1_2[0]*p1_1[1], -p1_2[0]],
                    [0,0,0, p1_1[0], p1_1[1], 1, -p1_2[1]*p1_1[0], -p1_2[1]*p1_1[1], -p1_2[1]],
                    [p2_1[0], p2_1[1], 1,0,0,0, -p2_2[0]*p2_1[0], -p2_2[0]*p2_1[1], -p2_2[0]],
                    [0,0,0, p2_1[0], p2_1[1], 1, -p2_2[1]*p2_1[0], -p2_2[1]*p2_1[1], -p2_2[1]],
                    [p3_1[0], p3_1[1], 1,0,0,0, -p3_2[0]*p3_1[0], -p3_2[0]*p3_1[1], -p3_2[0]],
                    [0,0,0, p3_1[0], p3_1[1], 1, -p3_2[1]*p3_1[0], -p3_2[1]*p3_1[1], -p3_2[1]],
                    [p4_1[0], p4_1[1], 1,0,0,0, -p4_2[0]*p4_1[0], -p4_2[0]*p4_1[1], -p4_2[0]],
                    [0,0,0, p4_1[0], p4_1[1], 1, -p4_2[1]*p4_1[0], -p4_2[1]*p4_1[1], -p4_2[1]]])
    ...'''
    #for f matrix
    A = np.array([[p1_1[0]*p1_2[0], p1_1[1]*p1_2[0], p1_2[0], p1_1[0]*p1_2[1] , p1_1[1]*p1_2[1] ,p1_2[1], p1_1[0], p1_1[1], 1],
                    [p2_1[0]*p2_2[0], p2_1[1]*p2_2[0], p2_2[0], p2_1[0]*p2_2[1] , p2_1[1]*p2_2[1] ,p2_2[1], p2_1[0], p2_1[1], 1],
                    [p3_1[0]*p3_2[0], p3_1[1]*p3_2[0], p3_2[0], p3_1[0]*p3_2[1] , p3_1[1]*p3_2[1] ,p3_2[1], p3_1[0], p3_1[1], 1],
                    [p4_1[0]*p4_2[0], p4_1[1]*p4_2[0], p4_2[0], p4_1[0]*p4_2[1] , p4_1[1]*p4_2[1] ,p4_2[1], p4_1[0], p4_1[1], 1],
                    [p5_1[0]*p5_2[0], p5_1[1]*p5_2[0], p5_2[0], p5_1[0]*p5_2[1] , p5_1[1]*p5_2[1] ,p5_2[1], p5_1[0], p5_1[1], 1],
                    [p6_1[0]*p6_2[0], p6_1[1]*p6_2[0], p6_2[0], p6_1[0]*p6_2[1] , p6_1[1]*p6_2[1] ,p6_2[1], p6_1[0], p6_1[1], 1],
                    [p7_1[0]*p7_2[0], p7_1[1]*p7_2[0], p7_2[0], p7_1[0]*p7_2[1] , p7_1[1]*p7_2[1] ,p7_2[1], p7_1[0], p7_1[1], 1],
                    [p8_1[0]*p8_2[0], p8_1[1]*p8_2[0], p8_2[0], p8_1[0]*p8_2[1] , p8_1[1]*p8_2[1] ,p8_2[1], p8_1[0], p8_1[1], 1]])

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

        #h:
        #samples = random.sample(putative_matches,4)
        #f:
        samples = random.sample(putative_matches,8)

        f = get_f(putative_matches, kp1, kp2, samples)

        for pair in putative_matches:
            pt_1 = kp1[pair[0]].pt
            pt_2 = kp2[pair[1]].pt
            homo_pt_1 = np.append(pt_1,[1])
            pred_homo_right = f @ homo_pt_1
            pred_right = np.array([pred_homo_right[0]/pred_homo_right[2], pred_homo_right[1]/pred_homo_right[2]])
            diff = np.subtract(pt_2, pred_right)
            error = diff @ diff.transpose()

            if error < 10:
                #average_res += error
                inliers+=1

        if inliers > bestcount:
            best = f
            bestcount = inliers
            bestset = samples
            #best_average_res = average_res

    #best_average_res = best_average_res / bestcount
    #print("best_average_res", best_average_res)

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
