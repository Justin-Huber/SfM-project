import cv2

def populateKeypointsAndDescriptors():
    sift = cv2.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
