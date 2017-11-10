import cv2
import numpy as np



def Q(img1, img2):
    M, N = img1.shape
    return M

def mse(img1, img2):
    M, N = img1.shape
    sum_ = 0
    for i in range(M):
        for j in range(N):
            sum_ += (img1[i][j] - img2[i][j])**2/(M*N)

    return sum_

img = cv2.imread('imgs/Image42.png', 0)
blur = cv2.imread('imgs/blur.png', 0)
print(mse(img,blur))